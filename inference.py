import argparse, os
import torch
import re, json
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from einops import rearrange, repeat
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
import cv2
import time
from datetime import datetime
from main.tficon.ldm.util import instantiate_from_config, process_img, load_prompt_embedding, load_model_from_config
from main.tficon.ldm.models.diffusion.ddim import DDIMSampler
from main.tficon.ldm.models.diffusion.dpm_solver import DPMSolverSampler
from huggingface_hub import hf_hub_download


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, nargs="?", 
                        default="a professional photograph of a doggy, ultra realistic",
                        help="the prompt to render"
                        )

    parser.add_argument("--init-img", type=str, nargs="?", help="path to the input image")

    parser.add_argument("--ref-img", type=list, nargs="?", help="path to the input image")
    
    parser.add_argument("--seg", type=str, nargs="?", help="path to the input image")
        
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="./outputs")

    parser.add_argument("--dpm_steps", type=int, default=20, help="number of ddim sampling steps")

    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")

    parser.add_argument("--C", type=int, default=4, help="latent channels")
    
    parser.add_argument("--f", type=int, default=16, help="downsampling factor, most often 8 or 16")

    parser.add_argument("--n_samples", type=int, default=1, 
                        help="how many samples to produce for each given prompt. A.k.a batch size")

    parser.add_argument("--scale", type=float, default=2.5, 
                        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))")
    
    parser.add_argument("--config", type=str, default="./configs/stable-diffusion/v2-inference.yaml",
                        help="path to config which constructs model")
    
    parser.add_argument("--ckpt", type=str, default="./ckpt/v2-1_512-ema-pruned.ckpt", help="path to checkpoint of model")
    
    parser.add_argument("--seed", type=int, default=3407, help="the seed (for reproducible sampling)")
    
    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
    
    parser.add_argument("--root", type=str, help="", default='./inputs/cross_domain') 
    
    parser.add_argument("--domain", type=str, help="", default='cross') 
    
    parser.add_argument("--dpm_order", type=int, help="", choices=[1, 2, 3], default=2) 
    
    parser.add_argument("--tau_a", type=float, help="", default=0.4)
      
    parser.add_argument("--tau_b", type=float, help="", default=0.8)
          
    parser.add_argument("--gpu", type=str, help="", default='cuda:0')

    parser.add_argument('--masks', type=str, help='Path to the json file containing masks.')
    
    opt = parser.parse_args()

    return opt

def choose_mask(config: dict) -> list[str, str]:
    decision = np.random.choice(config)
    return decision['image'], decision['mask']

def main():
    print('Welcome to the inference script! In a moment we will start...')
    opt = parse_arguments()
    opt.mask = []
    output_folder = 'outputs'

    with open(opt.masks, 'r') as config_f:
        masks_config = json.load(config_f)

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    hf_hub_download(repo_id="stabilityai/stable-diffusion-2-1-base", 
                    local_dir='ckpt', 
                    filename="v2-1_512-ema-pruned.ckpt")
     
    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")

    # The scale used in the paper
    if opt.domain == 'cross':
        opt.scale = 5.0
        file_name = "cross_domain"
    elif opt.domain == 'same':
        opt.scale = 2.5
        file_name = "same_domain"
    else:
        raise ValueError("Invalid domain")

    batch_size = opt.n_samples
    sample_path = os.path.join(output_folder, file_name)
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, opt.gpu)    
    model = model.to(device)
    sampler = DPMSolverSampler(model)
    
    for subdir, _, files in os.walk(opt.root):
        for file in files:
            torch.cuda.empty_cache()
            file_path = os.path.join(subdir, file)
            print('Start processing file: ',file_path, flush=True)
            result = re.search(r'./inputs/[^/]+/(.+)/bg\d+\.', file_path)
            if result:
                prompt = 'tomato on a plant, photorealistic'
                
            if file_path.endswith('.jpg') or file_path.endswith('.jpeg') or file_path.endswith('.png'):
                if file.startswith('bg'):
                    opt.init_img = file_path
                elif file.startswith('mask'):
                    opt.mask.append(file_path)
                    
            if file == files[-1]:
                seed_everything(opt.seed)

                for mask in opt.mask:
                    opt.ref_image, opt.seg = choose_mask(config=masks_config)
                    img = cv2.imread(mask, 0)
                    # Threshold the image to create binary image
                    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                    # Find the contours of the white region in the image
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    # Find the bounding rectangle of the largest contour
                    x, y, new_w, new_h = cv2.boundingRect(contours[0])
                    # Calculate the center of the rectangle
                    center_x = x + new_w / 2
                    center_y = y + new_h / 2
                    # Calculate the percentage from the top and left
                    center_row_from_top = round(center_y / 512, 2)
                    center_col_from_left = round(center_x / 512, 2)

                    aspect_ratio = new_h / new_w
                    
                    if aspect_ratio > 1:  
                        scale = new_w * aspect_ratio / 256  
                        scale = new_h / 256
                    else:  
                        scale = new_w / 256
                        scale = new_h / (aspect_ratio * 256) 
                        
                    scale = round(scale, 2)
                    
                    # =============================================================================================
            
                    assert prompt is not None
                    data = [batch_size * [prompt]]
                    
                    # read background image              
                    assert os.path.isfile(opt.init_img)
                    init_image, target_width, target_height = process_img(opt.init_img, scale)
                    init_image = repeat(init_image.to(device), '1 ... -> b ...', b=batch_size)
                    save_image = init_image.clone()

                    # read foreground image and its segmentation map
                    ref_image, width, height, segmentation_map  = process_img(opt.ref_img, scale, seg=opt.seg, target_size=(target_width, target_height))
                    ref_image = repeat(ref_image.to(device), '1 ... -> b ...', b=batch_size)

                    segmentation_map_orig = repeat(torch.tensor(segmentation_map)[None, None, ...].to(device), '1 1 ... -> b 4 ...', b=batch_size)
                    segmentation_map_save = repeat(torch.tensor(segmentation_map)[None, None, ...].to(device), '1 1 ... -> b 3 ...', b=batch_size)
                    segmentation_map = segmentation_map_orig[:, :, ::8, ::8].to(device)

                    top_rr = int((0.5*(target_height - height))/target_height * init_image.shape[2])  # xx% from the top
                    bottom_rr = int((0.5*(target_height + height))/target_height * init_image.shape[2])  
                    left_rr = int((0.5*(target_width - width))/target_width * init_image.shape[3])  # xx% from the left
                    right_rr = int((0.5*(target_width + width))/target_width * init_image.shape[3]) 

                    center_row_rm = int(center_row_from_top * target_height)
                    center_col_rm = int(center_col_from_left * target_width)

                    step_height2, remainder = divmod(height, 2)
                    step_height1 = step_height2 + remainder
                    step_width2, remainder = divmod(width, 2)
                    step_width1 = step_width2 + remainder
                        
                    # compositing in pixel space for same-domain composition
                    save_image[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2] \
                            = save_image[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2].clone() \
                            * (1 - segmentation_map_save[:, :, top_rr:bottom_rr, left_rr:right_rr]) \
                            + ref_image[:, :, top_rr:bottom_rr, left_rr:right_rr].clone() \
                            * segmentation_map_save[:, :, top_rr:bottom_rr, left_rr:right_rr]

                    # save the mask and the pixel space composited image
                    save_mask = torch.zeros_like(init_image) 
                    save_mask[:, :, center_row_rm - step_height1:center_row_rm + step_height2, center_col_rm - step_width1:center_col_rm + step_width2] = 1

                    image = Image.fromarray(((save_image/torch.max(save_image.max(), abs(save_image.min())) + 1) * 127.5)[0].permute(1,2,0).to(dtype=torch.uint8).cpu().numpy())
                    image.save('./outputs/cp_bg_fg.jpg')
                    print('Saved pre-processed image.', flush=True)

                    precision_scope = autocast if opt.precision == "autocast" else nullcontext
                    
                    # image composition
                    print('Starting image composition...', flush=True)
                    with torch.no_grad():
                        with precision_scope("cuda"):
                            for prompts in data:
                                print(prompts)
                                c, uc, inv_emb = load_prompt_embedding(model, opt, prompts, inv=True)
                                
                                if opt.domain == 'same': # same domain
                                    init_image = save_image
                                
                                T1 = time.time()
                                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  
                                
                                # ref's location in ref image in the latent space
                                top_rr = int((0.5*(target_height - height))/target_height * init_latent.shape[2])  
                                bottom_rr = int((0.5*(target_height + height))/target_height * init_latent.shape[2])  
                                left_rr = int((0.5*(target_width - width))/target_width * init_latent.shape[3])  
                                right_rr = int((0.5*(target_width + width))/target_width * init_latent.shape[3]) 
                                                        
                                new_height = bottom_rr - top_rr
                                new_width = right_rr - left_rr
                                
                                step_height2, remainder = divmod(new_height, 2)
                                step_height1 = step_height2 + remainder
                                step_width2, remainder = divmod(new_width, 2)
                                step_width1 = step_width2 + remainder
                                
                                center_row_rm = int(center_row_from_top * init_latent.shape[2])
                                center_col_rm = int(center_col_from_left * init_latent.shape[3])
                                
                                param = [max(0, int(center_row_rm - step_height1)), 
                                        min(init_latent.shape[2] - 1, int(center_row_rm + step_height2)),
                                        max(0, int(center_col_rm - step_width1)), 
                                        min(init_latent.shape[3] - 1, int(center_col_rm + step_width2))]
                                
                                ref_latent = model.get_first_stage_encoding(model.encode_first_stage(ref_image))
                            
                                shape = [init_latent.shape[1], init_latent.shape[2], init_latent.shape[3]]
                                z_enc, _ = sampler.sample(steps=opt.dpm_steps,
                                                        inv_emb=inv_emb,
                                                        unconditional_conditioning=uc,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        eta=opt.ddim_eta,
                                                        order=opt.dpm_order,
                                                        x_T=init_latent,
                                                        width=width,
                                                        height=height,
                                                        DPMencode=True,
                                                        )
                                
                                z_ref_enc, _ = sampler.sample(steps=opt.dpm_steps,
                                                            inv_emb=inv_emb,
                                                            unconditional_conditioning=uc,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            eta=opt.ddim_eta,
                                                            order=opt.dpm_order,
                                                            x_T=ref_latent,
                                                            DPMencode=True,
                                                            width=width,
                                                            height=height,
                                                            ref=True,
                                                            )
                                
                                samples_orig = z_enc.clone()

                                # inpainting in XOR region of M_seg and M_mask
                                z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
                                    = z_enc[:, :, param[0]:param[1], param[2]:param[3]] \
                                    * segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr] \
                                    + torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=device) \
                                    * (1 - segmentation_map[:, :, top_rr:bottom_rr, left_rr:right_rr])

                                samples_for_cross = samples_orig.clone()
                                samples_ref = z_ref_enc.clone()
                                samples = z_enc.clone()

                                # noise composition
                                if opt.domain == 'cross': 
                                    samples[:, :, param[0]:param[1], param[2]:param[3]] = torch.randn((1, 4, bottom_rr - top_rr, right_rr - left_rr), device=device) 
                                    # apply the segmentation mask on the noise
                                    samples[:, :, param[0]:param[1], param[2]:param[3]] \
                                            = samples[:, :, param[0]:param[1], param[2]:param[3]].clone() \
                                            * (1 - segmentation_map[:, :, top_rr: bottom_rr, left_rr: right_rr]) \
                                            + z_ref_enc[:, :, top_rr: bottom_rr, left_rr: right_rr].clone() \
                                            * segmentation_map[:, :, top_rr: bottom_rr, left_rr: right_rr]
                                
                                mask = torch.zeros_like(z_enc, device=device)
                                mask[:, :, param[0]:param[1], param[2]:param[3]] = 1
                                                    
                                samples, _ = sampler.sample(steps=opt.dpm_steps,
                                                            inv_emb=inv_emb,
                                                            conditioning=c,
                                                            batch_size=opt.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=opt.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=opt.ddim_eta,
                                                            order=opt.dpm_order,
                                                            x_T=[samples_orig, samples.clone(), samples_for_cross, samples_ref, samples, init_latent],
                                                            width=width,
                                                            height=height,
                                                            segmentation_map=segmentation_map,
                                                            param=param,
                                                            mask=mask,
                                                            target_height=target_height, 
                                                            target_width=target_width,
                                                            center_row_rm=center_row_from_top,
                                                            center_col_rm=center_col_from_left,
                                                            tau_a=opt.tau_a,
                                                            tau_b=opt.tau_b,
                                                            )
                                    
                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                                
                                T2 = time.time()
                                print('Running Time: %s s' % ((T2 - T1)), flush=True)
                                
                                for x_sample in x_samples:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))

                                    path = os.path.join(subdir, '%s' % opt.init_img.split('/')[-1])
                                    print('Path to the sample: ', path, flush=True)
                                    temp_path = os.path.join(subdir, 'temp_%s.png' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                                    img.save(path)
                                    img.save(temp_path)
                                    base_count += 1
                
                path = os.path.join(sample_path, f"{base_count:05}_{prompts[0]}.png")
                print('Saving the final file: ', path, flush=True)
                img.save(path)
                opt.mask.clear()

                del x_samples, samples, z_enc, z_ref_enc, samples_orig, samples_for_cross, samples_ref, mask, x_sample, img, c, uc, inv_emb
                del param, segmentation_map, top_rr, bottom_rr, left_rr, right_rr, target_height, target_width, center_row_rm, center_col_rm
                del init_image, init_latent, save_image, ref_image, ref_latent, prompt, prompts, data, binary, contours

    print(f"Your samples are ready and waiting for you here: \n{sample_path} \nEnjoy.")


if __name__ == "__main__":
    main()

