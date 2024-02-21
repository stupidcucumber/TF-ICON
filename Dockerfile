FROM python:3.10-bullseye

WORKDIR /app
ADD requirements.txt ./
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip3 install --no-cache -r requirements.txt

COPY . ./
CMD ["python3", "inference.py", "--ckpt", "./ckpt/v2-1_512-ema-pruned.ckpt", "--root", "./inputs/same_domain", "--domain", "same", "--dpm_steps", "20", "--dpm_order", "2", "--scale", "2.5", "--tau_a", "0.4", "--tau_b", "0.8", "--outdir", "./outputs", "--gpu", "cuda:0", "--seed", "3407", "--masks", "inputs/input_foregrounds/tomatoes/config.json"]