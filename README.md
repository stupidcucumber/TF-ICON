# How to run?
To run the script I suggest using docker:
```
docker-compose up
```
On run, the container will obtain access to two volumes, which are 'inputs' and 'outputs' folder on the host computer. There will be out inputs and outputs of the model.

# How to create inputs?
The folder inputs contains folder 'same_domain', which means all inputs samples contain foreground and background of the same style. There you create a new folder and name it as a prompt. 

Name conventions for the inputs:
- 'bg\*.png', name starts with *bg* means that it is a background image.
- 'fg\*.png', name starts with *fg* means that it is a foreground image.
- 'fg\*_mask.png', name starts with *fg* and ends with *mask* means it is a mask for the foreground image.
- 'mask_bg_fg_\*.png', name starts with *mask_bg_fg* means it is a mask for the box in the background to paste foreground image in.

For now, you can generate inputs only manually, but for generating masks we provide more convenient way of labeling – *prepare_input.py* script. It resizes background to 512x512 and saves masks you have chosen during labeling to the input folder.

# How I can obtain results?
Results are being stored in the outputs folder.