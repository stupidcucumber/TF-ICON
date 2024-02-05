FROM python:3.10-bullseye

WORKDIR /app
ADD requirements.txt ./
RUN pip3 install --no-cache -r requirements.txt

COPY . ./
CMD ["python3", "scripts/main_tf_icon.py", "--ckpt", "<path/to/model.ckpt/>", \
                                "--root", "./inputs/same_domain",            \
                                "--domain", "same",                          \
                                "--dpm_steps", "20",                         \
                                "--dpm_order", "2",                          \
                                "--scale", "2.5",                            \
                                "--tau_a", "0.4",                            \
                                "--tau_b", "0.8",                            \
                                "--outdir", "./outputs",                     \
                                "--gpu", "cuda:0"                            \
                                "--seed", "3407"]