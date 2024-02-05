FROM python

WORKDIR /app
ADD requirements.txt ./

COPY . ./
CMD ["python", "scripts/main_tf_icon.py", "--ckpt", "<path/to/model.ckpt/>", \
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