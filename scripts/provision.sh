#!/bin/bash

echo "[*] Copying codebase to tpu-test-0";
gcloud alpha compute tpus tpu-vm scp ../diotima/ tpu-test-0:~/diotima \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b \
       --recurse;
gcloud alpha compute tpus tpu-vm scp ./optimize.py tpu-test-0:~/ \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b;
gcloud alpha compute tpus tpu-vm scp ../wandb.key tpu-test-0:~/ \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b;
gcloud alpha compute tpus tpu-vm scp ../setup.py tpu-test-0:~/ \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b;
gcloud alpha compute tpus tpu-vm scp ../environment.yml tpu-test-0:~/ \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b;
# gcloud alpha compute tpus tpu-vm scp ./test_devices.py tpu-test-0:~/ \
#        --worker=all \
#        --batch-size=8 \
#        --zone=us-central2-b;

echo "[*] Procuring tpu-test-0...";
gcloud alpha compute tpus tpu-vm ssh tpu-test-0 \
       --worker=all \
       --batch-size=8 \
       --zone=us-central2-b \
       --command='wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O ~/miniconda.sh; \
       chmod +x ~/miniconda.sh; \
       ~/miniconda.sh -b -p ~/miniconda; \
       export PATH=~/miniconda/bin:$PATH; \
       conda init bash; \
       conda update -n base -c defaults conda -y; \
       source $HOME/miniconda/bin/activate; \
       conda env create -f environment.yml; \
       conda activate diotima; \
       wandb login $(cat wandb.key); \
       pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html; \
       pip install einops;';