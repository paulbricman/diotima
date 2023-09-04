#!/bin/bash

# echo "[*] Creating TPU Pod...";
# gcloud compute tpus tpu-vm create tpu-test-0 \
#        --zone=us-central2-b \
#        --accelerator-type=v4-64 \
#        --version=tpu-ubuntu2204-base;

echo "[*] Copying codebase to tpu-test-0";
gcloud compute tpus tpu-vm scp ../wandb.key tpu-test-0:~/ \
       --worker=all \
       --zone=us-central2-b;
gcloud compute tpus tpu-vm scp ../setup.py tpu-test-0:~/ \
       --worker=all \
       --zone=us-central2-b;
gcloud compute tpus tpu-vm scp ../environment.yml tpu-test-0:~/ \
       --worker=all \
       --zone=us-central2-b;
gcloud compute tpus tpu-vm scp ../diotima/ tpu-test-0:~/diotima \
       --worker=all \
       --zone=us-central2-b \
       --recurse;
gcloud compute tpus tpu-vm scp ./optimize.py tpu-test-0:~/ \
       --worker=all \
       --zone=us-central2-b;

echo "[*] Procuring tpu-test-0...";
gcloud compute tpus tpu-vm ssh tpu-test-0 \
       --worker=all \
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

echo "[*] Running master script on tpu-test-0..."; \
gcloud compute tpus tpu-vm ssh tpu-test-0 \
       --worker=all \
       --zone=us-central2-b \
       --command='export PATH=~/miniconda/bin:$PATH; \
       source $HOME/miniconda/bin/activate; \
       conda activate diotima; \
       JAX_DISABLE_JIT=1 python optimize.py;';

echo "[*] Cleaning up tpu-vm-"$i"..."; \
gcloud compute tpus tpu-vm ssh tpu-test-0 \
       --worker=all \
       --zone=us-central2-b \
       --command="rm -rf ~/*" ;

# echo "[*] Deleting tpu-test-0...";
# gcloud compute tpus tpu-vm delete tpu-test-0 --zone=us-central2-b;

# Stop all VMs.
# for i in {0..0}; do echo "[*] Stopping tpu-vm-"$i"..."; gcloud compute tpus tpu-vm stop tpu-test-$i --zone=us-central2-b; done

# Start all VMs.
# for i in {0..0}; do echo "[*] Starting tpu-vm-"$i"..."; gcloud compute tpus tpu-vm start tpu-test-$i --zone=us-central2-b; done

# Fetch checkpoints.
# echo "[*] Copying codebase to tpu-vm-"$i"..."; gcloud compute tpus tpu-vm scp tpu-test-0:~/config.pickle ./ --zone=us-central2-b
