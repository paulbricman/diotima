#!/bin/bash

# Copy codebase on all VMs.
for i in {0..0}; do echo "[*] Copying codebase to tpu-vm-"$i"..."; \
                    gcloud compute tpus tpu-vm scp ../wandb.key tpu-test-$i:~/ \
                           --worker=all \
                           --zone=us-central2-b; \
                    gcloud compute tpus tpu-vm scp ../setup.py tpu-test-$i:~/ \
                           --worker=all \
                           --zone=us-central2-b; \
                    gcloud compute tpus tpu-vm scp ../environment.yml tpu-test-$i:~/ \
                           --worker=all \
                           --zone=us-central2-b; \
                    gcloud compute tpus tpu-vm scp ../diotima/ tpu-test-$i:~/diotima \
                           --worker=all \
                           --zone=us-central2-b \
                           --recurse; \
                    gcloud compute tpus tpu-vm scp ./optimize.py tpu-test-$i:~/ \
                           --worker=all \
                           --zone=us-central2-b; \
    done;

# Run VMs (w/ local env var).
for i in {0..0}; do echo "[*] Running master script on tpu-vm-"$i"..."; \
                    gcloud compute tpus tpu-vm ssh tpu-test-$i \
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
                          python optimize.py' ; \
    done;

# Clean up VMs.
for i in {0..0}; do echo "[*] Cleansing tpu-vm-"$i"..."; \
                    gcloud compute tpus tpu-vm ssh tpu-test-$i \
                           --worker=all \
                           --zone=us-central2-b \
                           --command="rm -rf ~/*" ; \
    done;

# Create TPU VMs.
# for i in {0..0}; do echo "[*] Starting tpu-vm-"$i"..."; \
#                     gcloud compute tpus tpu-vm create tpu-test-$i \
#                            --zone=us-central2-b \
#                            --accelerator-type=v4-64 \
#                            --version=tpu-ubuntu2204-base; \
#     done;

# Fetch checkpoints.
# echo "[*] Copying codebase to tpu-vm-"$i"..."; gcloud compute tpus tpu-vm scp tpu-test-0:~/config.pickle ./ --zone=us-central2-b

# Delete all VMs.
# for i in {0..0}; do echo "[*] Deleting tpu-vm-"$i"..."; gcloud compute tpus tpu-vm delete tpu-test-$i --zone=us-central2-b; done

# Stop all VMs.
# for i in {0..0}; do echo "[*] Stopping tpu-vm-"$i"..."; gcloud compute tpus tpu-vm stop tpu-test-$i --zone=us-central2-b; done

# Start all VMs.
# for i in {0..0}; do echo "[*] Starting tpu-vm-"$i"..."; gcloud compute tpus tpu-vm start tpu-test-$i --zone=us-central2-b; done