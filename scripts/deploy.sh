#!/bin/bash

# 1. Create TPU VMs.
for i in {0..0}; do echo "[*] Starting tpu-vm-"$i"..."; \
                    gcloud compute tpus tpu-vm create tpu-test-$i \
                           --zone=us-central1-f \
                           --accelerator-type=v2-8 \
                           --version=tpu-ubuntu2204-base; \
    done

# 2. Copy codebase on all VMs.
for i in {0..0}; do echo "[*] Copying codebase to tpu-vm-"$i"..."; \
                    gcloud compute tpus tpu-vm scp ../environment.yml tpu-test-$i:~/ \
                           --zone=us-central1-f; \
    done

gcloud compute tpus tpu-vm scp ../diotima/ tpu-test-$i:~/diotima \
       --zone=us-central1-f \
       --recurse; \
    gcloud compute tpus tpu-vm scp ../setup.py tpu-test-$i:~/diotima \
           --zone=us-central1-f; \

# 3. Run provisioning script on all VMs (w/ local env var).



# 4. Run optimization script on all VMs (w/ local env var).

# 5. Fetch checkpoints.

# 6. Delete all VMs.
    for i in {0..0}; do echo "[*] Deleting tpu-vm-"$i"..."; gcloud compute tpus tpu-vm delete tpu-test-$i --zone=us-central1-f; done

# Scraps

gcloud compute tpus tpu-vm list --zone=us-central1-f

gcloud compute tpus tpu-vm ssh tpu-test --zone=us-central1-f

gcloud compute tpus tpu-vm describe tpu-test-0  --zone=us-central1-f

gcloud compute tpus tpu-vm ssh tpu-test-0 --zone=us-central1-f

for i in {0..0}; do echo "[*] Printing process ID to tpu-vm-"$i"..."; \
                    gcloud compute tpus tpu-vm ssh tpu-test-$i \
                           --zone=us-central1-f \
                           --command="echo 'hello world'"; \
    done

wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O ~/miniconda.sh
chmod +x ~/miniconda.sh
~/miniconda.sh -b -p ~/miniconda
export PATH="~/miniconda/bin:$PATH"
conda init bash
conda env create -f environment.yml
conda activate diotima
mkdir ckpts
