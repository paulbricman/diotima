#!/bin/bash

# 1. Create TPU VMs.
for i in {0..1}; do echo "[*] Starting tpu-vm-"$i"..."; \
                    gcloud compute tpus tpu-vm create tpu-test-$i \
                           --zone=us-central1-f \
                           --accelerator-type=v2-8 \
                           --version=tpu-ubuntu2204-base; \
    done

# 2. Copy codebase on all VMs.
for i in {1..1}; do echo "[*] Copying codebase to tpu-vm-"$i"..."; \
                    # gcloud compute tpus tpu-vm scp ../wandb.key tpu-test-$i:~/ \
                    #        --zone=us-central1-f; \
                    # gcloud compute tpus tpu-vm scp ../setup.py tpu-test-$i:~/ \
                    #        --zone=us-central1-f; \
                    # gcloud compute tpus tpu-vm scp ../environment.yml tpu-test-$i:~/ \
                    #        --zone=us-central1-f; \
                    gcloud compute tpus tpu-vm scp ../diotima/ tpu-test-$i:~/diotima \
                           --zone=us-central1-f \
                           --recurse; \
                    gcloud compute tpus tpu-vm scp ./optimize.py tpu-test-$i:~/ \
                           --zone=us-central1-f; \
    done

# 3. Run VMs (w/ local env var).
for i in {0..1}; do echo "[*] Running master script on tpu-vm-"$i"..."; \
                    gcloud compute tpus tpu-vm ssh tpu-test-$i \
                           --zone=us-central1-f \
                           --command="wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh -O ~/miniconda.sh; \
                          chmod +x ~/miniconda.sh; \
                          ~/miniconda.sh -b -p ~/miniconda; \
                          export PATH=\"~/miniconda/bin:$PATH\"; \
                          conda init bash; \
                          source \"$HOME/miniconda/bin/activate\"; \
                          conda env create -f environment.yml; \
                          conda activate diotima; \
                          mkdir ckpts; \
                          export JAX_COORD_ADDR=10.128.0.11:8888; \
                          export JAX_NUM_HOSTS=2; \
                          export JAX_PROCESS_ID=$i; \
                          wandb login $\(cat wandb.key\); \
                          python optimize.py" & \
    done

# 4. Fetch checkpoints.
echo "[*] Copying codebase to tpu-vm-"$i"..."; gcloud compute tpus tpu-vm scp tpu-test-0:~/config.pickle ./ --zone=us-central1-f

# 5. Delete all VMs.
for i in {0..0}; do echo "[*] Deleting tpu-vm-"$i"..."; gcloud compute tpus tpu-vm delete tpu-test-$i --zone=us-central1-f; done

---

# Clean up VMs without deleting them (i.e. hold onto VMs).
for i in {0..1}; do echo "[*] Running master script on tpu-vm-"$i"..."; \
                    gcloud compute tpus tpu-vm ssh tpu-test-$i \
                           --zone=us-central1-f \
                           --command="rm -rf ./*" & \
    done

# pip install --upgrgade jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
