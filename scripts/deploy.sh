#!/bin/bash

# 1. Create TPU VMs.
for i in {1..1}; do echo "[*] Starting tpu-vm-"$i"..."; gcloud compute tpus tpu-vm create tpu-test-$i --zone=europe-west4-a --accelerator-type=v3-8 --version=tpu-ubuntu2204-base; done

# 2. Set (different) process_ids as env var on all VMs.

# 3. Copy codebase on all VMs.

# 4. Run master script on all VMs.

# 5. Stop all VMs.
for i in {1..1}; do echo "[*] Stopping tpu-vm-"$i"..."; gcloud compute tpus tpu-vm stop tpu-test-$i --zone=europe-west4-a; done

# 6. Delete all VMs.
for i in {1..1}; do echo "[*] Deleting tpu-vm-"$i"..."; gcloud compute tpus tpu-vm delete tpu-test-$i --zone=europe-west4-a; done
