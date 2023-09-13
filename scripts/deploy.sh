echo "[*] Running master script on tpu-test-0..."; \
gcloud compute tpus tpu-vm ssh tpu-test-0 \
       --worker=all \
       --zone=us-central2-b \
       --command='export PATH=~/miniconda/bin:$PATH; \
       source $HOME/miniconda/bin/activate; \
       conda activate diotima; \
       python optimize.py;';

# echo "[*] Cleaning up tpu-vm-..."; \
# gcloud compute tpus tpu-vm ssh tpu-test-0 \
#        --worker=all \
#        --zone=us-central2-b \
#        --command="rm -rf ~/*" ;

# echo "[*] Cleaning up tpu-vm-..."; \
# gcloud compute tpus tpu-vm ssh tpu-test-0 \
#        --worker=all \
#        --zone=us-central2-b \
#        --command="rm -rf ~/diotima" ;

# Fetch checkpoints.
# echo "[*] Copying codebase to tpu-vm-"$i"..."; gcloud compute tpus tpu-vm scp tpu-test-0:~/config.pickle ./ --zone=us-central2-b