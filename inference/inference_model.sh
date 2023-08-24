#!/usr/bin/env bash
#SBATCH -A SNIC2022-1-37 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=T4:1  # We’re launching 2 nodes with 4 Nvidia V100 GPUs each
#SBATCH -t 0-3:30:00

singularity run -B /mimer/NOBACKUP/groups/globalpoverty1/ /cephyr/NOBACKUP/groups/globalpoverty1/singularity_imgs/container_latest.sif inference_model.py --model_name=bidirectional_resnet_lstm_10 --fold_config=ooc