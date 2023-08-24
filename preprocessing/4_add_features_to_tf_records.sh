#!/usr/bin/env bash
#SBATCH -A SNIC2022-1-37 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=T4:1  # We’re launching 1 node with 1 Nvidia T4 GPUs each
#SBATCH -t 0-2:00:00

singularity run -B /mimer/NOBACKUP/groups/globalpoverty1/data /cephyr/NOBACKUP/groups/globalpoverty1/singularity_imgs/container_latest.sif preprocessing/4_add_features_to_tf_records.py --data-path=/mimer/NOBACKUP/groups/globalpoverty1/data --split-id=9