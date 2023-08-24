#!/usr/bin/env bash
#SBATCH -A SNIC2022-1-37 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=A40:1  # We’re launching 1 nodes with 1 Nvidia A40 GPU
#SBATCH -t 0-3:00:00

singularity run -B /mimer/NOBACKUP/groups/globalpoverty1/ /cephyr/NOBACKUP/groups/globalpoverty1/singularity_imgs/container_latest.sif preprocessing/3_calculate_band_stats.py