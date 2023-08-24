#!/usr/bin/env bash
#SBATCH -A SNIC2022-3-38 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=T4:3  # We’re launching 2 nodes with 4 Nvidia V100 GPUs each
#SBATCH -t 0-1:00:00

model_fold=${1}
fold_config=${2}

echo "Model fold: $model_fold";
echo "Fold config: $fold_config";

singularity run /cephyr/NOBACKUP/groups/globalpoverty1/singularity_imgs/container_latest.sif baseline/knn_train.py --model_fold=$model_fold --fold_config=$fold_config --save_plots

singularity exec /cephyr/NOBACKUP/groups/globalpoverty1/singularity_imgs/container_latest.sif python