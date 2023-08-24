#!/usr/bin/env bash
#SBATCH -A SNIC2022-1-37 -p alvis
#SBATCH -N 1
#SBATCH --gpus-per-node=T4:1  # We’re launching 2 nodes with 4 Nvidia V100 GPUs each
#SBATCH -t 0-3:30:00

base_model_path=${1}
save_dir=${2}
model_fold=${3}
fold_config=${4}
eval_folds=${5}

echo "Base model path: $base_model_path";
echo "Model fold: $model_fold";
echo "Fold config: $fold_config";
echo "Eval fold: $eval_folds";
echo "Save dir: $save_dir";

singularity run -B /mimer/NOBACKUP/groups/globalpoverty1/ /cephyr/NOBACKUP/groups/globalpoverty1/singularity_imgs/container_latest.sif to_features.py --base_model_path=$base_model_path --save_dir=$save_dir --model_fold=$model_fold --fold_config=$fold_config --eval_folds=$eval_folds --save_csv --ten_frame_model