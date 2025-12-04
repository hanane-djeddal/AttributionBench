#!/bin/bash
#SBATCH --job-name=h100_bench_train_autoaisall_contrastive2 #augmented
#SBATCH --output=jz_%j_%x.out
#SBATCH --error=errjz_%j_%x.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=1:50:00
#SBATCH --constraint=h100
#SBATCH --cpus-per-task=16
#SBATCH --account=fiz@h100 ##
#SBATCH --qos=qos_gpu_h100-dev

export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge
module load arch/h100
module load pytorch-gpu/py3/2.8.0
export HF_HOME=$WORK/.cache/huggingface
export WANDB_MODE=offline

cd /lustre/fswork/projects/rech/fiz/udo61qq/Code/AttributionBench/zsh_scripts/

#sh run_llama2_7b.sh 
sh run_autoais_cont.sh 