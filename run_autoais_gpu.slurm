#!/bin/bash
#SBATCH --job-name=a100_LONGbench_train_autoaisall
#SBATCH --output=jz_%j_%x.out
#SBATCH --error=errjz_%j_%x.out
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --hint=nomultithread
#SBATCH --time=10:50:00
#SBATCH --constraint=a100
#SBATCH --cpus-per-task=24
#SBATCH --account=fiz@a100 ###SBATCH --qos=qos_gpu_a100-dev
#SBATCH -C a100

export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

module purge
module load arch/a100
module load pytorch-gpu/py3/2.8.0
export HF_HOME=$WORK/.cache/huggingface
export WANDB_MODE=offline

cd /lustre/fswork/projects/rech/fiz/udo61qq/Code/AttributionBench/zsh_scripts/

#sh run_autoais.sh
#sh run_llama2_7b.sh 
sh run_autoais_cont.sh 
