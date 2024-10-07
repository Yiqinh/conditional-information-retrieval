#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50
#SBATCH --mem=200G
#SBATCH --partition=sched_mit_psfc_gpu_r8

cd /pool001/spangher/alex/conditional-information-retrieval

source conda activate conditional_retrieval

pip install numpy

start_idx=$1
end_idx=$2


export VLLM_WORKER_MULTIPROC_METHOD=spawn

python3 interleaving/interleave_dpr.py --start_idx 0 --end_idx 100 --iterations=20