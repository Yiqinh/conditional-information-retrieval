#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=400G
#SBATCH --partition=sched_mit_psfc_gpu_r8

start_idx=$1
end_idx=$2

cd /pool001/spangher/alex/conditional-information-retrieval

conda init
source conda activate py39-retrieve-vllm

export OMP_NUM_THREADS=100
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python3 interleaving/interleave_v2_hybrid.py --start_idx=${start_idx} --end_idx=${end_idx} --iterations=20