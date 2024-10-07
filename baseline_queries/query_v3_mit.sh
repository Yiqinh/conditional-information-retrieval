#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50
#SBATCH --mem=200G
#SBATCH --partition=sched_mit_psfc_gpu_r8

start_idx=$1
end_idx=$2

cd /pool001/spangher/alex/conditional-information-retrieval
source conda activate py39-retrieve-vllm

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=50

python3 baseline_queries/query_v3.py --start_idx=${start_idx} --end_idx=${end_idx}