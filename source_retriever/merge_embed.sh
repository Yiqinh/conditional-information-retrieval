#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --partition=sched_mit_psfc_gpu_r8

cd /pool001/spangher/alex/conditional-information-retrieval
source conda activate conditional_retrieval

python3 source_retriever/merge_embed.py