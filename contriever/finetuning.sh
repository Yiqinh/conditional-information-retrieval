#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=50
#SBATCH --mem=400G
#SBATCH --partition=sched_mit_psfc_gpu_r8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=liheng@berkeley.edu


cd /pool001/spangher/alex/conditional-information-retrieval/contriever

conda init
source conda activate contriever
pip install torch

export OMP_NUM_THREADS=50

python3 finetuning.py
