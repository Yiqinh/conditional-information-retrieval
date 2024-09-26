#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=200G
#SBATCH --partition=isi

srun -N 1 --cpus-per-task=32 --gres=gpu:4 --partition=sched_mit_psfc_gpu_r8 --pty bash -i




cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval/contriever

module load conda
source conda activate contriever
pip install torch
python3 finetuning.py