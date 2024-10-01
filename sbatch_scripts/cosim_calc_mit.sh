#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=100G
#SBATCH --partition=sched_mit_psfc_gpu_r8

cd /pool001/spangher/alex/conditional-information-retrieval
pip install -U sentence-transformers
pip install numpy

python3 interleaving/cosim_calc.py
