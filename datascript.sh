#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=80GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

module load python/3.11
pip install -r requirements.txt
python3 data_vllm.py
