#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gpus-per-task=a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

module load python/3.11
pip install -r requirements.txt
python3 data_vllm_v2.py
