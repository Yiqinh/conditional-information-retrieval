#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --partition=isi

module load python/3.11
pip install -r requirements.txt
python3 data_vllm_70b.py
