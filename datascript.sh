#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem-per-gpu=100GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

module load conda
source conda activate CIR
pip install -r requirements.txt
module load python/3.11
python3 data_script.py
