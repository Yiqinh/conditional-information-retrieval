#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

conda env update -f env.yaml
source conda activate CIR
pip install -r requirements.txt

python3 source_retriever/source_retrieval.py