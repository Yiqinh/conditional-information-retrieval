#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

source conda activate CIR
module load python/3.11
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
pip install -r requirements.txt
python3 source_retriever/source_retrieval.py