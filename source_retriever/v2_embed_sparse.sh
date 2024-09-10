#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --partition=isi

source /home1/spangher/.bashrc
conda activate retriv-py39

cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

python3 source_retriever/v2_embed_sparse.py
