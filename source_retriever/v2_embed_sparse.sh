#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100-80gb
#SBATCH --mem=200GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu


source /home1/spangher/.bashrc
conda activate retriv-py39

cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

python3 source_retriever/v2_embed_sparse.py
