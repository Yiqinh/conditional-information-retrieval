#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=200G
#SBATCH --partition=isi

source /home1/spangher/.bashrc
conda activate vllm-retriv-py39

cd /project/jonmay_1426/spangher/conditional-information-retrieval

python3 source_retriever/v3_embed_dense.py