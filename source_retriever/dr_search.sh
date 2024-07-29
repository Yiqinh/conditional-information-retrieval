#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=isi

cd
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

conda activate retriv-py39

python3 source_retriever/dr_search.py --index_name vanilla_baseline