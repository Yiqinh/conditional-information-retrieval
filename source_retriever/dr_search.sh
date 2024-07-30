#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=400G
#SBATCH --partition=isi

source /home1/spangher/.bashrc

cd
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

conda activate retriv-py39

python3 source_retriever/dr_search.py --index_name vanilla_baseline