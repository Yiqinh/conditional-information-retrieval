#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=200G
#SBATCH --partition=isi

cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval/contriever

module load conda
source conda activate contriever
pip install torch
python3 finetuning.py