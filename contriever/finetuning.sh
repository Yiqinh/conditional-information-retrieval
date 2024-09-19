#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=150G
#SBATCH --partition=isi

cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

module load conda
source conda activate contriever

python3 contriever/finetuning.py