#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G
#SBATCH --partition=isi

cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval/contriever

module load conda
source conda activate contriever

python3 finetuning.py