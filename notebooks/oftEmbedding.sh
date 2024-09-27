#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G
#SBATCH --partition=isi


source /home1/spangher/.bashrc
source activate hay
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval/notebooks
python oftEmbedding.py