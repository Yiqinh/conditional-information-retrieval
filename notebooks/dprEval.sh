#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=400G
#SBATCH --partition=isi


source /home1/spangher/.bashrc
source activate hay
cd /project/jonmay_1426/spangher/conditional-information-retrieval/notebooks
python dprEval.py