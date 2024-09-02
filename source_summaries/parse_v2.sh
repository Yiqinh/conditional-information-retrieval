#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --mem=16G
#SBATCH --partition=isi

source /home1/spangher/.bashrc

cd
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval
pip install -U scikit-learn

python3 source_summaries/parse_v2.py