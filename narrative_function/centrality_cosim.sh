#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=5
#SBATCH --mem=40G
#SBATCH --partition=isi

cd
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

module load conda
source conda activate CIR

python -m pip install scipy
pip install -U sentence-transformers

python3 narrative_function/centrality_cosim.py