#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

cd
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

module load conda
source conda activate CIR

python3 cosine_sim.py