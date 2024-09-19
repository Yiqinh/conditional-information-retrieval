#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu


source /home1/spangher/.bashrc
source activate hay
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval/notebooks
python dprEval.py