#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --constraint=a100-80gb
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

cd
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

module load conda
source conda activate CIR

pip install --upgrade pip
pip install farm-haystack[colab,inference,metrics]

python3 notebooks/dataLoader.py
python3 notebooks/dataTraining.py
