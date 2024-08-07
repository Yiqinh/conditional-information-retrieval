#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=400G
#SBATCH --partition=isi

source /home1/spangher/.bashrc
conda activate vllm-py310

cd
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

#module purge
#eval "$(conda shell.bash hook)"

#conda env update -f env.yaml
#pip install -r requirements.txt

python3 baseline_queries/query2.py