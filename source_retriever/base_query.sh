#!/bin/sh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=100GB
#SBATCH --cpus-per-gpu=10
#SBATCH --partition=gpu

cd
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

source conda activate CIR
conda env update -f env.yaml
pip install -r requirements.txt

python3 source_retriever/base_query.py --start_idx 0 --end_idx 100 --model meta-llama/Meta-Llama-3-8B-Instruct