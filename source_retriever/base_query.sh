#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=400G
#SBATCH --partition=isi

source /home1/spangher/.bashrc
cd
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

#module purge
#eval "$(conda shell.bash hook)"

conda activate vllm-py310
#conda env update -f env.yaml
#pip install -r requirements.txt

python source_retriever/base_query.py --start_idx 0 --end_idx 272900 --model meta-llama/Meta-Llama-3-70B-Instruct
