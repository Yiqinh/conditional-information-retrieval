#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=400G
#SBATCH --partition=isi

source /home1/spangher/.bashrc

conda init
source conda activate vllm-retriv-py39  

start_idx=$1
end_idx=$2

cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

export OMP_NUM_THREADS=50
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HOME=/project/jonmay_231/spangher/huggingface_cache

python3 interleaving/interleave_v2.py --start_idx=${start_idx} --end_idx=${end_idx} --iterations=20