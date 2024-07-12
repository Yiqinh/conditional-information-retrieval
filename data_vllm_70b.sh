#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=400G
#SBATCH --partition=isi


cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval
source /home1/spangher/.bashrc
conda activate py311

start_idx=$1
step=$2
iterations=$3
end_idx=$((start_idx + step))

for ((i=0; i<iterations; i++)); do
    python3 data_vllm_70b.py --start_idx ${start_idx} --end_idx ${end_idx}
    start_idx=${end_idx}
    end_idx=$((start_idx + step))
done