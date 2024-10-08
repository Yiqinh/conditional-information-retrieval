#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=100
#SBATCH --mem=400G
#SBATCH --partition=sched_mit_psfc_gpu_r8


source /home/spangher/.bashrc
conda activate alex

python label_discourse_vllm.py \
  --input_data_file ../data/quick-discourse-prompts-to-run.csv.gz \
  --id_col url \
  --prompt_col prompt \
  --output_file  ../data/v3_source_summaries/discourse-labeled/discourse_labeled.txt \

