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

start_idx=$1
step=$2
iterations=$3
iterations=$((iterations + 1))
end_idx=$((start_idx + step))

for ((i=0; i<iterations; i++)); do
    python data_vllm_70b_v3.py \
      --start_idx ${start_idx} \
      --end_idx ${end_idx} \
      --id_col article_url \
      --text_col article_text \
      --data_dir ../data \
      --source_data_file   full-source-scored-data.jsonl.gz \
      --output_file  ../data/v3_source_summaries/test_sources.txt \
      --do_article_gen \
      --do_source_summ \
      --do_narr_key_prompt \
      --do_cent_prompt 

    start_idx=${end_idx}
    end_idx=$((start_idx + step))
done
