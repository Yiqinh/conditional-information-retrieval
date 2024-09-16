#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=80:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=20
#SBATCH --mem=150G
#SBATCH --partition=isi

# Activate your environment or load modules if needed
# module load python/3.x
# source /path/to/your/venv/bin/activate

# Run the script with desired parameters
python train_sentence_similarity_model.py \
    --model_name 'microsoft/mpnet-base' \
    --data_file 'similarity_training_data/source-triplets.jsonl' \
    --output_dir 'models/mpnet-base-all-nli-triplet' \
    --num_train_epochs 1 \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --learning_rate 2e-5 \
    --warmup_ratio 0.1 \
    --fp16 \
    --eval_strategy 'steps' \
    --eval_steps 100 \
    --save_strategy 'steps' \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 100 \
    --run_name 'mpnet-base-all-nli-triplet' \
    --test_size 0.1 \
    --train_subset_size 100000 \
    --do_initial_evaluation
