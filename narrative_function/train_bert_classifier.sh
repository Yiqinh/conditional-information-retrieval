#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --partition=isi

python train_bert_classifier.py
