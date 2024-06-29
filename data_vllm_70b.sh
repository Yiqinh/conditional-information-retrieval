#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=100G
#SBATCH --partition=isi

module load python/3.11
pip install -r requirements.txt
python3 data_vllm_70b.py --start_idx 0 --end_idx 10_000
python3 data_vllm_70b.py --start_idx 10_000 --end_idx 20_000
python3 data_vllm_70b.py --start_idx 20_000 --end_idx 30_000
python3 data_vllm_70b.py --start_idx 30_000 --end_idx 40_000
python3 data_vllm_70b.py --start_idx 40_000 --end_idx 50_000
python3 data_vllm_70b.py --start_idx 50_000 --end_idx 60_000
python3 data_vllm_70b.py --start_idx 60_000 --end_idx 70_000
python3 data_vllm_70b.py --start_idx 70_000 --end_idx 80_000
python3 data_vllm_70b.py --start_idx 80_000 --end_idx 90_000
python3 data_vllm_70b.py --start_idx 90_000 --end_idx 100_000
python3 data_vllm_70b.py --start_idx 100_000 --end_idx 110_000
python3 data_vllm_70b.py --start_idx 110_000 --end_idx 120_000
python3 data_vllm_70b.py --start_idx 120_000 --end_idx 130_000
python3 data_vllm_70b.py --start_idx 130_000 --end_idx 140_000
python3 data_vllm_70b.py --start_idx 140_000 --end_idx 150_000
python3 data_vllm_70b.py --start_idx 150_000 --end_idx 160_000
python3 data_vllm_70b.py --start_idx 160_000 --end_idx 170_000
python3 data_vllm_70b.py --start_idx 170_000 --end_idx 180_000
python3 data_vllm_70b.py --start_idx 180_000 --end_idx 190_000
python3 data_vllm_70b.py --start_idx 190_000 --end_idx 200_000
python3 data_vllm_70b.py --start_idx 200_000 --end_idx 210_000
python3 data_vllm_70b.py --start_idx 210_000 --end_idx 220_000
python3 data_vllm_70b.py --start_idx 220_000 --end_idx 230_000
python3 data_vllm_70b.py --start_idx 230_000 --end_idx 240_000
python3 data_vllm_70b.py --start_idx 240_000 --end_idx 250_000
python3 data_vllm_70b.py --start_idx 250_000 --end_idx 260_000
python3 data_vllm_70b.py --start_idx 260_000 --end_idx 270_000
python3 data_vllm_70b.py --start_idx 270_000 --end_idx 280_000
python3 data_vllm_70b.py --start_idx 280_000 --end_idx 290_000
python3 data_vllm_70b.py --start_idx 290_000 --end_idx 300_000
python3 data_vllm_70b.py --start_idx 300_000 --end_idx 310_000
python3 data_vllm_70b.py --start_idx 310_000 --end_idx 320_000
python3 data_vllm_70b.py --start_idx 320_000 --end_idx 330_000
python3 data_vllm_70b.py --start_idx 330_000 --end_idx 340_000
python3 data_vllm_70b.py --start_idx 340_000 --end_idx 350_000
python3 data_vllm_70b.py --start_idx 350_000 --end_idx 360_000
python3 data_vllm_70b.py --start_idx 360_000 --end_idx 370_000
python3 data_vllm_70b.py --start_idx 370_000 --end_idx 380_000
python3 data_vllm_70b.py --start_idx 380_000 --end_idx 390_000
