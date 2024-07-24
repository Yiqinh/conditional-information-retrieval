#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=2:00:00

cd
cd /project/jonmay_231/spangher/Projects/conditional-information-retrieval

module purge
module load gcc/8.3.0
module load python/3.11

python3 baseline_queries/organize_queries.py
