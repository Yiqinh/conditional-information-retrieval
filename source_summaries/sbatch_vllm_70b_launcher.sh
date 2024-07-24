#!/bin/bash

# Example usage: ./launch_batches.sh 200000 5000 11

# Launch multiple jobs (modify as needed)
sbatch data_vllm_70b.sh 0 10000 11
sbatch data_vllm_70b.sh 110000 10000 11
sbatch data_vllm_70b.sh 220000 10000 11
