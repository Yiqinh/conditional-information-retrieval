#!/bin/bash

# Example usage: ./launch_batches.sh 200000 5000 11

# Launch multiple jobs (modify as needed)
sbatch data_vllm_70b__run_all_source_summaries.sh 0 10000 11
sbatch data_vllm_70b__run_all_source_summaries.sh 110000 10000 11
sbatch data_vllm_70b__run_all_source_summaries.sh 220000 10000 11
