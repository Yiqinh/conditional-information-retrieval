#!/bin/bash

# Example usage: ./launch_batches.sh 200000 5000 11
# arguments: start_idx, step, iterations
# start_idx: the starting index for the first batch
# step: the number of rows to process in each batch
# iterations: the number of batches to process
# The script will launch iterations batches, each processing step rows starting from start_idx
# and ending at start_idx + step
# it will process: start_idx, start_idx + step, start_idx + 2*step, ..., start_idx + iterations*step
# for a total of step*iterations rows.

# Launch multiple jobs (modify as needed)
sbatch data_vllm_70b__run_all_source_summaries.sh 0 10000 10
sbatch data_vllm_70b__run_all_source_summaries.sh 110000 10000 10
sbatch data_vllm_70b__run_all_source_summaries.sh 220000 10000 10
