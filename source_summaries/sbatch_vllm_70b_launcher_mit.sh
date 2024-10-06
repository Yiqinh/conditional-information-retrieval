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
 sbatch data_vllm_70b__run_all_source_summaries_mit.sh 0 5000 4
 sbatch data_vllm_70b__run_all_source_summaries_mit.sh 20000 5000 4
 sbatch data_vllm_70b__run_all_source_summaries_mit.sh 40000 5000 5

sbatch data_vllm_70b__run_all_source_summaries_mit.sh 0 20000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 20000 40000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 40000 60000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 60000 80000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 80000 100000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 100000 120000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 120000 140000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 140000 160000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 160000 180000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 180000 200000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 200000 220000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 220000 240000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 240000 260000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 260000 280000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 280000 300000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 300000 320000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 320000 340000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 340000 360000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 360000 380000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 380000 400000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 400000 420000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 420000 440000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 440000 460000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 460000 480000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 480000 500000 1


sbatch data_vllm_70b__run_all_source_summaries_mit.sh 0 50000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 50000 100000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 100000 150000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 150000 200000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 200000 250000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 250000 300000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 300000 350000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 350000 400000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 400000 450000 1
sbatch data_vllm_70b__run_all_source_summaries_mit.sh 450000 500000 1

