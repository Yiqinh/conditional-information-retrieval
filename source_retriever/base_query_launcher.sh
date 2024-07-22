#!/bin/bash

# Launch multiple jobs (modify as needed)
total=300000
for ((i=0; i<total; i+=30000)); do
    end_idx=$((i + 30000))
    sbatch base_query.sh --start_idx "${i}" --end_idx "${end_idx}"
done

