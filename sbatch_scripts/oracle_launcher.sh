#!/bin/bash


cd /pool001/spangher/alex/conditional-information-retrieval/sbatch_scripts


start_idx=$1
end_idx=$2
step=$3

for ((i=$start_idx; i<$end_idx; i+=$step)); do
    sbatch interleave_oracle.sh $i $((i + step))
done