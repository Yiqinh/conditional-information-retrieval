#!/bin/bash


cd /pool001/spangher/alex/conditional-information-retrieval/sbatch_scripts


step=$1
end_idx=$2

for ((i=0; i<end_idx; i+=${step})); do
    sbatch interleave_mit.sh ${i} $((i + step))
done
