#!/bin/bash
run_list="run1 run2 run3"
#k_list="1"
k_list="5 10 15 20 25 30"
for run in $run_list; do
    for k in $k_list; do
        sbatch run_main.sh "$run" "$k"
        echo "Submitted job: $run/k-$k"
    done
done