#!/bin/bash

dir=$PWD
parentdir="$(dirname "$dir")"
FLUSHDIR="$(dirname "$parentdir")"

source $FLUSHDIR/vsOED_env/bin/activate
cd $FLUSHDIR

id1=$1
id2=$2
post=$3

for id in $(seq $id1 $id2); do
    for h in 1 2 3 4 5 7 10; do
        if [ $post = 'GMM' ]; then
            python ./experiments/sir.py --id=$id --n-stage=$h --post-net-type=$post --actor-lr=1e-4 --data-folder=./SIR/sir_sde_data_"$((id+1))" --save-folder=./results/sir/"$post"_1e-4/terminal_id"$id"_h"$h"/
        else
            echo "NFs"
        fi
    done
done