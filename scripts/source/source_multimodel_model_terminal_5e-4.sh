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
    for h in 30 20 15 10 5 1; do
        if [ $post = 'GMM' ]; then
            python ./experiments/source.py --model='multi' --model-weight=1 --poi-weight=0 --goal-weight=0 --id=$id --n-stage=$h --actor-lr=5e-4 --post-net-type=$post --save-folder=./results/source/multimodel/model/"$post"_5e-4/terminal_id"$id"_h"$h"/
        else
            echo "NFs"
        fi
    done
done
