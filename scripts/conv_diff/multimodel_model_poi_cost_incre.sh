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
    for h in 20 15 10 5 1; do
        if [ $post = 'GMM' ]; then
            python ./experiments/conv_diff.py --id=$id --n-stage=$h --model='multi' --include-goal=False --include-cost=True --model-weight=1 --poi-weight=1 --goal-weight=0 --n-incre=$h --discount=0.9 --post-net-type=$post --save-folder=./results/conv_diff/multimodel/model_poi_cost/"$post"/incre_id"$id"_h"$h"/
        else
            echo "NFs"
        fi
    done
done