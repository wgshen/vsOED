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
            python ./experiments/source.py --id=$id --n-stage=$h --include-nuisance=True --include-goal=True --poi-weight=1 --goal-weight=1 --post-net-type=$post --actor-lr=5e-4 --save-folder=./results/source/unimodel/nuis_poi_goal/"$post"/terminal_id"$id"_h"$h"/
        else
            echo "NFs"
        fi
    done
done