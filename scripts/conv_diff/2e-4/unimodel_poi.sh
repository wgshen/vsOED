#!/bin/bash

dir=$PWD
dir="$(dirname "$dir")"
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
            python ./experiments/conv_diff.py --id=$id --n-stage=$h --model='uni' --include-goal=False --include-cost=False --model-weight=0 --poi-weight=1 --goal-weight=0 --n-incre=1 --discount=1.0 --actor-lr=2e-4 --post-net-type=$post --save-folder=./results/conv_diff/unimodel/poi/"$post"_2e-4/terminal_id"$id"_h"$h"/
        else
            echo "NFs"
        fi
    done
done

for id in $(seq $id1 $id2); do
    for h in 20 15 10 5 1; do
        if [ $post = 'GMM' ]; then
            python ./experiments/conv_diff.py --id=$id --n-stage=$h --model='uni' --include-goal=False --include-cost=False --model-weight=0 --poi-weight=1 --goal-weight=0 --n-incre=$h --discount=0.9 --actor-lr=2e-4 --post-net-type=$post --save-folder=./results/conv_diff/unimodel/poi/"$post"_2e-4/incre_id"$id"_h"$h"/
        else
            echo "NFs"
        fi
    done
done
