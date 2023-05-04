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
    for h in 30 27 24 21 18 15 12 9 6 3 1; do
        if [ $post = 'GMM' ]; then
            python ./experiments/source.py --id=$id --n-stage=$h --discount=0.9 --transition=0 --post-net-type=$post --n-incre=$h --save-folder=./results/source/unimodel/poi/"$post"/incre_id"$id"_h"$h"/
        else
            echo "NFs"
        fi
    done
done
