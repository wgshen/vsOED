#!/bin/bash

dir=$PWD
parentdir="$(dirname "$dir")"
FLUSHDIR="$(dirname "$parentdir")"

source $FLUSHDIR/vsOED_env/bin/activate
cd $FLUSHDIR

python ./experiments/conv_diff.py --n-stage=15 --model='uni' --include-goal=False --include-cost=False --model-weight=0 --poi-weight=1 --goal-weight=0 --post-lr=1e-3 --post-gamma=0.9999 --actor-lr=1e-3 --actor-gamma=0.9999 --critic-lr=1e-3 --critic-gamma=0.9999 --n-incre=1 --discount=1 --target-lr=0.1 --design-noise-scale=0.05 --design-noise-decay=0.9999 --transition=10000 --save-folder=./results/conv_diff/conv_diff_3/


python ./experiments/conv_diff.py --n-stage=15 --model='uni' --include-goal=False --include-cost=True --model-weight=0 --poi-weight=1 --goal-weight=0 --post-lr=1e-3 --post-gamma=0.9999 --actor-lr=1e-3 --actor-gamma=0.9999 --critic-lr=1e-3 --critic-gamma=0.9999 --n-incre=1 --discount=1 --target-lr=0.1 --design-noise-scale=0.05 --design-noise-decay=0.9999 --transition=10000 --save-folder=./results/conv_diff/conv_diff_4/


python ./experiments/conv_diff.py --n-stage=15 --model='multi' --include-goal=False --include-cost=False --model-weight=1 --poi-weight=0 --goal-weight=0 --post-lr=1e-3 --post-gamma=0.9999 --actor-lr=1e-3 --actor-gamma=0.9999 --critic-lr=1e-3 --critic-gamma=0.9999 --n-incre=1 --discount=1 --target-lr=0.1 --design-noise-scale=0.05 --design-noise-decay=0.9999 --transition=10000 --save-folder=./results/conv_diff/conv_diff_5/

python ./experiments/conv_diff.py --n-stage=15 --model='multi' --include-goal=False --include-cost=False --model-weight=0 --poi-weight=1 --goal-weight=0 --post-lr=1e-3 --post-gamma=0.9999 --actor-lr=1e-3 --actor-gamma=0.9999 --critic-lr=1e-3 --critic-gamma=0.9999 --n-incre=1 --discount=1 --target-lr=0.1 --design-noise-scale=0.05 --design-noise-decay=0.9999 --transition=10000 --save-folder=./results/conv_diff/conv_diff_6/


python ./experiments/conv_diff.py --n-stage=15 --model='multi' --include-goal=True --include-cost=False --model-weight=1 --poi-weight=1 --goal-weight=1 --post-lr=1e-3 --post-gamma=0.9999 --actor-lr=1e-3 --actor-gamma=0.9999 --critic-lr=1e-3 --critic-gamma=0.9999 --n-incre=1 --discount=1 --target-lr=0.1 --design-noise-scale=0.05 --design-noise-decay=0.9999 --transition=10000 --save-folder=./results/conv_diff/conv_diff_7/