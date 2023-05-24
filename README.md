# vsOED
Policy gradient (PG) based variational sequential optimal experimental design (vsOED)


# Installation
The code was tested in Python 3.9.7. GPU acceleration is highly recommended.
Use `venv` to install:
```
python -m venv vsOED_env
source vsOED_env/bin/activate
pip install -r requirements.txt
```

# Source location finding experiment (uni-model)
## PoI inference OED
### GMM
#### Terminal information gain
```
python ./experiments/source.py --id=0 --n-stage=30 --actor-lr=5e-4 --post-net-type=GMM --save-folder=./results/source/unimodel/poi/GMM/terminal_id0_h30/
```
#### Incremental information gain
```
python ./experiments/source.py --id=0 --n-stage=30 --discount=0.9 --transition=0 --post-net-type=GMM --n-incre=30 --save-folder=./results/source/unimodel/poi/GMM/incre_id0_h30/
```
### NFs
#### Terminal information gain
```
python ./experiments/source.py --id=0 --n-stage=30 --post-net-type=NFs --n-trans=4 --Split1 0 3 --Split2 1 2 --save-folder=./results/source/unimodel/poi/NFs/terminal_id0_h30/
```
#### Incremental information gain
```
python ./experiments/source.py --id=0 --n-stage=30 --discount=0.9 --transition=0 --post-net-type=NFs --n-incre=30 --n-trans=4 --Split1 0 3 --Split2 1 2 --save-folder=./results/source/unimodel/poi/NFs/incre_id0_h30/
```
## QoI goal-oriented OED
### GMM
#### Terminal information gain
```
python ./experiments/source.py --id=0 --n-stage=30 --include-goal=True --poi-weight=0 --goal-weight=1 --actor-lr=5e-4 --post-net-type=GMM --save-folder=./results/source/unimodel/goal/GMM/terminal_id0_h30/
```
#### Incremental information gain
```
python ./experiments/source.py --id=0 --n-stage=30 --include-goal=True --poi-weight=0 --goal-weight=1 --discount=0.9 --transition=0 --post-net-type=GMM --n-incre=30 --save-folder=./results/source/unimodel/goal/GMM/incre_id0_h30/
```

# Source location finding experiment (multi-model)
## Model discrimination OED (other OED scenarios are omitted here)
### GMM
#### Terminal information gain
```
python ./experiments/source.py --model='multi' --model-weight=1 --poi-weight=0 --goal-weight=0 --id=0 --n-stage=30 --actor-lr=2e-4 --post-net-type=GMM --save-folder=./results/source/multimodel/model/GMM/terminal_id0_h30/
```
#### Incremental information gain
```
python ./experiments/source.py --model='multi' --model-weight=1 --poi-weight=0 --goal-weight=0 --id=0 --n-stage=30 --n-incre=30 --discount=0.9 --transition=0 --post-net-type=GMM --save-folder=./results/source/multimodel/model/GMM/incre_id0_h30/
```


# CES experiment
## GMM
```
python ./experiments/ces.py --id=0 --n-stage=10 --post-net-type=GMM --save-folder=./results/ces/GMM/terminal_id0_h10/
```
## NFs
```
python ./experiments/ces.py --id=0 --n-stage=10 --post-net-type=NFs --n-trans=4 --save-folder=./results/ces/NFs/terminal_id0_h10/
```

# SIR experiment
Please first generate training data and testing data:
```
python ./experiments/generate_sir_samples.py --random-seed=1 --save-folder=./SIR/sir_sde_data_1/
```
## GMM
```
python ./experiments/sir.py --id=0 --n-stage=10 --post-net-type=GMM --actor-lr=5e-4 --critic-lr=1e-3 --post-lr=5e-4 --data-folder=./SIR/sir_sde_data_1 --save-folder=./results/sir/GMM/terminal_id0_h10/
```
## NFs
```
python ./experiments/sir.py --id=0 --n-stage=10 --post-net-type=NFs --n-trans=10 --actor-lr=5e-4 --critic-lr=1e-3  --post-lr=1e-3 --data-folder=./SIR/sir_sde_data_1 --save-folder=./results/sir/NFs/terminal_id0_h10/
```

# Convection-diffusion experiment
## Model discrimination OED (other OED scenarios are omitted here)
### GMM
```
python ./experiments/conv_diff.py --id=0 --n-stage=10 --model='multi' --include-goal=False --include-cost=True --cost-ratio=0.1 --model-weight=1 --poi-weight=0 --goal-weight=0 --n-incre=1 --discount=1 --actor-lr=5e-4 --post-net-type=GMM --save-folder=./results/conv_diff/multimodel/model_cost_0.1/GMM/terminal_id0_h10/ --n-update=10
```
