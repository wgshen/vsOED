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
## GMM
### Terminal information gain

### Incremental information gain


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
