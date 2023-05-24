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
# SIR experiment
Please first generate training data and testing data:
```
python ./experiments/generate_sir_samples.py --random-seed=1 --save-folder=./SIR/sir_sde_data_1/
```
## GMM
```
python ./experiments/sir.py --id=0 --n-stage=10 --post-net-type=GMM --actor-lr=5e-4 --critic-lr=1e-3 --post-lr=5e-4 --data-folder=./SIR/sir_sde_data_1 --save-folder=./results/sir/GMM_5e-4_1e-3_5e-4/terminal_id0_h10/
```
