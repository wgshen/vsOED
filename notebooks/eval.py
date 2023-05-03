import os,sys,inspect
# current_dir = '/home/jiayuand/seqOED_variational/examples/location'
# parent_dir = '/home/jiayuand/seqOED_variational'

# current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = os.path.dirname(os.path.dirname(current_dir))
parent_dir = '/scratch/xhuan_root/xhuan1/wgshen/vsOED/'
sys.path.insert(0, parent_dir) 

import numpy as np
# from scipy.stats import norm, beta, dirichlet
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from   torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
if torch.cuda.is_available():  
    dev = "cuda:0" 
else:  
    dev = "cpu"  
device = torch.device(dev) 
torch.set_default_device(device)
dtype = torch.float32
torch.set_default_dtype(dtype)

from vsOED import VSOED, PGvsOED, GMM_NET, NFs, POST_APPROX
from vsOED.utils import *
from vsOED.models import *

import dowel
import joblib

n_pois = [4]
n_stage = 5
n_incre = 1

# model = SOURCE()

# post_approx = POST_APPROX(n_stage=1, n_obs=1, n_design=1, n_pois=[1], mu_bounds={'poi': [[0, 1]]}, max_sigmas={'poi': [[1]]})

# vsoed = PGvsOED(n_stage=1, n_param=1, n_design=1, n_obs=1, 
#                  model=model, prior=model, design_bounds=[[0, 1]], post_approx=post_approx,
#                 kld_reward_fun=post_approx.reward_fun)

def reward_fun(*args, **kws):
    return 0

folder = parent_dir + f'experiments/source_{len(n_pois)}model_{n_stage}stage_{n_incre}incre_GMM/'

dowel.logger.add_output(dowel.StdOutput())
dowel.logger.add_output(dowel.CsvOutput(folder + 'progress.csv'))
dowel.logger.add_output(dowel.TextOutput(folder + 'progress.txt'))

vsoed = joblib.load(folder + 'itr_1000.pkl')

set_random_seed(EVAL_SEEDS[0])
ret = vsoed.asses(2000, n_contrastive_sample=int(1e6), return_all=True, dowel=dowel, save_path=folder + 'evaluation.pt')
# averaged_reward = ret['averaged_reward']
# params = ret['params']
# ds_hist = ret['ds_hist']
# ys_hist = ret['ys_hist']
# rewards_hist = ret['rewards_hist']
# averaged_reward