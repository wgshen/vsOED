import torch
import math
from .utils import *

class CONV_DIFF(object):
    def __init__(self, multimodel=False, include_str_wid=False, include_goal=False, include_cost=False, cost_ratio=0.2):
        self.multimodel = multimodel
        self.include_str_wid = include_str_wid
        self.include_goal = include_goal
        self.include_cost = include_cost
        self.cost_ratio = cost_ratio
        data_folder = './experiments/conv_diff_nets/'
        data_folder += 'random_strength_and_width/' if include_str_wid else 'fixed_strength_and_width/'
        if multimodel:
            self.model_nets = []
            self.flux_nets = []
            for model in range(3):
                fname = data_folder + f'model_{model + 1}/model_nets.pt'
                net = torch.load(fname)
                self.model_nets.append(net)
                if include_goal:
                    fname = data_folder + f'model_{model + 1}/flux_net.pt'
                    net = torch.load(fname)
                    self.flux_nets.append(net)
        else:
            fname = data_folder + f'model_2/model_nets.pt'
            self.model_nets = torch.load(fname)
            fname = data_folder + f'model_2/flux_net.pt'
            self.flux_net = torch.load(fname)
    
    def get_n_param(self, model):
        if model == -1:
            return self.get_n_param(1)
        else:
            return (model + 1) * 2 + 2 + 2 * self.include_str_wid

    def deterministic(self, stage, params, ds, xps=None):

        n_sample = max(len(params), len(ds))
        
        if not self.multimodel:
            n_param = self.get_n_param(1)
            inputs = torch.zeros(n_sample, n_param + 2)
            inputs[:, :n_param] = params[:, :n_param]
            inputs[:, n_param:] = ds
            G = self.model_nets[stage](inputs)
        else:
            G = torch.zeros(n_sample, 1)
            model_idxs = params[:, 0].to(int)
            for model in range(3):
                idxs = model_idxs == model
                n_param = self.get_n_param(model)
                inputs = torch.zeros(idxs.sum(), n_param + 2)
                inputs[:, :n_param] = params[idxs, 1:1+n_param]
                inputs[:, n_param:] = ds if len(ds) == 1 else ds[idxs]
                G[idxs] = self.model_nets[model][stage](inputs)
                
        return G, torch.tensor(0.05)
        
    def model(self, stage, params, ds, xps=None):
        with torch.no_grad():
            mu, sigma = self.deterministic(stage, params, ds, xps)
            y = torch.randn(mu.shape) * sigma + mu
        return y

    def flux(self, params):
        with torch.no_grad():
            if not self.multimodel:
                n_param = self.get_n_param(1)
                flux = self.flux_net(params[:, :n_param])
            else:
                flux = torch.ones(len(params), 1)
                model_idxs = params[:, 0].to(int)
                for model in range(3):
                    idxs = model_idxs == model
                    n_param = self.get_n_param(model)
                    flux[idxs] = self.flux_nets[model](params[idxs, 1:1+n_param])
            return torch.log(torch.abs(flux) + 1e-27)
            # return flux
    
    def loglikeli(self, stage, ys, params, ds, xps=None, true_param=None):
        """
        y : (n or 1, n_obs)
        theta : (n or 1, n_param)
        d : (n or 1, n_design)
        """
        with torch.no_grad():
            if true_param is None or not self.multimodel:
                idxs = torch.arange(len(params))
            else:
                idxs = params[:, 0] == true_param[0]
            params = params[idxs]
            mu, sigma = self.deterministic(stage, params, ds, xps)
            if len(ys) == 1:
                ys = ys.expand(len(params), -1)
            # normal = torch.distributions.Normal(mu, sigma)
            # log_prob = normal.log_prob(ys)
            # log_prob = log_prob.sum(-1)
            log_prob = norm_logpdf(ys, mu, sigma)
            return log_prob
    
    def log_model_prior(self, *args, **kws):
        if self.multimodel:
            return math.log(1 / 3.0)
        else:
            return 0
    
    def log_poi_prior(self, params):
        return 0

    def log_goal_prior(self, *args, **kws):
        return 0
    
    def rvs(self, n_sample):
        with torch.no_grad():
            if self.multimodel:
                model_idxs = torch.randint(0, 3, size=(n_sample,))
                thetas = torch.rand(n_sample, 6)
                thetas[model_idxs < 2, -2:] = 0
                thetas[model_idxs < 1, -4:] = 0
                params = torch.cat([model_idxs.reshape(-1, 1), thetas], dim=1)
            else:
                thetas = torch.rand(n_sample, 4)
                params = thetas
            if self.include_str_wid:
                strength = torch.rand(n_sample, 1) * 5
                width = torch.rand(n_sample, 1) * 0.08 + 0.02
                etas = torch.cat([strength, width], dim=-1)
                params = torch.cat([params, etas], dim=1)
            wind_speed = torch.rand(n_sample, 1) * 20
            wind_angle = torch.rand(n_sample, 1) * math.pi * 2
            wind = torch.cat([wind_speed, wind_angle], dim=-1)
            params = torch.cat([params, wind], dim=1)
            if self.include_goal:
                goals = self.flux(params)
                params = torch.cat([params, goals], dim=1)
            return params

    def reward_fun(self, stage, ds_hist, ys_hist, xps_hist, params):
        if not self.include_cost or stage >= ds_hist.shape[1]:
            return 0
        else:
            c = self.cost_ratio
            return -c * torch.linalg.norm(ds_hist[:, stage, :] - xps_hist[:, stage, :], dim=-1)
        
    def xp_f(self, stage, xps, ds, ys):
        return ds.clone()