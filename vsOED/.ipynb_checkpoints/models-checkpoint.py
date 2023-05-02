import torch
import math
from .utils import *

class SOURCE(object):
    def __init__(self, multimodel=False, include_nuisance=False, include_goal=False):
        self.multimodel = multimodel
        self.include_nuisance = include_nuisance
        self.include_goal = include_goal
    
    def extract_params(self, params):
        base_signal = 0.1
        max_signal  = 1e-4
        if not self.multimodel:
            if self.include_nuisance:
                base_signal = params[:, 4:5]
                max_signal = params[:, 5:6]
            theta1 = params[:, 0:1]
            theta2 = params[:, 1:2]
            theta3 = params[:, 2:3]
            theta4 = params[:, 3:4]
            theta5 = None
            theta6 = None
            model_idxs = None
        else:
            if self.include_nuisance:
                base_signal = params[:, 7:8]
                max_signal = params[:, 8:9]
            model_idxs = params[:, 0:1]
            theta1 = params[:, 1:2]
            theta2 = params[:, 2:3]
            theta3 = params[:, 3:4]
            theta4 = params[:, 4:5]
            theta5 = params[:, 5:6]
            theta6 = params[:, 6:7]
        return (base_signal, max_signal, model_idxs, 
            theta1, theta2, theta3, theta4, theta5, theta6)

    def deterministic(self, params, ds, xps=None):

        d1     = ds[:, 0:1]
        d2     = ds[:, 1:2]

        if not self.multimodel:
            (base_signal, max_signal, _,
            theta1, theta2, theta3, theta4, _, _) = self.extract_params(params)

            G = ( base_signal 
                + 1/( (theta1-d1)**2 + (theta2-d2)**2 + max_signal)
                + 1/( (theta3-d1)**2 + (theta4-d2)**2 + max_signal) )
        else:
            (base_signal, max_signal, model_idxs,
            theta1, theta2, theta3, theta4, theta5, theta6) = self.extract_params(params)

            G = ( base_signal 
                + 1/( (theta1-d1)**2 + (theta2-d2)**2 + max_signal)
                + ( 1/( (theta3-d1)**2 + (theta4-d2)**2 + max_signal ) ) * (model_idxs >= 1) 
                + ( 1/( (theta5-d1)**2 + (theta6-d2)**2 + max_signal ) ) * (model_idxs >= 2) )

        return torch.log(G), torch.tensor(0.5)
        
    def model(self, stage, params, ds, xps=None):

        with torch.no_grad():
            mu, sigma = self.deterministic(params, ds, xps)
            y = torch.randn(mu.shape) * sigma + mu
        return y

    def flux(self, params):
        with torch.no_grad():
            d1 = 6.0
            if not self.multimodel:
                (base_signal, max_signal, _,
                theta1, theta2, theta3, theta4, _, _) = self.extract_params(params)

                tmp = max_signal + (theta1 - d1) ** 2
                flux_1 = -(theta1 - d1) * math.pi * torch.sqrt(tmp) / tmp ** 2 
                tmp = max_signal + (theta3 - d1) ** 2
                flux_2 = -(theta3 - d1) * math.pi * torch.sqrt(tmp) / tmp ** 2 
                flux = flux_1 + flux_2
            else:
                (base_signal, max_signal, model_idxs,
                theta1, theta2, theta3, theta4, theta5, theta6) = self.extract_params(params)

                tmp = max_signal + (theta1 - d1) ** 2
                flux_1 = -(theta1 - d1) * math.pi * torch.sqrt(tmp) / tmp ** 2 
                tmp = max_signal + (theta3 - d1) ** 2
                flux_2 = -(theta3 - d1) * math.pi * torch.sqrt(tmp) / tmp ** 2 * (model_idxs >= 1)
                tmp = max_signal + (theta5 - d1) ** 2
                flux_3 = -(theta5 - d1) * math.pi * torch.sqrt(tmp) / tmp ** 2 * (model_idxs >= 2)
                flux = flux_1 + flux_2 + flux_3
            return torch.log(torch.abs(flux))
    
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
            mu, sigma = self.deterministic(params, ds, xps)
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
        with torch.no_grad():
            if self.include_nuisance:
                return 0
            else:
                if not self.multimodel:
                    thetas = params[:, :4]
                    log_prob = norm_logpdf(thetas, 0, 1)
                else:
                    model_idxs = params[:, 0]
                    thetas = params[:, 1:7].reshape(-1, 3, 2)
                    log_probs = norm_logpdf(thetas, 0, 1)
                    log_prob = torch.zeros(len(log_probs))
                    log_prob[model_idxs == 0] = log_probs[model_idxs == 0][:, 0]
                    log_prob[model_idxs == 1] = log_probs[model_idxs == 1][:, :2].sum(-1)
                    log_prob[model_idxs == 2] = log_probs[model_idxs == 2].sum(-1)
            return log_prob

    def log_goal_prior(self, *args, **kws):
        return 0
    
    def rvs(self, n_sample):
        with torch.no_grad():
            if self.multimodel:
                model_idxs = torch.randint(0, 3, size=(n_sample,))
                thetas = torch.randn(n_sample, 6)
                thetas[model_idxs < 2, -2:] = 0
                thetas[model_idxs < 1, -4:] = 0
                params = torch.cat([model_idxs.reshape(-1, 1), thetas], dim=1)
            else:
                thetas = torch.randn(n_sample, 4)
                params = thetas
            if self.include_nuisance:
                base_signals = torch.rand(n_sample, 1) * 0.15 + 0.05
                max_signals = torch.rand(n_sample, 1) * (2e-4 - 5e-5) + 5e-5
                etas = torch.cat([base_signals, max_signals], dim=-1)
                params = torch.cat([params, etas], dim=1)
            if self.include_goal:
                goals = self.flux(params)
                params = torch.cat([params, goals], dim=1)
            return params

    def reward_fun(self, *args, **kws):
        return 0

class CES(object):
    def __init__(self):
        self.beta = torch.distributions.beta.Beta(1, 1)
        self.dirichlet = torch.distributions.dirichlet.Dirichlet(torch.tensor([1.0, 1.0, 1.0]))
        self.normal = torch.distributions.normal.Normal(1, 3)
    
    def deterministic(self, theta, d, xp=None):
        tau = 0.005
        
        rho = theta[:, 0:1] * 0.99 + 0.01
        alpha = theta[:, 1:3]
        alpha = torch.cat([alpha, 1 - alpha.sum(-1, keepdim=True)], dim=-1)
        logu = theta[:, 3:4]
        u = torch.exp(logu)

        d1     = d[:, 0:3] * 100
        d2     = d[:, 3:6] * 100

        U1 = (((d1 ** rho) * alpha).sum(-1, keepdim=True)) ** (1 / rho)
        U2 = (((d2 ** rho) * alpha).sum(-1, keepdim=True)) ** (1 / rho)
        mu_eta = (U1 - U2) * u
        sigma_eta = (1 + torch.linalg.norm(d1 - d2, ord=2, dim=-1, keepdim=True)) * tau * u
        return mu_eta, sigma_eta
        
    def model(self, stage, theta, d,  xp=None):

        with torch.no_grad():
            epsilon = 2**(-22)
            lower = torch.log(torch.tensor(epsilon / (1 - epsilon)))
            upper = -lower
            mu_eta, sigma_eta = self.deterministic(theta, d, xp)
            eta = torch.randn(mu_eta.shape) * sigma_eta + mu_eta

            y = torch.clamp(eta, lower, upper)
            y = y / upper

            return y
    
    def loglikeli(self, stage, y, theta, d, xp=None, true_theta=None):
        """
        y : (n or 1, n_obs)
        theta : (n or 1, n_param)
        d : (n or 1, n_design)
        """
        with torch.no_grad():
            crit = 1e-40
            epsilon = 2**(-22)
            lower = torch.log(torch.tensor(epsilon / (1 - epsilon)))
            upper = -lower
            mu_eta, sigma_eta = self.deterministic(theta, d, xp)
            y_eta = y * upper
            if len(y_eta) == 1:
                y_eta = y_eta.expand(len(theta), -1)
            mask_nan = torch.isnan(mu_eta)
            mu_eta[mask_nan] = 0
            normal = torch.distributions.Normal(mu_eta, sigma_eta)
            lower_logcdf = normal.cdf(torch.tensor([[lower]]))
            upper_logcdf = normal.cdf(2 * mu_eta - upper)
            mask_lower = lower_logcdf < crit
            mask_upper = upper_logcdf < crit
            asymptotic_lower = normal.log_prob(torch.tensor([[lower]])) - torch.log(crit + torch.abs((lower - mu_eta) / sigma_eta))
            asymptotic_upper = normal.log_prob(2 * mu_eta - upper) - torch.log(crit + torch.abs((2 * mu_eta - upper - mu_eta) / sigma_eta))
            lower_logcdf[mask_lower] = 1
            upper_logcdf[mask_upper] = 1
            lower_logcdf = torch.log(lower_logcdf)
            upper_logcdf = torch.log(upper_logcdf)
            lower_logcdf[mask_lower] = asymptotic_lower[mask_lower]
            upper_logcdf[mask_upper] = asymptotic_upper[mask_upper]
            log_prob = normal.log_prob(y_eta) + math.log(upper)
            idxs_upper = y_eta == upper #torch.abs(y_eta - upper) < 1e-6
            log_prob[idxs_upper] = upper_logcdf[idxs_upper]
            idxs_lower = y_eta == lower # torch.abs(y_eta - lower) < 1e-6
            log_prob[idxs_lower] = lower_logcdf[idxs_lower]
            log_prob[mask_nan] = -1000000
            log_prob = log_prob.sum(-1)
#         print(log_prob.max())
            return log_prob
    
    
    def log_poi_prior(self, params):
        """
        A function to evaluate the prior logpdf of parameter samples.
        
        Parameters
        ----------
        prior_samples : numpy.ndarray of size (n_sample, n_param)
        """
        with torch.no_grad():
            rho = params[:, 0]
            alpha = params[:, 1:3]
            alpha = torch.cat([alpha, 1 - alpha.sum(-1, keepdim=True)], dim=-1)
            alpha[alpha == 0.] = 1e-6
            alpha /= alpha.sum(-1, keepdim=True)
            logu = params[:, 3]

            rho_logpdf = self.beta.log_prob(rho).view(-1)
            idxs_valid = alpha[:, -1] >= 0
            alpha_logpdf = torch.zeros(len(params)) - 10000
            alpha_logpdf[idxs_valid] = self.dirichlet.log_prob(alpha[idxs_valid]).view(-1)
            logu_logpdf = self.normal.log_prob(logu).view(-1)

            return rho_logpdf + alpha_logpdf + logu_logpdf
        
    
    def rvs(self, n_sample):
        """
        A function to generate samples from the prior.
        
        Parameters
        ----------
        n_sample : int
        """
        with torch.no_grad():
            rho = self.beta.rsample((n_sample, 1))
            alpha = self.dirichlet.rsample((n_sample, ))
            logu = self.normal.rsample((n_sample, 1))
            return torch.cat([rho, alpha[:, :2], logu], dim=-1)
        
        
    def reward_fun(self, *args, **kws):
        return 0
    
    
class SIR(object):
    def __init__(self, foldername):
        self.foldername = foldername
        self.mode = 'train'
        self.train_epoch = -1
        self.resample_epochs = 2000
        self.load_train_data(foldername, 0)
    
    def reset_train(self):
        self.mode = 'train'
        self.train_epoch = -1
        
    def reset_test(self):
        self.mode = 'eval'
        
    def xp_f(self, xps, stage, ds, ys):
        return (100 - xps) * ds / 100 + xps
        
    def grid(self, d, xp=None):
        if xp is None:
            grid = (d * 100).int() + 1
        else:
            new_xp = self.xp_f(xp, None, d, None)
            grid = (new_xp * 100).int() + 1
        grid = torch.minimum(grid, torch.tensor(10000))
        return grid
        
    def model(self, stage, theta, d,  xp):
        with torch.no_grad():
            # if self.mode == 'train':
            idxs = self.idxs
            grid = self.grid(d, xp).reshape(-1) # (n_sample)
            assert len(grid) == len(idxs)
            ys = self.data['ys'][idxs.to('cpu')][torch.arange(len(grid)).to('cpu'), 
                                                 grid.to('cpu')].to(idxs.device).reshape(-1, 1)
            # elif self.mode == 'eval':
                # grid = self.grid(d, xp).reshape(-1)
                # ys = self.data['ys'][torch.arange(len(grid)), grid].reshape(-1, 1)

        return ys
    
    def load_train_data(self, foldername, idx=None):
        if idx is not None:
            self.data = torch.load(self.foldername + f'/train_data/sir_train_data_{idx}.pt')
        else:
            self.data = torch.load(self.foldername + f'/train_data/sir_train_data_{self.train_epoch // self.resample_epochs}.pt')
        self.mode = 'train'
    
    def load_test_data(self, foldername):
        self.data = torch.load(self.foldername + f'/test_data/sir_test_samples.pt')
        # self.hists = torch.load(self.foldername + f'test_data/sir_test_total_hists.pt')
        self.mode = 'eval'
        self.reset_test()
        
    def loglikeli(self, stage, y, theta, d, xp):
        """
        y : (n or 1, n_obs)
        theta : (n or 1, n_param)
        d : (n or 1, n_design)
        """
        assert self.mode == 'eval'
        with torch.no_grad():
            if stage == 0:
                self.test_idx += 1
            n = len(theta)
            grid = self.grid(d, xp).reshape(-1)[0].item() # int
            locs = self.hists['hist_locs'][:, grid] # (n_constr_sample + 1, bins + 3)
            probs = self.hists['hist_probs'][:, grid] # (n_constr_sample + 1, bins + 2)
            idxs = torch.searchsorted(locs, y.expand(n, 1), right=True)
            idxs = idxs.reshape(-1) - 1
            log_prob = probs[np.arange(n), idxs].log()
            if log_prob[self.test_idx] == -float('Inf'):
                idxs = torch.searchsorted(locs, y.expand(n, 1), right=False)
                idxs = idxs.reshape(-1) - 1
                log_prob = probs[np.arange(n), idxs].log()
                
            log_prob[0], log_prob[self.test_idx] = log_prob[self.test_idx], log_prob[0]
            
        return log_prob

    def logpdf(self, prior_samples):
        idxs = self.idxs
        return self.data['log_probs'][idxs.to('cpu')].to(idxs.device)
    
    def rvs(self, n_sample):
        if self.mode == 'train':
            if self.train_epoch > 0 and self.train_epoch % self.resample_epochs == 0:
                self.load_train_data(self.foldername)
            self.train_epoch += 1
        elif self.mode == 'eval':
            assert n_sample <= self.data['num_samples']
        p = torch.ones(self.data['num_samples']) / self.data['num_samples']
        idxs = torch.multinomial(p, n_sample, replacement=False)
        self.idxs = idxs
        return self.data['log_prior_samples'][idxs.to('cpu')].to(idxs.device)
        
    def reward_fun(self, *args, **kws):
        return 0