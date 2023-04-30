import torch
import math

class SOURCE(object):
    def __init__(self, multimodel=False, include_nuisance=False, include_goal=False):
        self.multimodel = multimodel
        self.include_nuisance = include_nuisance
        self.include_goal = include_goal
    
    def deterministic(self, params, ds, xps=None):

        d1     = ds[:, 0:1]
        d2     = ds[:, 1:2]

        if not self.multimodel:
            if self.include_nuisance:
                base_signal = params[:, 4:5]
                max_signal = params[:, 5:6]
            else:
                base_signal = 0.1
                max_signal  = 1e-4
            
            theta1 = params[:, 0:1]
            theta2 = params[:, 1:2]
            theta3 = params[:, 2:3]
            theta4 = params[:, 3:4]

            G = ( base_signal 
                + 1/( (theta1-d1)**2 + (theta2-d2)**2 + max_signal)
                + 1/( (theta3-d1)**2 + (theta4-d2)**2 + max_signal) )
        else:
            model_idxs = params[:, 0:1]
            theta1 = params[:, 1:2]
            theta2 = params[:, 2:3]
            theta3 = params[:, 3:4]
            theta4 = params[:, 4:5]
            theta5 = params[:, 5:6]
            theta6 = params[:, 6:7]
            if self.include_nuisance:
                base_signal = params[:, 7:8]
                max_signal = params[:, 8:9]
            else:
                base_signal = 0.1
                max_signal  = 1e-4

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
        d1 = 4.0
        if not self.multimodel:
            if self.include_nuisance:
                base_signal = params[:, 4:5]
                max_signal = params[:, 5:6]
            else:
                base_signal = 0.1
                max_signal  = 1e-4
            theta1 = params[:, 0:1]
            theta2 = params[:, 1:2]
            theta3 = params[:, 2:3]
            theta4 = params[:, 3:4]

            tmp = max_signal + (theta_1 - d1) ** 2
            flux_1 = -(theta_1 - d1) * math.pi * torch.sqrt(tmp) / tmp ** 2 
            tmp = max_signal + (theta_3 - d1) ** 2
            flux_2 = -(theta_3 - d1) * math.pi * torch.sqrt(tmp) / tmp ** 2 
            flux = flux_1 + flux_2
        else:
            if self.include_nuisance:
                base_signal = params[:, 7:8]
                max_signal = params[:, 8:9]
            else:
                base_signal = 0.1
                max_signal  = 1e-4
            model_idxs = params[:, 0:1]
            theta1 = params[:, 1:2]
            theta2 = params[:, 2:3]
            theta3 = params[:, 3:4]
            theta4 = params[:, 4:5]
            theta5 = params[:, 5:6]
            theta6 = params[:, 6:7]

            tmp = max_signal + (theta_1 - d1) ** 2
            flux_1 = -(theta_1 - d1) * math.pi * torch.sqrt(tmp) / tmp ** 2 
            tmp = max_signal + (theta_3 - d1) ** 2
            flux_2 = -(theta_3 - d1) * math.pi * torch.sqrt(tmp) / tmp ** 2 * (model_idxs >= 1)
            tmp = max_signal + (theta_5 - d1) ** 2
            flux_3 = -(theta_5 - d1) * math.pi * torch.sqrt(tmp) / tmp ** 2 * (model_idxs >= 2)
            flux = flux_1 + flux_2 + flux_3
        return flux
    
    def loglikeli(self, stage, ys, params, ds, xps=None, true_param=None):
        """
        y : (n or 1, n_obs)
        theta : (n or 1, n_param)
        d : (n or 1, n_design)
        """
        with torch.no_grad():
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
            return torch.log(1 / 3.0)
        else:
            return 0
    
    def log_poi_prior(self, params):
        if self.include_nuisance:
            return 0
        else:
            if not self.multimodel:
                thetas = params[:, :4]
                log_prob = norm_logpdf(thetas, 0, 1)
            else:
                model_idxs = params[:, 0]
                thetas = prior_samples[:, 1:7].reshape(-1, 3, 2)
                log_probs = norm_logpdf(theta, 0, 1)
                log_prob = torch.zeros(len(log_probs))
                log_prob[model_idxs == 0] = log_probs[model_idxs == 0][:, 0]
                log_prob[model_idxs == 1] = log_probs[model_idxs == 1][:, :2].sum(-1)
                log_prob[model_idxs == 2] = log_probs[model_idxs == 2].sum(-1)
        return log_prob

    def log_goal_prior(self, *args, **kws):
        return 0
    
    def rvs(self, n_sample):
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







