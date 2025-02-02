import torch
import torch.nn as nn
import torch.optim as optim
from .utils import *

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, nodes, act): 
        super().__init__()
        
        self.layers = len(nodes)
        modules     = []
        modules.append( nn.Linear(in_dim, nodes[0]) )
        modules.append( act)
        
        for i in range(self.layers-1):
            modules.append( nn.Linear(nodes[i], nodes[i+1]) )
            modules.append( act)
            
        modules.append( nn.Linear(nodes[self.layers-1], out_dim)) 
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x)

class cINN(nn.Module):  
    """
    The class for conditional Invertible Neural Network
    Parameters
    ----------
    n_theta: int
        Dimension of parameter space
    n_input : int
        Equals to the number of experiments  * (the dimension of observation + the dimension of design space)
    nodes_s; nodes_t : list
        The hidden layers and the activation function in s and t networks  
    act: torch.nn
        Activation functions
    Split1, Split2: list
        List of indexes that partition pois into two vectors of similar sizes 
    alpha: int
        Limit the value in exp(s()) not greater than alpha, avoid overflow
    Methods
    -------
    forward()
        Produces the transformed parameter in Gaussian space and the log-determinant along the transformation.
    """
    def __init__(self, n_theta, n_input,
                    nodes_s, nodes_t, act,  # the hidden layers and the activation function in s and t networks 
                    Split1, Split2,         # Split: list of indexes that partition pois into two vectors of similar sizes 
                    network=FCNN,
                    alpha = None            
                    ):
        super().__init__()
        self.theta_dim  = n_theta
        self.input_dim  = n_input 
        self.Split1     = Split1 
        self.Split2     = Split2 

        self.s1         = network( self.input_dim + len(self.Split2), len(self.Split1), nodes_s, act) 
        self.t1         = network( self.input_dim + len(self.Split2), len(self.Split1), nodes_t, act)
        self.s2         = network( self.input_dim + len(self.Split1), len(self.Split2), nodes_s, act)
        self.t2         = network( self.input_dim + len(self.Split1), len(self.Split2), nodes_t, act)
        self.alpha      = alpha

    def forward(self, inputs, thetas):             
        theta1, theta2  = thetas[:, self.Split1], thetas[:, self.Split2]
        theta2_y        = torch.cat((inputs, theta2), 1)  
        
        s1_trans        = self.s1(theta2_y)
        if self.alpha is not None:
            s1_trans = (2 * self.alpha / torch.pi) * torch.atan(s1_trans / self.alpha)
        t1_trans        = self.t1(theta2_y)
        theta1_trans    = theta1 * torch.exp(s1_trans) + t1_trans 

        theta1_trans_y  = torch.cat((inputs, theta1_trans), 1)    
        s2_trans        = self.s2(theta1_trans_y)
        if self.alpha is not None:
            s2_trans = (2 * self.alpha / torch.pi) * torch.atan(s2_trans / self.alpha)
        t2_trans        = self.t2(theta1_trans_y)
        theta2_trans    = theta2 * torch.exp(s2_trans) + t2_trans  
        
        transformed_theta = torch.cat((theta1_trans, theta2_trans), 1)
        log_det           = torch.sum(s1_trans, axis = 1) +  torch.sum(s2_trans, axis = 1)
        
        return transformed_theta, log_det
    
class NFs(nn.Module):
    """
    The class for Normalizing flows using cINN transformation
    Parameters
    ----------
    n_input : int
        Equals to the number of experiments  * (the dimension of observation + the dimension of design space)
    n_theta: int
        Dimension of parameter space
    Split1, Split2: list
        List of indexes that partition pois into two vectors of similar sizes 
    num_trans: int
        The number of entire transformation on all parameters. 
    Methods
    -------
    forward()
        Produces the approximating posterior logpdf of the pameters, conditioned on the previous experimental observation and designs.
    """
    def __init__(self, network, n_input, n_theta, Split1, Split2, num_trans): 
        super().__init__()
        self.n_theta    = n_theta
        
        if Split1 is None:
            self.Split1, self.Split2   = np.split(np.random.permutation(n_theta), [int(n_theta/2)])   
        else:
            self.Split1 =  np.array(Split1) 
            self.Split2 =  np.array(Split2) 
        
        self.flows        = [cINN(n_theta, n_input, [256, 256, 256], [256, 256, 256], nn.ReLU(), self.Split1, self.Split2, network, alpha = 10) for i in range(num_trans)]
       # self.flows        = [cINN(n_theta, n_input, [128, 128, 128], [128, 128, 128], nn.ReLU(), self.Split1, self.Split2, network, alpha = 10) for i in range(num_trans)]
       
       
        self.flows_list   = nn.ModuleList(self.flows) 
        self.feature_net  = network(n_input, n_input, [256, 256, 256], nn.ReLU())           
     #   self.feature_net  = network(n_input, n_input, [128, 128], nn.ReLU())                ##### sir

    def forward(self, inputs, thetas):                         
        inputs_feature  = self.feature_net(inputs)      
        thetas = thetas[:, :self.n_theta] 

        m, _    = thetas.shape
        log_det = torch.zeros(m)

        for flow in self.flows_list: 
            thetas,   ld = flow.forward(inputs_feature, thetas)                                   
            log_det    += ld
      
        norm_logprob  =  torch.distributions.MultivariateNormal(torch.zeros(thetas.shape[1]), 
                            torch.eye(thetas.shape[1])).log_prob(thetas)
        logpdf_NF  = norm_logprob + log_det
        logprobs   = logpdf_NF 
        logprobs   = torch.logaddexp(logprobs, torch.tensor(math.log(1e-27)) )
            
        return logprobs


class GMM_NET(nn.Module):
    def __init__(self, n_input, n_theta, n_mixture, mu_bounds, max_sigmas, truncnorm_info=None, activate=None):
        super().__init__()
        self.n_input = n_input
        self.n_theta = n_theta
        self.n_mixture = n_mixture
        if isinstance(mu_bounds, (list, tuple)):
            mu_bounds = torch.tensor(mu_bounds) # (n_theta, 2)
        self.mu_bounds = mu_bounds
        if isinstance(max_sigmas, (list, tuple)):
            max_sigmas = torch.tensor(max_sigmas) # (n_theta)
        self.max_sigmas = max_sigmas
        # truncnorm_info is a list or tuple, each element is [theta_idx, truncnorm_a, truncnorm_b]
        if truncnorm_info is None:
            truncnorm_info = []
        idxs_norm = list(range(n_theta))
        idxs_truncnorm = []
        truncnorm_bounds = []
        for info in truncnorm_info:
            idx = info[0]
            idxs_norm.remove(idx)
            idxs_truncnorm.append(idx)
            truncnorm_bounds.append(info[1:])
        self.idxs_norm = idxs_norm
        self.idxs_truncnorm = idxs_truncnorm
        self.truncnorm_bounds = torch.tensor(truncnorm_bounds)
        if activate is None:
            activate = nn.ReLU
        feature_dimns = [n_input, 256, 256]
        weight_dimns = [256, 256, 256, n_mixture]
        mu_dimns = [256, 256, 256, n_mixture * n_theta]
        sigma_dimns = [256, 256, 256, n_mixture * n_theta]
        feature_layers = []
        weight_layers = []
        mu_layers = []
        sigma_layers = []
        for i in range(len(feature_dimns) - 1):
            feature_layers.append(nn.Linear(feature_dimns[i], feature_dimns[i + 1]))
            feature_layers.append(activate())
        for i in range(len(weight_dimns) - 1):
            weight_layers.append(nn.Linear(weight_dimns[i], weight_dimns[i + 1]))
            if i < len(weight_dimns) - 2:
                weight_layers.append(activate())
        weight_layers.append(nn.LogSoftmax(dim=1))
        for i in range(len(mu_dimns) - 1):
            mu_layers.append(nn.Linear(mu_dimns[i], mu_dimns[i + 1]))
            if i < len(mu_dimns) - 2:
                mu_layers.append(activate())
        for i in range(len(sigma_dimns) - 1):
            sigma_layers.append(nn.Linear(sigma_dimns[i], sigma_dimns[i + 1]))
            if i < len(sigma_dimns) - 2:
                sigma_layers.append(activate())
        self.feature_net = nn.Sequential(*feature_layers)
        self.weight_net = nn.Sequential(*weight_layers)
        self.mu_net = nn.Sequential(*mu_layers)
        self.sigma_net = nn.Sequential(*sigma_layers)
        self.param_bounds = torch.tensor([-10, 10]).reshape(-1, 2)

    def forward(self, inputs, thetas):
        thetas = thetas[:, :self.n_theta]
        thetas = thetas.reshape(-1, 1, self.n_theta)
        features = self.feature_net(inputs)
        logweights = self.weight_net(features) # (N, n_mixture)
        mus = self.mu_net(features) # (N, n_mixture*n_theta)
        mus = torch.sigmoid(mus)
        sigmas = self.sigma_net(features)
        sigmas = torch.sigmoid(sigmas)

        mus = mus.reshape(-1, self.n_mixture, self.n_theta) # (N, n_mixture, n_theta)
        mus   = mus * (self.mu_bounds[:, 1] - self.mu_bounds[:, 0]) + self.mu_bounds[:, 0]
        sigmas = sigmas.reshape(-1, self.n_mixture, self.n_theta) # (N, n_mixture, n_theta)
        sigmas = sigmas * self.max_sigmas + 1e-5

        if len(self.idxs_norm) > 0:
            normal = torch.distributions.Normal(mus[..., self.idxs_norm], sigmas[..., self.idxs_norm])
            logprob_mixture_normal = normal.log_prob(thetas[..., self.idxs_norm])
        else:
            logprob_mixture_normal = 0

        if len(self.idxs_truncnorm) > 0:
            truncnorm = TruncatedNormal(mus[..., self.idxs_truncnorm], sigmas[..., self.idxs_truncnorm], 
                self.truncnorm_bounds[:, 0], self.truncnorm_bounds[:, 1])
            logprob_mixture_truncnorm = truncnorm.log_prob(thetas[..., self.idxs_truncnorm])
        else:
            logprob_mixture_truncnorm = 0

        logprob_mixture = (logprob_mixture_truncnorm + logprob_mixture_normal).sum(-1)
        logprob_mixture = torch.cat([logprob_mixture, torch.full((len(logprob_mixture), 1), math.log(1e-27))], dim=1)
        
        logweights = torch.cat([logweights, torch.zeros(len(logweights), 1)], dim=1)
        
        logprobs = torch.logsumexp(logprob_mixture + logweights, dim=-1)
        
        return logprobs


class POST_APPROX(object):
    def __init__(self,  n_stage, n_design, n_obs, 
        n_pois=None, # number of parameter of interest
        n_nuisps=None, # number of nuisance parameters
        n_goals=None, # number of goals
        model_weight=1, poi_weight=1, goal_weight=0,
        mu_bounds=None, max_sigmas=None, truncnorm_info=None, 
        n_mixture=8, n_trans=4, Split1=None, Split2=None, network=FCNN,
        activate=None, prior=None, 
        n_incre=1, share_interm_net=False,
        model_post_lrs=None, model_post_lr_scheduler_gammas=None,
        poi_post_lrs=None, poi_post_lr_scheduler_gammas=None,
        goal_post_lrs=None, goal_post_lr_scheduler_gammas=None,
        use_NFs=False, log_every=1, dowel=None):
        
        self.log_every = log_every
        self.dowel = dowel

        self.n_stage = n_stage
        self.n_design = n_design
        self.n_obs = n_obs
        assert n_pois is not None or n_nuisps is not None or n_goals is not None

        # check the number of parameters of interest
        def check_param_num(ns):
            if ns is None:
                return []
            elif isinstance(ns, int):
                return [ns]
            else:
                return ns
        def check_model_num(n_model, ns):
            if n_model == 0:
                return len(ns)
            else:
                assert n_model == len(ns) or len(ns) == 0
                return n_model
        self.n_pois = check_param_num(n_pois)
        self.n_model = len(self.n_pois)
        self.n_poi_max = max(self.n_pois + [0])
        self.n_nuisps = check_param_num(n_nuisps)
        self.n_model = check_model_num(self.n_model, self.n_nuisps)
        self.n_nuisp_max = max(self.n_nuisps + [0])
        self.n_goals = check_param_num(n_goals)
        self.n_model = check_model_num(self.n_model, self.n_goals)
        self.n_goal_max = max(self.n_goals + [0])

        assert self.n_model > 0
        if self.n_model == 1:
            model_weight = 0
        if self.n_poi_max == 0:
            poi_weight = 0
        if self.n_goal_max == 0:
            goal_weight = 0
        assert model_weight != 0 or poi_weight != 0 or goal_weight != 0
        self.model_weight = model_weight
        self.poi_weight = poi_weight
        self.goal_weight = goal_weight

        assert mu_bounds is not None
        assert max_sigmas is not None
        if truncnorm_info is None:
            truncnorm_info = {
            'poi': [None] * self.n_model,
            'goal': [None] * self.n_model}
        self.mu_bounds = mu_bounds
        self.max_sigmas = max_sigmas
        self.truncnorm_info = truncnorm_info

        if activate is None:
            activate = nn.ReLU
        self.prior = prior

        assert n_incre <= n_stage
        self.n_incre = n_incre
        stages_incre = [int(n_stage * (i + 1) / n_incre) - 1 for i in range(n_incre)]
        self.stages_incre = stages_incre
        if self.dowel is not None:
            self.dowel.logger.log('Learn the posterior approximator at stages: ' + str(stages_incre))
        if self.n_incre == 1:
            share_interm_net = False
        self.share_interm_net = share_interm_net

        self.use_NFs = use_NFs
        if self.use_NFs:
            self.network = network
            self.num_trans = n_trans 
            self.Split1   = Split1
            self.Split2   = Split2

        candidate_lrs = [[1e-3] * n_incre, [1e-4] * (n_incre - 1) + [1e-3]]
        candidate_gammas = [[0.9999] * n_incre, [0.999999] * (n_incre - 1) + [0.9999]]
        if model_post_lrs is None:
            model_post_lrs = candidate_lrs[share_interm_net]
        if model_post_lr_scheduler_gammas is None:
            model_post_lr_scheduler_gammas = candidate_gammas[share_interm_net]
        if poi_post_lrs is None:
            poi_post_lrs = candidate_lrs[share_interm_net]
        if poi_post_lr_scheduler_gammas is None:
            poi_post_lr_scheduler_gammas = candidate_gammas[share_interm_net]
        if goal_post_lrs is None:
            goal_post_lrs = candidate_lrs[share_interm_net]
        if goal_post_lr_scheduler_gammas is None:
            goal_post_lr_scheduler_gammas = candidate_gammas[share_interm_net]
        
        if self.model_weight != 0:
            self.model_post_nets = {}
            self.model_post_optimizers = {}
            self.model_post_schedulers = {}
            for i, stage in enumerate(stages_incre):
                if stage == stages_incre[-1] or not share_interm_net:
                    model_post_dimns = [(stage + 1) * (self.n_design + self.n_obs), 256, 256, 256, self.n_model]
                else:
                    model_post_dimns = [(n_incre - 1) * share_interm_net + (stages_incre[-2] + 1) * (self.n_design + self.n_obs), 256, 256, 256, self.n_model]
                model_post_layers = []
                for ii in range(len(model_post_dimns) - 1):
                    model_post_layers.append(nn.Linear(model_post_dimns[ii], model_post_dimns[ii + 1]))
                    if ii < len(model_post_dimns) - 2:
                        model_post_layers.append(activate())
                model_post_layers.append(nn.LogSoftmax(dim=1))
                if stage == stages_incre[0] or stage == stages_incre[-1] or not share_interm_net:
                    post_net = nn.Sequential(*model_post_layers)
                    post_optimizer = optim.Adam(post_net.parameters(), lr=model_post_lrs[i])
                    post_scheduler = torch.optim.lr_scheduler.ExponentialLR(post_optimizer, gamma=model_post_lr_scheduler_gammas[i])
                self.model_post_nets[stage] = post_net
                self.model_post_optimizers[stage] = post_optimizer
                self.model_post_schedulers[stage] = post_scheduler
            
        if self.poi_weight != 0:
            self.poi_post_nets = {}
            self.poi_post_optimizers = {}
            self.poi_post_schedulers = {}
            for i, stage in enumerate(stages_incre):
                if stage == stages_incre[0] or stage == stages_incre[-1] or not share_interm_net:
                    post_nets = []
                    post_optimizers = []
                    post_schedulers = []
                    for j, n_poi in enumerate(self.n_pois):
                        h = stage + 1 if stage == stages_incre[-1] or not share_interm_net else stages_incre[-2] + 1
                        input_dimn = (n_incre - 1) * share_interm_net * (stage != stages_incre[-1]) + h * (self.n_design + self.n_obs)
                        if not self.use_NFs:
                            post_net = GMM_NET(input_dimn, n_poi, n_mixture, mu_bounds['poi'][j], max_sigmas['poi'][j], truncnorm_info['poi'][j], activate)
                        else:
                            ###########################################
                            post_net = NFs(self.network, input_dimn, n_poi, self.Split1, self.Split2,  self.num_trans)
                            ###########################################
                        optimizer = optim.Adam(post_net.parameters(), lr=poi_post_lrs[i])
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=poi_post_lr_scheduler_gammas[i])
                        post_nets.append(post_net)
                        post_optimizers.append(optimizer)
                        post_schedulers.append(scheduler)
                    self.poi_post_nets[stage] = post_nets
                    self.poi_post_optimizers[stage] = post_optimizers
                    self.poi_post_schedulers[stage] = post_schedulers
                else:
                    self.poi_post_nets[stage] = post_nets
                    self.poi_post_optimizers[stage] = post_optimizers
                    self.poi_post_schedulers[stage] = post_schedulers

        if self.goal_weight != 0:
            self.goal_post_nets = {}
            self.goal_post_optimizers = {}
            self.goal_post_schedulers = {}
            for i, stage in enumerate(stages_incre):
                if stage == stages_incre[0] or stage == stages_incre[-1] or not share_interm_net:
                    post_nets = []
                    post_optimizers = []
                    post_schedulers = []
                    for j, n_goal in enumerate(self.n_goals):
                        h = stage + 1 if stage == stages_incre[-1] or not share_interm_net else stages_incre[-2] + 1
                        input_dimn = (n_incre - 1) * share_interm_net * (stage != stages_incre[-1]) + h * (self.n_design + self.n_obs)
                        if not self.use_NFs:
                            post_net = GMM_NET(input_dimn, n_goal, n_mixture, mu_bounds['goal'][j], max_sigmas['goal'][j], truncnorm_info['goal'][j], activate)
                        else:
                            ###########################################
                            post_net = NFs(self.network, input_dimn, n_goal, self.Split1, self.Split2,  self.num_trans)
                            ###########################################
                        optimizer = optim.Adam(post_net.parameters(), lr=goal_post_lrs[i])
                        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=goal_post_lr_scheduler_gammas[i])
                        post_nets.append(post_net)
                        post_optimizers.append(optimizer)
                        post_schedulers.append(scheduler)
                    self.goal_post_nets[stage] = post_nets
                    self.goal_post_optimizers[stage] = post_optimizers
                    self.goal_post_schedulers[stage] = post_schedulers
                else:
                    self.goal_post_nets[stage] = post_nets
                    self.goal_post_optimizers[stage] = post_optimizers
                    self.goal_post_schedulers[stage] = post_schedulers
                
    def form_input(self, stage, ds_hist, ys_hist):
        """
        Form the input of the posterior approximation network.
        """
        if stage >= self.stages_incre[-1] or not self.share_interm_net:
            X = torch.cat([ds_hist[:, :stage+1].view(len(ds_hist), -1), 
                ys_hist[:, :stage+1].view(len(ys_hist), -1)], dim=1)
            return X
        else:
            idx = self.stages_incre.index(stage)
            X0 = torch.zeros(len(ds_hist), self.n_incre - 1)
            X0[:, idx] = 1.0
            X1 = torch.zeros(len(ds_hist), (self.stages_incre[-2] + 1) * (self.n_design))
            X2 = torch.zeros(len(ys_hist), (self.stages_incre[-2] + 1) * (self.n_obs))
            X1[:, :(stage + 1) * self.n_design] = ds_hist[:, :stage+1].view(len(ds_hist), -1)
            X2[:, :(stage + 1) * self.n_obs] = ys_hist[:, :stage+1].view(len(ys_hist), -1)
            return torch.cat([X0, X1, X2], dim=1)
                 
    def log_post(self, stage=None, ds_hist=None, ys_hist=None, params=None, inputs=None, which='poi'):
        """
        Get the predictions of the posterior approximation network.
        """
        if stage is None:
            stage = self.stages_incre[-1]
        if inputs is not None:
            X = inputs
        else:
            X = self.form_input(stage, ds_hist, ys_hist)
        if which == 'model':
            assert self.n_model > 1 and self.model_weight != 0
            logpost = self.model_post_nets[stage](X)
            if params is not None:
                model_idxs = params[:, 0].to(int)
                logpost = logpost[range(len(logpost)), model_idxs]
        else:
            if self.n_model > 1:
                model_idxs = params[:, 0].to(int)
            else:
                model_idxs = torch.zeros(len(X)).to(int)
            if which == 'poi':
                assert self.n_poi_max > 0 and self.poi_weight != 0
                params = params[:, 1:1+self.n_poi_max] if self.n_model > 1 else params[:, :self.n_poi_max]
            elif which == 'goal':
                assert self.n_goal_max > 0 and self.goal_weight != 0
                params = params[:, -self.n_goal_max:]
            logpost = torch.zeros(len(X))
            for m_idx in range(self.n_model):
                idxs = (model_idxs == m_idx)
                if which == 'poi':
                    logpost[idxs] = self.poi_post_nets[stage][m_idx](X[idxs], params[idxs])
                elif which == 'goal':
                    logpost[idxs] = self.goal_post_nets[stage][m_idx](X[idxs], params[idxs])
        return logpost

    def log_model_post(self, stage=None, ds_hist=None, ys_hist=None, params=None, inputs=None):
        """
        Get the model posterior.
        """
        return self.log_post(stage, ds_hist, ys_hist, params, inputs, 'model')

    def log_poi_post(self, stage=None, ds_hist=None, ys_hist=None, params=None, inputs=None):
        """
        Get the PoI posterior
        """
        return self.log_post(stage, ds_hist, ys_hist, params, inputs, 'poi')

    def log_goal_post(self, stage=None, ds_hist=None, ys_hist=None, params=None, inputs=None):
        """
        Get the QoI posterior.
        """
        return self.log_post(stage, ds_hist, ys_hist, params, inputs, 'goal')
    
    def train(self, ds_hist, ys_hist, xps_hist, params, l, n_update=3):
        """
        Train the posterior approximation networks.
        """
        for stage in self.stages_incre:
            X = self.form_input(stage, ds_hist, ys_hist)
            for t in range(n_update):
                # train post net
                if self.model_weight != 0:
                    log_model_post = self.log_model_post(stage=stage, params=params, inputs=X)
                    loss = torch.mean(-log_model_post)
                    self.model_post_optimizers[stage].zero_grad()
                    loss.backward()
                    self.model_post_optimizers[stage].step()
                if self.poi_weight != 0:
                    log_poi_post = self.log_poi_post(stage=stage, params=params, inputs=X)
                    loss = torch.mean(-log_poi_post)
                    for optimizer in self.poi_post_optimizers[stage]:
                        optimizer.zero_grad()
                    loss.backward()  
                    for optimizer in self.poi_post_optimizers[stage]:
                        optimizer.step()
                if self.goal_weight != 0:
                    log_goal_post = self.log_goal_post(stage=stage, params=params, inputs=X)
                    loss = torch.mean(-log_goal_post)
                    for optimizer in self.goal_post_optimizers[stage]:
                        optimizer.zero_grad()
                    loss.backward()  
                    for optimizer in self.goal_post_optimizers[stage]:
                        optimizer.step()
            if self.model_weight != 0:
                self.model_post_schedulers[stage].step()
            if self.poi_weight != 0:
                for scheduler in self.poi_post_schedulers[stage]:
                    scheduler.step()
            if self.goal_weight != 0:
                for scheduler in self.goal_post_schedulers[stage]:
                    scheduler.step()
        if l % self.log_every == 0 and self.dowel is not None:
            if self.model_weight != 0:
                self.dowel.logger.log(f'log model post after training: {log_model_post.mean()}')
            if self.poi_weight != 0:
                self.dowel.logger.log(f'log param_of_interest post after training: {log_poi_post.mean()}')
            if self.goal_weight != 0:
                self.dowel.logger.log(f'log goal post after training: {log_goal_post.mean()}')
        
    def reward_fun(self, stage, ds_hist, ys_hist, xps_hist, params):
        """
        Calculate the information gain rewards.
        """
        with torch.no_grad():
            if stage == 0:
                try:
                    self.log_model_prior = self.prior.log_model_prior(params)
                except:
                    self.log_model_prior = 0
                try:
                    self.log_poi_prior = self.prior.log_poi_prior(params)
                except:
                    self.log_poi_prior = 0
                try:
                    self.log_goal_prior = self.prior.log_goal_prior(params)
                except:
                    self.log_goal_prior = 0
            if stage in self.stages_incre:
                X = self.form_input(stage, ds_hist, ys_hist)
                if self.model_weight != 0:
                    log_model_post = self.log_model_post(stage=stage, params=params, inputs=X)
                    model_kld = log_model_post - self.log_model_prior
                    self.log_model_prior = log_model_post
                else:
                    model_kld = 0
                if self.poi_weight != 0:
                    log_poi_post = self.log_poi_post(stage=stage, params=params, inputs=X)
                    poi_kld = log_poi_post - self.log_poi_prior
                    self.log_poi_prior = log_poi_post
                else:
                    poi_kld = 0
                if self.goal_weight != 0:
                    log_goal_post = self.log_goal_post(stage=stage, params=params, inputs=X)
                    goal_kld = log_goal_post - self.log_goal_prior
                    self.log_goal_prior = log_goal_post
                else:
                    goal_kld = 0
                return self.model_weight * model_kld + self.poi_weight * poi_kld + self.goal_weight * goal_kld
            else:
                return 0