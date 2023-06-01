import numpy as np
from .utils import *
from functools import partial

class VSOED(object):
    def __init__(self, 
                 n_stage, n_param, n_design, n_obs, 
                 model, prior, design_bounds,
                 nkld_reward_fun=None, kld_reward_fun=None,
                 phys_state_info=None, post_approx=None, random_state=None):
        
        set_random_seed(random_state)
        
        assert isinstance(n_stage, int) and n_stage > 0, (
               "n_stage should be an integer greater than 0.")
        self.n_stage = n_stage
        assert isinstance(n_param, int) and n_param > 0, (
              "n_param should be an integer greater than 0.")
        self.n_param = n_param
        assert isinstance(n_design, int) and n_design > 0, (
               "n_design should be an integer greater than 0.")
        self.n_design = n_design
        assert isinstance(n_obs, int) and n_obs > 0, (
               "n_obs should be an integer greater than 0.")
        self.n_obs = n_obs
        
        self.model = model
        self.m_f = model.model
        self.model_fun = self.m_f
        self.loglikeli = model.loglikeli

        self.prior = prior
        self.prior_rvs = prior.rvs

        assert isinstance(design_bounds, (list, tuple)), (
               "design_bounds should be a list or tuple of " 
               "size (n_design, 2).")
        assert len(design_bounds) == n_design, (
               "Length of design_bounds should equal n_design.")
        for i in range(n_design):
            assert len(design_bounds[i]) == 2, (
                   "Each entry of prior_info is of size 2, including "
                   "lower bound and upper bound.")
            l_b, u_b = design_bounds[i]
            assert isinstance(l_b, (int, float)), (
                   "{}-th lower bound should be a number.".format(i))
            assert isinstance(u_b, (int, float)), (
                   "{}-th upper_bound should be a number.".format(i))
        # size (n_design, 2)
        self.design_bounds = torch.tensor(design_bounds)

        # Non-KL-divergence based reward function
        if nkld_reward_fun is None:
            self.nkld_rw_f = partial(tmp_fun, ret='0')
        else:
            assert callable(nkld_reward_fun), (
                   "nkld_reward_fun should be a function.")
            self.nkld_rw_f = nkld_reward_fun
        # KL-divergence based reward function
        if kld_reward_fun is None:
            self.kld_rw_f = partial(tmp_fun, ret='0')
        else:
            assert callable(kld_reward_fun), (
                   "kld_reward_fun should be a function.")
            self.kld_rw_f = kld_reward_fun

        if phys_state_info is None:
            self.n_xp = 0
            self.init_xp = torch.tensor([])
            self.xp_f = partial(tmp_fun, ret='empty tensor')
        else:
            assert (isinstance(phys_state_info, (list, tuple))
                    and len(phys_state_info) == 3), (
               "phys_state_info should be a list or tuple of "
               "length 3, including n_phys_state, init_phys_state and "
               "phys_state_fun.")
            n_xp, init_xp, xp_f = phys_state_info
            assert isinstance(n_xp, int) and n_xp >= 0, (
                   "n_phys_state should be a non-negative interger.")
            self.n_xp = n_xp
            assert (isinstance(init_xp, (list, tuple))
                    and len(init_xp) == n_xp), (
                   "init_phys_state should be a list or tuple"
                   "of size n_phys_state.")
            self.init_xp = torch.tensor(init_xp)
            assert callable(xp_f), (
                   "phys_state_fun should be a function.")
            self.xp_f = xp_f
        self.n_phys_state = self.n_xp
        self.init_phys_state = self.init_xp
        self.phys_state_fun = self.xp_f

        self.post_approx = post_approx

    def get_rewards(self, stage=0, 
                    ds_hist=None, ys_hist=None, 
                    xps_hist=None, params=None,
                    include_kld_rewards=True):
        """
        A function to compute rewards of sequences.

        Parameters
        ----------
        stage : int, optional(default=0)
            The stage index. 0 <= stage <= n_stage.
        ds_hist : torch.Tensor of size (n_traj, n_stage, n_design), 
                  optional(default=None)
            The design variable histories.
        ys_hist : torch.Tensor of size (n_traj, n_stage, n_obs), 
                  optional(default=None)
            The observation histories. 
        xps_hist : torch.Tensor of size (n_traj, n_stage + 1, n_phys_state), 
              optional(default=None)
            The physical state histories.
        params : torch.Tensor of size (n_traj, n_param),
             optional(default=None)
            The true underlying parameters.
        include_kld_rewards : bool, optional(default=None)
            Whether to include information-gain rewards.

        Returns
        -------
        A torch.Tensor which are the rewards.
        """
        assert stage >= 0 and stage <= self.n_stage
        nkld_rewards = self.nkld_rw_f(stage, ds_hist, ys_hist, xps_hist, params)
        if include_kld_rewards:
            kld_rewards = self.kld_rw_f(stage, ds_hist, ys_hist, xps_hist, params)
        else:
            kld_rewards = 0
        return nkld_rewards + kld_rewards