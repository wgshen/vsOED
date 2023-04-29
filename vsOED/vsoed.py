import numpy as np
from functools import partial
from .utils import *
import emcee

class VSOED(object):
    """
    A class for variational sequential optimal experimental design (vsOED). 
    This class takes basic inputs (e.g., forward model, reward function, 
    physical state transition, and dimensions of the problem, etc.) to construct 
    a vsOED framework. However, this class does not include method functions to 
    solve the vsOED problem.
    This code accommodates continuous unknown parameters with user-defined 
    prior, continuous design with upper and lower bounds, and additive Gaussian 
    noise on observations.

    Parameters
    ----------
    model_fun : function
        Forward model function G_k(theta, d_k, x_{k,p}). It will be abbreviated 
        as m_f inside this class.
        The forward model function should take following inputs:
            * theta, numpy.ndarray of size (n_sample or 1, n_param)
                Parameter samples.
            * d, numpy.ndarray of size (n_sample or 1, n_design)
                Designs.
            * xp, numpy.ndarray of size (n_sample or 1, n_phys_state)
                Physical states.
        and the output is 
            * numpy.ndarray of size (n_sample, n_obs).
        When the first dimension of theta, d or xp is 1, it should be augmented
        to align with the first dimension of other inputs (i.e., we reuse it for
        all samples).
    n_stage : int
        Number of experiments to be designed and conducted.
    n_param : int
        Dimension of parameter space, should not be greater than 3 in this
        version because we are using grid discritization on parameter sapce.
    n_design : int
        Dimension of design space.
    n_obs : int
        Dimension of observation space.
    design_bounds : list, tuple or numpy.ndarray of size (n_design, 2)
        It includes the constraints of the design variable. In this version, we
        only allow to set hard limit constraints. In the future version, we may
        allow users to provide their own constraint function.
        The length of design_bounds should be n_design.
        k-th entry of design_bounds is a list, tuple or numpy.ndarray like 
        (lower_bound, upper_bound) for the limits of k-th design variable.
    noise_info : list, tuple or numpy.ndarray of size (n_obs, 3)
        It includes the statistics of additive Gaussian noise.
        The length of noise_info should be n_obs. k-th entry of noise_info is a 
        list, tuple or numpy.ndarray including
            * noise_loc : float or int
            * noise_base_scale : float or int
                It will be abbreviated as noise_b_s in this class.
            * noise_ratio_scale : float or int
                It will be abbreviated as noise_r_s in this class.
        The corresponding noise will follow a gaussian distribution with mean
        noise_loc, std (noise_base_scale + noise_ratio_scale * abs(G)).
    reward_fun : function, optional(default=None)
        User-provided non-KL-divergence based reward function 
        g_k(x_k, d_k, y_k). It will be abbreviated as nlkd_rw_f inside this 
        class.
        The reward function should take following inputs:
            * stage : int
                The stage index of the experiment.
            * xbs : None or numpy.ndarray of size 
                   (n_traj or 1, n_grid ** n_param, n_param + 1)
                Grid discritization of the belief state.
            * xps : np.ndarray of size (n_traj or 1, n_phys_state)
                The physical state.
            * ds : np.ndarray of size (n_traj or 1, stage + 1, n_design)
                The design variables from 0-th up to stage-th experiments.
            * ys : np.ndarray of size (n_traj or 1, stage + 1, n_obs)
                The observations from 0-th up to stage-th experiments.
        and the output is 
            * A np.ndarray of size n_traj or 1 which are the rewards.
        Note that the information gain is computed within this class, and does
        not needed to be included in reward_fun.
        When reward_fun is None, the stage reward would be 0, only KL divergence
        from the prior to the posterior will be considered.
    phys_state_info : list, tuple or numpy.ndarray of size (3), 
                      optional(default=None)
        When phy_state_info is None, then there is no physical states in this 
        sOED problem, otherwise it includes the following information of 
        physical states:
            * n_phys_state : int
                Dimension of physical state.
                It will be abbreviated as n_xp inside this class.
            * init_phys_state : list, or tuple
                Initial physical states.
                It will be abbreviated as init_xp inside this class.
                The length of init_phys_state should n_phys_state.
                In the future, we will let phys_state_fun provide the initial 
                physical state, such that it could be stochastic.
            * phys_state_fun : function
                Function to update physical state.
                x_{k+1,p} = phys_state_fun(x_{k,p}, d_k, y_k).
                It will be abbreviated as xp_f inside this class.
                The physical state transition function should take following 
                inputs:
                    * xps : np.ndarray of size (n_traj or 1, n_phys_state)
                        The old physical state before conducting stage-th 
                        experiement.
                    * stage : int
                        The stage index of the experiment.
                    * ds : np.ndarray of size 
                           (n_traj or 1, stage + 1, n_design)
                        The design variables from 0-th up to stage-th 
                        experiments.
                    * ys : np.ndarray of size (n_traj or 1, stage + 1, n_obs)
                        The observations from 0-th up to stage-th experiments.
                and the output is 
                    * numpy.ndarray of size (n_traj or 1, n_xp)
                Note that the update of belief state is realized in this class, 
                and does not need to be provided by users.
    post_approx : class, optional(default=None)
        User-provided posterior approximator for variational estimator.
        ==================================================================
    use_grid_kld : bool, optional(default=False)
        ==================================================================
    n_grid : int, optional(default=50)
        Number of grid points to discretize each dimension of parameter space
        to store the belief state if use_grid_kld is True. Using grid 
        discretization is only practical when the dimension is smaller than 5. 
    param_bounds : list, tuple or numpy.ndarray of size (n_param, 2), 
                   optional(default=None)
        ================================================================== 
    post_rvs_method : str, optional(default="MCMC")
        Method to sample from the posterior, including:
            * "MCMC", Markov chain Monte Carlo via emcee.
            * "Rejection", rejection sampling, only allowed for 1D parameter.
    random_state : int, optional(default=None)
        It is used as the random seed.        

    Methods
    -------
    check_grid(),
        Check if using grids to compute KL divergence is applicable. 
    prior_logpdf(), prior_pdf()
        Evaluate the prior logpdf (pdf) of parameter samples.
    prior_rvs()
        Generate samples from the prior.
    post_logpdf(), post_pdf()
        Evaluate the posterior logpdf (pdf) of parameter samples.
    post_rvs()
        Generate samples from the posterior.
    xb_f()
        Update the belief state with a single observation and design.
    get_xb()
        Update the belief state with a sequence of observations and designs.
    get_xps()
        Update the physical state with sequences of observations and designs.
    get_reward()
        Get the reward at a given stage with given state, design and 
        observation.
    get_total_reward()
        Get the total reward give a sequence of designs and observations.

    Future work
    -----------
    Let users provide their own prior sample generator and prior PDF evaluator.
    Let users provide more complex constraints on design variables.
    Let users provide their own measurement noise function.
    Consider random initial physical state.
    Use underscore to make variables not directly accessible by users, like
    "self._n_stage", and use @property to make it indirectly accessible.
    """
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
        
        # assert callable(model_fun), (
        #        "model_fun should be a function.")
        self.m_f = model.model
        self.model_fun = self.m_f
        self.loglikeli = model.loglikeli

        self.prior_logpdf = prior.logpdf
        def prior_pdf(x):
            return torch.exp(prior.logpdf(x))
        self.prior_pdf = prior_pdf
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
            def nkld_rw_f(*args, **kws):
                return 0
            self.nkld_rw_f = nkld_rw_f
        else:
            assert callable(stage_reward_fun), (
                   "nkld_reward_fun should be a function.")
            self.nkld_rw_f = nkld_reward_fun
        # KL-divergence based reward function
        if kld_reward_fun is None:
            def kld_rw_f(*args, **kws):
                return 0
            self.kld_rw_f = kld_rw_f
        else:
            assert callable(kld_reward_fun), (
                   "kld_reward_fun should be a function.")
            self.kld_rw_f = kld_reward_fun

        if phys_state_info is None:
            self.n_xp = 0
            self.init_xp = ()
            def xp_f(*args, **kws):
                return torch.tensor([])
            self.xp_f = xp_f
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
                    xps_hist=None, thetas=None,
                    include_kld_rewards=True):
        """
        A function to compute rewards of sequences at given "stage", with belief 
        states "xbs", physical states "xps", designs 'ds' and observations 'ys'.

        Parameters
        ----------
        stage : int, optional(default=0)
            The stage index. 0 <= stage <= n_stage.
        xbs : numpy.ndarray of size (n_traj or 1, n_grid ** n_param, 
              n_param + 1), optional(default=None)
            The belief states at stage "stage". 
        xps : numpy.ndarray of size (n_traj or 1, n_phys_state), 
              optional(default=None)
            The physical states at stage "stage".
        ds_hist : numpy.ndarray of size (n_traj or 1, n_stage, n_design), 
                  optional(default=None)
            The design variable histories.
        ys_hist : numpy.ndarray of size (n_traj or 1, n_stage, n_obs), 
                  optional(default=None)
            The observation histories. 
        thetas : numpy.ndarray of size (n_traj or 1, n_param),
             optional(default=None)
            The true underlying parameters.
        use_grid_kld : bool, optional(default=None)
            Whether use grids to compute KL divergence.

        Returns
        -------
        A numpy.ndarray of size n_traj or 1 which are the rewards.
        """
        assert stage >= 0 and stage <= self.n_stage
        nkld_rewards = self.nkld_rw_f(stage, ds_hist, ys_hist, xps_hist, thetas)
        if include_kld_rewards:
            kld_rewards = self.kld_rw_f(stage, ds_hist, ys_hist, xps_hist, thetas)
        else:
            kld_rewards = 0
        return nkld_rewards + kld_rewards
