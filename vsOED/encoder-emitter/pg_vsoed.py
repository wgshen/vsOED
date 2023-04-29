import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .vsoed import VSOED
from .utils import *
import time
from functools import partial
from tqdm import trange

class PGvsOED(VSOED):
    """
    A class for solving sequential optimal experimental design (sOED) problems 
    using policy gradient (PG) method. This class is based on the SOED class, 
    and realize the method functions of PG upon it.
    Please refer to .

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
    prior_info : list, tuple or numpy.ndarray of size (n_param, 3)
        It includes the information of the prior. In this version, we only 
        allow to use independent normal or uniform distributions on each 
        dimension of parameter space. In the future version, we will let users 
        provide their owe functions to generate samples from the prior and 
        evaluate prior PDFs. 
        The length of prior_info should be n_param. k-th entry of prior_info 
        includes the following three components for k-th dimension of the 
        paramemter (could be list, tuple or numpy.ndarray in the following 
        ordering):
            * prior_type : str
                Type of prior of k-th parameter.
                "uniform" indicates uniform distribution.
                "normal" or "gaussian" indicates normal distribution.
            * prior_loc : float or int
                Mean for normal, or left bound for uniform.
            * prior_scale : float or int
                Std for normal, or range for uniform.
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
            * xb : numpy.ndarray of size (n_grid ** n_param, n_param + 1)
                Grid discritization of the belief state.
            * xp : np.ndarray of size (n_phys_state)
                The physical state.
            * d : np.ndarray of size (n_design)
                The design variable.
            * y : np.ndarray of size (n_obs)
                The observation.
        and the output is 
            * A float which is the reward.
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
                    * xp : np.ndarray of size (n_sample or 1, n_phys_state)
                        The old physical state before conducting stage-th 
                        experiement.
                    * stage : int
                        The stage index of the experiment.
                    * d : np.ndarray of size (n_sample or 1, n_design)
                        The design variables at stage-th experiment.
                    * y : np.ndarray of size (n_sample or 1, n_obs)
                        The observations at stage-th expriments.
                and the output is 
                    * numpy.ndarray of size (n_sample, n_xp)
                Note that the update of belief state is realized in this class, 
                and does not need to be provided by users.
    n_grid : int, optional(default=50)
        Number of grid points to discretize each dimension of parameter space
        to store the belief state. Using grid discretization is only practical 
        when the dimension is not bigger than 3. 
        In the future version, we plan to use other techniques (MCMC, trasport
        map) to represent the posterior distribution, and n_grid will pribably
        be discarded.
    post_rvs_method : str, optional(default="MCMC")
        Method to sample from the posterior, including:
            * "MCMC", Markov chain Monte Carlo via emcee.
            * "Rejection", rejection sampling, only allowed for 1D parameter.
    random_state : int, optional(default=None)
        It is used as the random seed.    
    actor_dimns : list, tuple or numpy.ndarray, optional(default=None)
        The dimensions of hidden layers of actor (policy) network.
    critic_dimns : list, tuple or numpy.ndarray, optional(default=None)
        The dimensions of hidden layers of critic (action value function) net.
    double_precision : bool, optional(default=False)
        Whether use double precision or single precison for the pytorch network,
        single precision is sufficiently accurate.

    Methods
    -------
    initialize()
        Initialize sOED.
    initialize_actor(), initialize_policy()
        Initialize the actor network. These two functions are equivalent.
    initialize_critic()
        Initialize the critic network.
    load_actor(), load_policy()
        Load an actor network. These two functions are equivalent.
    load_critic()
        Load an critic network. 
    get_actor(), get_policy()
        Return the actor network. These two functions are equivalent.
    get_critic()
        Return the critic network.
    form_actor_input()
        Form the inputs of actor network.
    form_critic_input()
        Form the inputs of critic network.
    get_designs()
        Get designs given sequences of historical designs and observations 
        by running the actor network.
    get_design()
        Get a single design given a sequence of historical designs and 
        observations by running the actor network.
    get_action_value()
        Get the action value (Q-value) given historical designs, observations,
        and current design.
    train()
        Run policy gradient for give number of udpates.
    asses()
        Asses the performance of current policy.

    Future work
    -----------
    Normalize the output of neural network.
    Use MCMC or transport map to replace the grid discretization.
    Let users provide their own prior sample generator and prior PDF evaluator.
    Let users provide more complex constraints on design variables.
    Let users provide their own measurement noise function.
    Consider random initial physical state.
    Use underscore to make variables not directly accessible by users, like
    "self._n_stage", and use @property to make it indirectly accessible.
    Allow users to use gpu to accelerate the NN computation. It is not important
    in this version, because the NN is small and using cpu is fast enough.
    """
    def __init__(self, model,
                 n_stage, n_param, n_design, n_obs, 
                 prior, design_bounds, noise_info, 
                 reward_fun=None, phys_state_info=None, 
                 post_approx=None, use_grid_kld=False, 
                 n_grid=50, param_bounds=None, 
                 use_PCE=False, n_contrastive_sample=10000, 
                 post_rvs_method="MCMC", random_state=None,
                 encoder_dimns=None, actor_dimns=None, critic_dimns=None,
                 activate=None,
                 double_precision=False, device=torch.device('cpu')):
        super().__init__(model, n_stage, n_param, n_design, n_obs, 
                         prior, design_bounds, noise_info, 
                         reward_fun, phys_state_info, 
                         post_approx, use_grid_kld, 
                         n_grid, param_bounds, 
                         post_rvs_method, random_state)
        if random_state is None:
            random_state = np.random.randint(1e6)
        torch.manual_seed(random_state)

        self.use_PCE = use_PCE
        self.n_contrastive_sample = n_contrastive_sample

        if activate is None:
            activate = nn.ReLU
        self.activate = activate

        assert isinstance(double_precision, bool), (
            "double_precision should either be True or False.")
        if double_precision:
            self.dtype = torch.float64
        else:
            self.dtype = torch.float32
        self.device = device

        # Initialize the actor (policy) network and critic network.
        self.encoder_input_dimn = self.n_design + self.n_obs + self.n_xp
        if encoder_dimns is None:
            encoder_dimns = [self.encoder_input_dimn * 4, self.encoder_input_dimn * 4]
        encoder_output_dimn = encoder_dimns[-1]
        self.encoder_dimns = encoder_dimns
        self.actor_input_dimn = encoder_output_dimn + self.n_xp
        self.critic_input_dimn = encoder_output_dimn + self.n_xp + self.n_design
        self.actor_dimns = actor_dimns
        self.critic_dimns = critic_dimns
        self.initialize()

        self.initialize_policy = self.initialize_actor
        self.load_policy = self.load_actor
        self.get_policy = self.get_actor

        self.actor = partial(self.get_designs, return_all_stages=False, use_target=False)
        self.actors = partial(self.get_designs, return_all_stages=True, use_target=False)
        self.actor_target = partial(self.get_designs, return_all_stages=False, use_target=True)
        self.actors_target = partial(self.get_designs, return_all_stages=True, use_target=True)
        self.critic = partial(self.get_action_values, return_all_stages=False, use_target=False)
        self.critics = partial(self.get_action_values, return_all_stages=True, use_target=False)
        self.critic_target = partial(self.get_action_values, return_all_stages=False, use_target=True)
        self.critics_target = partial(self.get_action_values, return_all_stages=True, use_target=True)

    def initialize(self):
        self.initialize_encoder(self.encoder_dimns)
        self.initialize_actor(self.actor_dimns)
        self.initialize_critic(self.critic_dimns)
        self.design_noise_scale = None
        try:
            self.post_approx.reset()
        except:
            pass

    def initialize_encoder(self, encoder_dimns):
        self.encoder_dimns = np.copy(encoder_dimns)
        encoder_dimns = np.append(self.encoder_input_dimn, encoder_dimns)
        self.encoder_actor_net = Encoder(encoder_dimns, self.activate, self.device, self.dtype).to(self.device, self.dtype)
        self.encoder_critic_net = Encoder(encoder_dimns, self.activate, self.device, self.dtype).to(self.device, self.dtype)
        self.update = 0

    def initialize_actor(self, actor_dimns=None):
        """
        Initialize the actor (policy) network.

        Parameters
        ----------
        actor_dimns : list, tuple or numpy.ndarray, optional(default=None)
            The dimensions of hidden layers of actor (policy) network.
        """
        NoneType = type(None)
        assert isinstance(actor_dimns, (list, tuple, np.ndarray, NoneType)), (
               "actor_dimns should be a list, tuple or numpy.ndarray of "
               "integers.")
        output_dimn = self.n_design
        if actor_dimns is None:
            actor_dimns = (self.actor_input_dimn * 10,  self.actor_input_dimn * 10)
        self.actor_dimns = np.copy(actor_dimns)
        actor_dimns = np.append(np.append(self.actor_input_dimn,  actor_dimns), output_dimn)
        
        self.actor_net = Net(actor_dimns, self.activate, self.design_bounds, 
            'actor', self.device, self.dtype).to(self.device, self.dtype)
        self.actor_target_net = Net(actor_dimns, self.activate, self.design_bounds, 
            'actor', self.device, self.dtype).to(self.device, self.dtype)
        self.actor_target_net.load_state_dict(self.actor_net.state_dict())
        self.update = 0
        self.actor_optimizer = None
        self.actor_lr_scheduler = None

    def initialize_critic(self, critic_dimns=None):
        """
        Initialize the critic (actor-value function).

        Parameters
        ----------
        critic_dimns : list, tuple or numpy.ndarray, optional(default=None)
            The dimensions of hidden layers of critic (actor value function) 
            network.
        """
        NoneType = type(None)
        assert isinstance(critic_dimns, (list, tuple, np.ndarray, NoneType)), (
               "critic_dimns should be a list, tuple or numpy.ndarray of ",
               "integers.")
        output_dimn = 1
        if critic_dimns is None:
            critic_dimns = (self.critic_input_dimn * 10,  self.critic_input_dimn * 10)
        self.critic_dimns = np.copy(critic_dimns)
        critic_dimns = np.append(np.append(self.critic_input_dimn, critic_dimns), 
                                 output_dimn)
        self.critic_net = Net(critic_dimns, self.activate, 
            np.array([[-np.inf, np.inf]]), 'critic', self.device, self.dtype).to(self.device, self.dtype)
        self.critic_target_net = Net(critic_dimns, self.activate, 
            np.array([[-np.inf, np.inf]]), 'critic', self.device, self.dtype).to(self.device, self.dtype)
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        self.update = 0
        self.critic_optimizer = None  
        self.critic_lr_scheduler = None   

    def load_actor(self, net, optimizer=None):
        """
        Load the actor network with single precision.

        Parameters
        ----------
        net : nn.Module
            A pre-trained PyTorch network with input dimension actor_input_dimn 
            and output dimension n_design.
        optimizer : algorithm of torch.optim
            An optimizer corresponding to net.
        """
        try:
            net = net.to(self.device, self.dtype)
            output = net(torch.zeros(1, self.actor_input_dimn).to(self.device, self.dtype))
            assert output.shape[1] == self.n_design, (
                   "Output dimension should be {}.".format(self.n_design))
            self.actor_net = net
        except:
            print("Actor network should has "
                  "input dimension {}.".format(self.actor_input_dimn))
        self.actor_optimizer = optimizer
        self.update = 0

    def load_critic(self, net, optimizer=None):
        """
        Load the critic network with single precision.

        Parameters
        ----------
        net : nn.Module
            A pre-trained PyTorch network with input dimension critic_input_dimn
            and output dimension 1.
        optimizer : algorithm of torch.optim
            An optimizer corresponding to net.
        """
        try:
            net = net.to(self.device, self.dtype)
            output = net(torch.zeros(1, self.critic_input_dimn).to(self.device, self.dtype))
            assert output.shape[1] == 1, (
                   "Output dimension should be 1.")
            self.critic_net = net
        except:
            print("Critic network should has input dimension {}.".format(self.critic_input_dimn))
        self.critic_optimizer = optimizer
        self.update = 0

    def get_actor(self):
        return self.actor_net

    def get_critic(self):
        return self.critic_net

    def get_designs(self, ds_hist, ys_hist, xps_hist, return_all_stages=False, use_target=False):
        """
        A function to get designs by running the policy network.

        Parameters
        ----------
        stage : int, optional(default=0)
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design),
                  optional(default=None)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs),
                  optional(default=None)
            n_traj sequences of observations before stage "stage". 

        Returns
        -------
        A numpy.ndarry of size (n_traj, n_design) which are designs.
        """
        assert ds_hist.shape[1] == ys_hist.shape[1]
        assert xps_hist.shape[1] == ds_hist.shape[1] + 1
        xbs_encoded = self.encoder_actor_net(ds_hist, ys_hist, xps_hist[:, :-1], return_all_stages)
        if return_all_stages:
            xps = torch.from_numpy(xps_hist).to(self.device, self.dtype)
        else:
            xps = torch.from_numpy(xps_hist[:, -1]).to(self.device, self.dtype)
        states = torch.cat([xbs_encoded, xps], dim=-1)
        if use_target:
            designs = self.actor_target_net(states)
        else:
            designs = self.actor_net(states)
        return designs # (n_traj, len(xps_hist), n_design) if returan_all_stages else (n_traj, n_design)

    def get_action_values(self, ds_hist, ys_hist, xps_hist, ds, return_all_stages=False, use_target=False, state_no_grad=False):
        """
        A function to get the Q-value by running the critic network.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs)
            n_traj sequences of observations before stage "stage". 
        ds : numpy.ndarray of size(n_traj or 1, n_design)
            Designs on which we want to get the Q value.

        Returns
        -------
        A numpy.ndarry of size (n_traj) which are Q values.
        """
        assert ds_hist.shape[1] == ys_hist.shape[1]
        assert xps_hist.shape[1] == ds_hist.shape[1] + 1
        if return_all_stages:
            assert ds.shape[1] == xps_hist.shape[1]
        else:
            assert len(ds.shape) == 2
        xbs_encoded = self.encoder_critic_net(ds_hist, ys_hist, xps_hist[:, :-1], return_all_stages)
        if return_all_stages:
            xps = torch.from_numpy(xps_hist).to(self.device, self.dtype)
        else:
            xps = torch.from_numpy(xps_hist[:, -1]).to(self.device, self.dtype)
        if isinstance(ds, np.ndarray):
            ds = torch.from_numpy(ds)
        ds = ds.to(self.device, self.dtype)
        states = torch.cat([xbs_encoded, xps], dim=-1)
        if state_no_grad:
            states = states.detach()
        if use_target:
            values = self.critic_target_net(states, ds)
        else:
            values = self.critic_net(states, ds)
        return values[..., 0] # (n_traj, len(xps_hist)) if returan_all_stages else (n_traj)


    def train(self, n_update=100, n_buffer=20000, n_traj=1000, n_batch=1000,
              discount=1,
              encoder_actor_optimizer=None,
              encoder_actor_lr_scheduler=None,
              encoder_critic_optimizer=None,
              encoder_critic_lr_scheduler=None,
              actor_optimizer=None,
              actor_lr_scheduler=None,
              n_critic_update=30,
              critic_optimizer=None,
              critic_lr_scheduler=None,
              n_post_approx_update=50,
              lr_target=0.05,
              design_noise_scale=None, 
              design_noise_decay=0.99,
              on_policy=False,
              use_PCE=None,
              use_PCE_incre=None,
              use_grid_kld=None,
              transition=5000,
              frozen=-1,
              grad_clip=np.inf,
              verbose=100):
        """
        A function to run policy gradient for given number of updates to find
        the optimal policy.

        Parameters
        ----------
        n_update : int, optional(default=3)
            Number of updates to find the optimal policy.  
        n_traj : int, optional(default=1000)
            Number of trajectories to sample during the training. 
        actor_optimizer : an algorithm of torch.optim, optional(default=None)
            The optimizer for the actor network. Example:
            torch.optim.SGD(<OBJECT INSTANCE NAME>.actor_net.parameters(),
                            lr=0.01)
        actor_lr_scheduler : a learning rate scheduler of 
                             torch.optim.lr_scheduler, optional(default=None)
            The learning rate scheduler for the actor optimizer. Example:
            torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=0.99)
        n_critic_update : int, optional(default=30)
            The number of updates to train the critic network within each policy
            update.
        critic_optimizer : an algorithm of torch.optim, optional(default=None)
            The optimizer for the critic network. Example:
            torch.optim.SGD(<OBJECT INSTANCE NAME>.critic_net.parameters(),
                            lr=0.01)
        critic_lr_scheduler : a learning rate scheduler of 
                              torch.optm.lr_scheduler, optional(default=None)
            The learning rate scheduler for the critic optimizer. Example:
            torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.99)
        design_noise_scale : int, list, tuple or numpy.ndarray of size 
                             (n_design), optional(default=None)
            The scale of additive exploration Gaussian noise on each dimension 
            of design variable.
        design_noise_decay : int or float, optional(default=0.99)
            The decay weight of design_noise_scale. The decay following
            design_noise_scale = design_noise_scale * design_noise_decay is 
            done after each update.
        on_policy : bool, optional(default=True)
            Whether use on-policy scheme or off-policy scheme.
            On-policy means the action value is estimated at the design that
            follows the current policy, for example,
            Q(x_k^i, d_k^i) <- g(x_k^i, d_k^i, y_k^i) +
                               Q(x_{k+1}^i, d_{k+1}^i). 
            d_{k+1}^i is the design that generated by following the policy (
            which is a noisy policy with additive noise on designs).
            Off-policy means the action value is estimated at the design that
            is not following the current policy, for example,
            Q(x_k^i, d_k^i) <- g(x_k^i, d_k^i, y_k^i) +
                               Q(x_{k+1}^i, mu(x_{k+1}^i)).
            Here the next design is mu(x_{k+1}^i), which is the clean output 
            from the actor network. Note that although mu(x_{k+1}^i) follows
            the actor network, it's not following the current true policy due
            to the lack of exploration noise. Moreover, if there is no noise
            on designs, then on-policy is equivalent to off-policy in this code.
        use_grid_kld : bool, optional(default=None)
            Whether use grid discretization to evaulate the KL divergence.
            Unlike use_grid_kld in the asses function which only changes
            self.use_grid_kld temporarily, self.use_grid_kld will be changed to 
            use_grid_kld permanently if it is specified in this function.
        """
        assert n_buffer >= n_traj
        assert n_buffer >= n_batch
        self.n_buffer = n_buffer
        self.n_traj = n_traj
        self.n_batch = n_batch
        # if encoder_actor_optimizer is None:
        #     if self.encoder_actor_optimizer is None:
        #         self.encoder_optimizer = optim.SGD(self.encoder_net.parameters(), lr=0.1)
        # else:
        #     self.encoder_optimizer = encoder_optimizer
        # if encoder_lr_scheduler is None:
        #     if self.encoder_lr_scheduler is None:
        #         self.encoder_lr_scheduler = optim.lr_scheduler.ExponentialLR(
        #             self.encoder_optimizer, gamma=0.99)
        # else:
        #     self.encoder_lr_scheduler = encoder_lr_scheduler
        self.encoder_actor_optimizer = encoder_actor_optimizer
        self.encoder_actor_lr_scheduler = encoder_actor_lr_scheduler
        self.encoder_critic_optimizer = encoder_critic_optimizer
        self.encoder_critic_lr_scheduler = encoder_critic_lr_scheduler
        if actor_optimizer is None:
            if self.actor_optimizer is None:
                self.actor_optimizer = optim.SGD(self.actor_net.parameters(), lr=0.1)
        else:
            self.actor_optimizer = actor_optimizer
        if actor_lr_scheduler is None:
            if self.actor_lr_scheduler is None:
                self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.actor_optimizer, gamma=0.99)
        else:
            self.actor_lr_scheduler = actor_lr_scheduler
        if critic_optimizer is None:
            if self.critic_optimizer is None:
                self.critic_optimizer = optim.SGD(self.critic_net.parameters(), lr=0.01)
        else:
            self.critic_optimizer = critic_optimizer
        if critic_lr_scheduler is None:
            if self.critic_lr_scheduler is None:
                self.critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.critic_optimizer, gamma=1)
        else:
            self.critic_lr_scheduler = critic_lr_scheduler
            
        self.lr_target = lr_target

        if self.update == 0:
            if design_noise_scale is None:
                if self.design_noise_scale is None:
                    self.design_noise_scale = (self.design_bounds[:, 1] -  self.design_bounds[:, 0]) / 20                
                    self.design_noise_scale[self.design_noise_scale == np.inf] = 5
                    
            elif isinstance(design_noise_scale, (list, tuple)):
                self.design_noise_scale = np.array(design_noise_scale)
            else:
                self.design_noise_scale = design_noise_scale
            self.design_noise_scale_init = np.copy(self.design_noise_scale)
        else:
            if design_noise_scale is not None and design_noise_scale != self.design_noise_scale_init:
                self.design_noise_scale = design_noise_scale
                self.design_noise_scale_init = np.copy(self.design_noise_scale)
            
            
        assert design_noise_decay > 0 and design_noise_decay <= 1

        if isinstance(use_PCE, bool):
            self.use_PCE = use_PCE
            if isinstance(use_PCE_incre, bool):
                self.use_PCE_incre = use_PCE_incre
        if isinstance(use_grid_kld, bool):
            self.use_grid_kld = use_grid_kld

        # Initialize replay buffer
        if self.update == 0:
            self.asses(n_batch * 2, self.design_noise_scale)
            self.buffer_thetas = self.thetas
            self.buffer_dcs_hist = self.dcs_hist
            self.buffer_ds_hist = self.ds_hist
            self.buffer_ys_hist = self.ys_hist
            self.buffer_xbs = self.xbs
            self.buffer_xps_hist = self.xps_hist
            self.buffer_rewards_hist = self.rewards_hist
            self.update_hist = []


        for l in range(n_update):
            # print('Update Level', self.update)
            t1 = time.time()
            self.asses(n_traj, self.design_noise_scale)  
            if self.update % verbose == 0:
                print('Update Level', self.update)
                print("Averaged total reward:  {:.4}".format(self.averaged_reward))

               # print(dcs_hist.shape, dcs_hist.mean(axis=(0, -1)).shape)

                print("Averaged designs:", self.dcs_hist.mean(axis=(0, 1)))
                # print("Averaged designs:", dcs_hist.mean(axis=(0)))
            self.update_hist.append(self.averaged_reward)                     
            ############################################
            ############################################
            # Update the replay buffer
            l_buffer = len(self.buffer_thetas)
            # idxs_left = np.random.choice(l_buffer, min(l_buffer, n_buffer - n_traj), replace=False)
            idxs_left = -(np.arange(min(l_buffer, n_buffer - n_traj)) + 1)[::-1]
            self.buffer_thetas = np.concatenate([self.buffer_thetas[idxs_left], self.thetas], 0)
            self.buffer_dcs_hist = np.concatenate([self.buffer_dcs_hist[idxs_left], self.dcs_hist], 0)
            self.buffer_ds_hist = np.concatenate([self.buffer_ds_hist[idxs_left], self.ds_hist], 0)
            self.buffer_ys_hist = np.concatenate([self.buffer_ys_hist[idxs_left], self.ys_hist], 0)
            if self.xbs is not None:
                self.buffer_xbs = np.concatenate([self.buffer_xbs[idxs_left], self.xbs], 0)
            self.buffer_xps_hist = np.concatenate([self.buffer_xps_hist[idxs_left], self.xps_hist], 0)
            self.buffer_rewards_hist = np.concatenate([self.buffer_rewards_hist[idxs_left], self.rewards_hist], 0)
            # print(f'buffer size: {len(self.buffer_thetas)}')
            l_buffer = len(self.buffer_thetas)
            p = np.array([i + 1 for i in range(l_buffer)], dtype=np.float)
            p /= p.sum()
            idxs_pick = np.random.choice(l_buffer, n_batch, p=p, replace=False)
            self.thetas = self.buffer_thetas[idxs_pick]
            self.dcs_hist = self.buffer_dcs_hist[idxs_pick]
            self.ds_hist = self.buffer_ds_hist[idxs_pick]
            self.ys_hist = self.buffer_ys_hist[idxs_pick]
            if self.xbs is not None:
                self.xbs = self.buffer_xbs[idxs_pick]
            self.xps_hist = self.buffer_xps_hist[idxs_pick]
            self.rewards_hist = self.buffer_rewards_hist[idxs_pick]
            ############################################
            # Train the post approxer
            t2 = time.time()
            if self.post_approx is not None and not self.use_grid_kld and not self.use_PCE:
                # self.post_approx.train(self.buffer_ds_hist, self.buffer_ys_hist, self.buffer_thetas, None, n_post_approx_update)
                # self.buffer_rewards_hist[:, -1] = self.get_rewards(self.n_stage,
                #                                             None,
                #                                             self.buffer_xps_hist[:, -1],
                #                                             self.buffer_ds_hist,
                #                                             self.buffer_ys_hist,
                #                                             self.buffer_thetas)
                self.post_approx.train(self.ds_hist, self.ys_hist, self.thetas, self.update, n_post_approx_update)
                self.rewards_hist[:, -1] = self.get_rewards(self.n_stage,
                                                            None,
                                                            self.xps_hist[:, -1],
                                                            self.ds_hist,
                                                            self.ys_hist,
                                                            self.thetas)
                pass
            ############################################
            if self.update <= frozen:
                self.update += 1
                continue
            # Form the inputs and target values of critic network, and form the 
            # inputs of the actor network.
            with torch.no_grad():
                g_critic = torch.zeros(n_batch, self.n_stage).to(self.device, self.dtype)
                g_critic[:, self.n_stage - 1] = torch.from_numpy(
                    self.rewards_hist[:, self.n_stage - 1:].sum(-1)).to(self.device, self.dtype)
                # if l < 0:
                factor = min(self.update / transition, 1)
                for k in range(self.n_stage - 1):
                    rewards = self.rewards_hist[:, k:]
                    discounts = np.array([discount ** kk for kk in range(rewards.shape[1])])
                    g_critic[:, k] = torch.from_numpy(
                    (rewards * discounts).sum(-1)).to(self.device, self.dtype) * (1 - factor)
                # else:
                if on_policy:
                    ds_next = self.ds_hist[:, 1:]
                    ds_next = np.concatenate([np.zeros((len(ds_next), 1, self.n_design)), ds_next], axis=1)
                    values_next = self.critics(self.ds_hist[:, :-1], self.ys_hist[:, :-1], 
                        self.xps_hist[:, :-1], ds_next)[:, 1:]
                else:
                    ds_next = self.actors_target(self.ds_hist[:, :-2], self.ys_hist[:, :-2], 
                        self.xps_hist[:, :-2])
                    ds_next = torch.cat([torch.zeros(len(ds_next), 1, self.n_design).to(self.device, self.dtype),
                        ds_next], dim=1)
                    values_next = self.critics_target(self.ds_hist[:, :-1], self.ys_hist[:, :-1], 
                        self.xps_hist[:, :-1], ds_next)[:, 1:]
                for k in range(self.n_stage - 1):
                    g_critic[:, k] += (discount * values_next[:, k] 
                    + torch.from_numpy(self.rewards_hist[:, k]).to(self.device, self.dtype)) * factor
            # Train critic.
            t3 = time.time()
            for _ in range(n_critic_update):
                y_critic = self.critics(self.ds_hist[:, :-1], self.ys_hist[:, :-1], self.xps_hist[:, :-1], self.ds_hist)
                loss = torch.mean((g_critic - y_critic) ** 2)
                self.encoder_critic_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                self.encoder_critic_optimizer.step()
                self.critic_optimizer.step()
            self.encoder_critic_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            t4 = time.time()
            # One step update on the actor network.
            # Add negative sign here because we want to do maximization.
            designs = self.actors(self.ds_hist[:, :-1], self.ys_hist[:, :-1], self.xps_hist[:, :-1])
            output = -self.critics(self.ds_hist[:, :-1], self.ys_hist[:, :-1], self.xps_hist[:, :-1], 
                designs, state_no_grad=True).mean()
            self.encoder_actor_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            output.backward()
            if grad_clip != np.inf:
                for param in self.actor_net.parameters():
                    param.grad = torch.clamp(param.grad, -grad_clip, grad_clip)
            self.encoder_actor_optimizer.step()
            self.actor_optimizer.step()
            self.encoder_actor_lr_scheduler.step()
            self.actor_lr_scheduler.step()


            t5 = time.time()

            for param, target_param in zip(self.actor_net.parameters(), self.actor_target_net.parameters()):
                target_param.data.copy_(self.lr_target * param.data + (1 - self.lr_target) * target_param.data)
            for param, target_param in zip(self.critic_net.parameters(), self.critic_target_net.parameters()):
                target_param.data.copy_(self.lr_target * param.data + (1 - self.lr_target) * target_param.data)

            self.update += 1
            self.design_noise_scale *= design_noise_decay
            t6 = time.time()

            # print(f'Assessment time: {t2 - t1:.3}')
            # print(f'Form input of actor and critic net: {t3 - t2:.3}')
            # print(f'Train critic: {t4 - t3:.3}')
            # print(f'Train actor: {t5 - t4:.3}')
            # print(f'Train post approxer: {t6 - t5:.3}')
            # print(f'Total one step time: {t6 - t1:.3}')
        # the_all = self.asses(n_traj, self.design_noise_scale, return_all = True)
        # return the_all
            
    def asses(self, n_traj=10000, design_noise_scale=None,
              use_grid_kld=None, use_PCE=None, use_PCE_incre=None,
              n_contrastive_sample=None,
              store_belief_state=False, return_all=False, theta_samples=None):
        """
        A function to asses the performance of current policy.

        Parameters
        ----------
        n_traj : int, optional(default=10000)
            Number of trajectories to sample during the assesment. 
        design_noise_scale : int, list, tuple or numpy.ndarray of size 
                             (n_design), optional(default=None)
            The scale of additive exploration Gaussian noise on each dimension 
            of design variable. When it is None, design_noise_scale will be
            set to 0.
        return_all : bool, optional(default=False)
            Return all information or not.
            If False, only return the averaged totoal reward.
            If True, return a tuple of all information generated during the 
            assesment, including 
            * averaged_reward (averaged total reward), float
            * thetas (parameters), numpy.ndarray of size (n_traj, n_param).
            * dcs_hist (clean designs), numpy.ndarray of size (n_traj,
                                                               n_stage,
                                                               n_design)
            * ds_hist (noisy designs), numpy.ndarray of size (n_traj, 
                                                              n_stage, 
                                                              n_design).
            * ys_hist (observations), numpy.ndarray of size (n_traj,
                                                             n_stage,
                                                             n_obs) .
            * xbs (terminal belief states), could either be None or
                numpy.ndarray of size (n_traj, 
                                       n_grid ** n_param, 
                                       n_param + 1),
                controlled by store_belief_state.
            * xps_hist (physical states), numpy.ndarray of size (n_traj,
                                                                 n_stage + 1,
                                                                 n_phys_state).
            * rewards_hist (rewards), numpy.ndarray of size (n_traj,
                                                             n_stage + 1).
        use_grid_kld : bool, optional(default=None)
            Whether use grid discretization to evaulate the KL divergence.
        store_belief_state : bool, optional(default=False)
            Whether store the belief states.

        Returns
        -------
        A float which is the averaged total reward.
        (optionally) other assesment results.
        """
        # Generate prior samples.
        if design_noise_scale is None:
            design_noise_scale = np.zeros(self.n_design)
        elif isinstance(design_noise_scale, (int, float)):
            design_noise_scale = np.ones(self.n_design) * design_noise_scale
        elif isinstance(design_noise_scale, (list, tuple, np.ndarray)):
            assert (isinstance(design_noise_scale, (list, tuple, np.ndarray))
                    and len(design_noise_scale) == self.n_design)
        if isinstance(use_grid_kld, bool):
            use_grid_kld_bckup = self.use_grid_kld
            self.use_grid_kld = use_grid_kld
        if isinstance(use_PCE, bool):
            use_PCE_bckup = self.use_PCE
            self.use_PCE = use_PCE
            if isinstance(use_PCE_incre, bool):
                self.use_PCE_incre = use_PCE_incre
        if self.use_PCE:
            if n_contrastive_sample is None:
                if self.n_contrastive_sample is not None:
                    n_contrastive_sample = self.n_contrastive_sample
                else:
                    n_contrastive_sample = 10000
        assert not (self.use_grid_kld and self.use_PCE), (
               "use_grid_kld and use_PCE cannot both be True")
        if store_belief_state:
            assert self.use_grid_kld, (
                   "use_grid_kld must be True if store_belief_state is True")
        if self.use_grid_kld:
            self.check_grid()
            self.init_xb = self.get_xb(None)
        
        if theta_samples is None:
            thetas = self.prior_rvs(n_traj)
        else:
            thetas = theta_samples

        if self.use_PCE:
            contrastive_thetas = self.prior_rvs(n_contrastive_sample)
            
        dcs_hist = np.zeros((n_traj, self.n_stage, self.n_design))
        ds_hist = np.zeros((n_traj, self.n_stage, self.n_design))
        Gs_hist = np.zeros((n_traj, self.n_stage, self.n_obs))
        ys_hist = np.zeros((n_traj, self.n_stage, self.n_obs))
        
        if store_belief_state:
            # We only store the terminal belief state.
            xbs = np.zeros((n_traj, *self.init_xb.shape))
        else:
            xbs = None
            
        # Store n_stage + 1 physical states.
        xps_hist = np.zeros((n_traj, self.n_stage + 1, self.n_xp))
        xps_hist[:, 0] = self.init_xp
        rewards_hist = np.zeros((n_traj, self.n_stage + 1))
        progress_points = np.rint(np.linspace(0, n_traj - 1, 30))

        for k in range(self.n_stage + 1):
            if k < self.n_stage:
                # Get clean designs.
                with torch.no_grad():
                    dcs = self.actor(ds_hist[:, :k], ys_hist[:, :k], xps_hist[:, :k + 1]).cpu().numpy()
                
                dcs_hist[:, k, :] = dcs
                # Add design noise for exploration. 

                ds = np.random.normal(loc=dcs, scale=design_noise_scale)
                ds = np.maximum(ds, self.design_bounds[:, 0])
                ds = np.minimum(ds, self.design_bounds[:, 1])
                ds_hist[:, k, :] = ds
                
                # Run the forward model to get observations.
                Gs = self.m_f(k, thetas, ds, xps_hist[:, k, :])
                Gs_hist[:, k, :] = Gs
                
                #print([thetas.shape, Gs.shape])
                
                ys = np.random.normal(Gs + self.noise_loc, self.noise_b_s + self.noise_r_s * np.abs(Gs))
               # print(ys.shape)
                
                ys_hist[:, k, :] = ys
                
                # Get rewards.
                # for i in range(n_traj):
                #     rewards_hist[i, k] = self.get_reward(k, 
                #                                          None, 
                #                                          xps_hist[i, k],
                #                                          ds[i],
                #                                          ys[i])
                rewards_hist[:, k] = self.get_rewards(k,
                                                      None,
                                                      xps_hist[:, k],
                                                      ds_hist,
                                                      ys_hist)
                # Update physical state.
                xps = self.xp_f(xps_hist[:, k],
                                k,
                                ds, ys)
                xps_hist[:, k + 1] = xps
            else:
                if self.use_grid_kld:
                    for i in range(n_traj):
                        # Get terminal belief state.
                        xb = self.get_xb(d_hist=ds_hist[i],  y_hist=ys_hist[i])
                        if store_belief_state:
                            xbs[i] = xb
                        # Get reward.
                        rewards_hist[i, k] = self.get_reward(k, 
                                                             xb, 
                                                             xps_hist[i, k],
                                                             None,
                                                             None)
                        # print('*' * (progress_points == i).sum(), end='')
                elif self.use_PCE:
                    for i in trange(n_traj):
                        contrastive_loglikelis = np.zeros((n_contrastive_sample + 1, self.n_stage))
                        for kk in range(self.n_stage):
                            loglikelis = self.loglikeli(kk, ys_hist[i:i+1, kk, :], 
                                np.concatenate([thetas[i:i+1], contrastive_thetas], 0), ds_hist[i:i+1, kk, :], xps_hist[i:i+1, kk, :])
                            contrastive_loglikelis[:, kk] = loglikelis
                        # contrastive_Gs = np.concatenate([Gs_hist[i:i+1], contrastive_Gs], axis=0)
                        # contrastive_likelis = norm_pdf(ys_hist[i:i+1], 
                        #                                contrastive_Gs + self.noise_loc,
                        #                                self.noise_b_s + self.noise_r_s * np.abs(contrastive_Gs))
                        contrastive_likelis = np.exp(contrastive_loglikelis)
                        if not self.use_PCE_incre:
                            contrastive_likelis = contrastive_likelis.prod(axis=-1)
                            contrastive_evid = contrastive_likelis.mean()
                            rewards_hist[i, k] = np.log(contrastive_likelis[0] / contrastive_evid)
                        else:
                            evid_prev = 1.0
                            for kk in range(self.n_stage):
                                evid = contrastive_likelis[:, :kk + 1].prod(axis=-1).mean()
                                rewards_hist[i, kk] = np.log(contrastive_likelis[0, kk] / evid * evid_prev)
                                evid_prev = evid


                else:
                    rewards_hist[:, k] = self.get_rewards(k,
                                                          None,
                                                          xps_hist[:, k],
                                                          ds_hist,
                                                          ys_hist,
                                                          thetas)
                    
                 #   print([k, rewards_hist.sum(-1).mean()])
                    
        averaged_reward = rewards_hist.sum(-1).mean()
        
     
        
#         print("Averaged total reward:  {:.4}".format(averaged_reward))
        
#        # print(dcs_hist.shape, dcs_hist.mean(axis=(0, -1)).shape)
        
#       #  print("Averaged designs:", dcs_hist.mean(axis=(0, -1)))
#         print("Averaged designs:", dcs_hist.mean(axis=(0, 1)))
        
        self.averaged_reward = averaged_reward
        self.thetas = thetas
        self.dcs_hist = dcs_hist
        self.ds_hist = ds_hist
        self.ys_hist = ys_hist
        self.xbs = xbs
        self.xps_hist = xps_hist
        self.rewards_hist = rewards_hist
        if isinstance(use_grid_kld, bool):
            self.use_grid_kld = use_grid_kld_bckup
        if isinstance(use_PCE, bool):
            self.use_PCE = use_PCE_bckup
        if return_all:
            return (averaged_reward, thetas, 
                    dcs_hist, ds_hist, ys_hist, 
                    xbs, xps_hist, 
                    rewards_hist, Gs_hist)
        else:
            return averaged_reward