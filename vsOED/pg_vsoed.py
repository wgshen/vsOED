import math
import torch
import torch.nn as nn
import torch.optim as optim
from .vsoed import VSOED
from .utils import *
import time
from tqdm import trange
import joblib

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
    get_action_values()
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
    def __init__(self, 
                 n_stage, n_param, n_design, n_obs, 
                 model, prior, design_bounds, 
                 nkld_reward_fun=None, kld_reward_fun=None, 
                 phys_state_info=None, post_approx=None, 
                 encoder_dimns=None, backend_dimns=None,
                 actor_dimns=None, critic_dimns=None, activate=None,
                 random_state=None):
        super().__init__(n_stage, n_param, n_design, n_obs, 
                         model, prior, design_bounds,
                         nkld_reward_fun, kld_reward_fun,
                         phys_state_info, post_approx, random_state)

        if activate is None:
            activate = nn.ReLU
        self.activate = activate

        assert encoder_dimns is None or backend_dimns is None
        # Initialize the actor (policy) network and critic network.
        if encoder_dimns is not None:
            self.use_encoder = True 
            self.encoder_input_dimn = self.n_design + self.n_obs + self.n_xp
            encoder_output_dimn = encoder_dimns[-1]
            self.encoder_dimns = encoder_dimns
            self.actor_input_dimn = encoder_output_dimn + self.n_xp
            self.critic_input_dimn = encoder_output_dimn + self.n_xp + self.n_design
        else:
            self.use_encoder = False
            self.backend_input_dimn = (self.n_stage + (self.n_stage - 1) * (self.n_obs + self.n_design))
            if backend_dimns is None:
                self.backend_net = None
                backend_output_dimn = self.backend_input_dimn
            else:
                backend_output_dimn = backend_dimns[-1]
            self.backend_dimns = backend_dimns
            self.actor_input_dimn = backend_output_dimn
            self.critic_input_dimn = backend_output_dimn + self.n_design
        if actor_dimns is None:
            actor_dimns = [256, 256, 256]
        if critic_dimns is None:
            critic_dimns = [256, 256, 256]
        self.actor_dimns = actor_dimns
        self.critic_dimns = critic_dimns
        self.initialize()


        # self.initialize_policy = self.initialize_actor
        # self.load_policy = self.load_actor
        # self.get_policy = self.get_actor

        self.actor = partial(self.get_designs, return_all_stages=False, use_target=False)
        self.actors = partial(self.get_designs, return_all_stages=True, use_target=False)
        self.actor_target = partial(self.get_designs, return_all_stages=False, use_target=True)
        self.actors_target = partial(self.get_designs, return_all_stages=True, use_target=True)
        self.critic = partial(self.get_action_values, return_all_stages=False, use_target=False)
        self.critics = partial(self.get_action_values, return_all_stages=True, use_target=False)
        self.critic_target = partial(self.get_action_values, return_all_stages=False, use_target=True)
        self.critics_target = partial(self.get_action_values, return_all_stages=True, use_target=True)


    def initialize(self):
        if self.use_encoder:
            encoder_dimns = [self.encoder_input_dimn] + list(self.encoder_dimns)
            self.encoder_actor_net = initialize_encoder(encoder_dimns, self.activate)
            self.encoder_critic_net = initialize_encoder(encoder_dimns, self.activate)
            self.encoder_actor_optimizer = None
            self.encoder_actor_lr_scheduler = None
            self.encoder_critic_optimizer = None
            self.encoder_critic_lr_scheduler = None
        self.initialize_backend(self.backend_dimns)
        self.initialize_actor(self.actor_dimns)
        self.initialize_critic(self.critic_dimns)
        self.design_noise_scale = None
        try:
            self.post_approx.reset()
        except:
            pass
        self.update = 0

    def initialize_backend(self, backend_dimns):
        if backend_dimns is not None:
            self.backend_dimns = backend_dimns.copy()
            backend_dimns = [self.backend_input_dimn] + list(backend_dimns)
            self.backend_net = Net(backend_dimns, nn.ReLU(), 
                None, 'backend')
            self.backend_target_net = Net(backend_dimns, nn.ReLU(), 
                None, 'backend')
            self.backend_target_net.load_state_dict(self.backend_net.state_dict())
            self.update = 0
        else:
            self.backend_net = None 
            self.backend_target_net = None

    def initialize_actor(self, actor_dimns=None):
        """
        Initialize the actor (policy) network.

        Parameters
        ----------
        actor_dimns : list, tuple or numpy.ndarray, optional(default=None)
            The dimensions of hidden layers of actor (policy) network.
        """
        assert isinstance(actor_dimns, (list, tuple)), (
               "actor_dimns should be a list or tuple of "
               "integers.")
        output_dimn = self.n_design
        self.actor_dimns = actor_dimns.copy()
        actor_dimns = [self.actor_input_dimn] + list(actor_dimns) + [output_dimn]
        
        self.actor_net = Net(actor_dimns, self.activate, self.design_bounds, 
            'actor', self.backend_net)
        self.actor_target_net = Net(actor_dimns, self.activate, self.design_bounds, 
            'actor', self.backend_target_net)
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
        assert isinstance(critic_dimns, (list, tuple)), (
               "critic_dimns should be a list or tuple of ",
               "integers.")
        output_dimn = 1
        self.critic_dimns = critic_dimns.copy()
        critic_dimns = [self.critic_input_dimn] + list(critic_dimns) + [output_dimn]
        self.critic_net = Net(critic_dimns, self.activate, 
            np.array([[-np.inf, np.inf]]), 'critic', self.backend_net, self.n_design)
        self.critic_target_net = Net(critic_dimns, self.activate, 
            np.array([[-np.inf, np.inf]]), 'critic', self.backend_target_net, self.n_design)
        self.critic_target_net.load_state_dict(self.critic_net.state_dict())
        self.update = 0
        self.critic_optimizer = None  
        self.critic_lr_scheduler = None   

    # def load_actor(self, net, optimizer=None):
    #     """
    #     Load the actor network with single precision.

    #     Parameters
    #     ----------
    #     net : nn.Module
    #         A pre-trained PyTorch network with input dimension actor_input_dimn 
    #         and output dimension n_design.
    #     optimizer : algorithm of torch.optim
    #         An optimizer corresponding to net.
    #     """
    #     try:
    #         net = net.to(self.device, self.dtype)
    #         output = net(torch.zeros(1, self.actor_input_dimn).to(self.device, self.dtype))
    #         assert output.shape[1] == self.n_design, (
    #                "Output dimension should be {}.".format(self.n_design))
    #         self.actor_net = net
    #     except:
    #         print("Actor network should has "
    #               "input dimension {}.".format(self.actor_input_dimn))
    #     self.actor_optimizer = optimizer
    #     self.update = 0

    # def load_critic(self, net, optimizer=None):
    #     """
    #     Load the critic network with single precision.

    #     Parameters
    #     ----------
    #     net : nn.Module
    #         A pre-trained PyTorch network with input dimension critic_input_dimn
    #         and output dimension 1.
    #     optimizer : algorithm of torch.optim
    #         An optimizer corresponding to net.
    #     """
    #     try:
    #         net = net.to(self.device, self.dtype)
    #         output = net(torch.zeros(1, self.critic_input_dimn).to(self.device, self.dtype))
    #         assert output.shape[1] == 1, (
    #                "Output dimension should be 1.")
    #         self.critic_net = net
    #     except:
    #         print("Critic network should has input dimension {}.".format(self.critic_input_dimn))
    #     self.critic_optimizer = optimizer
    #     self.update = 0

    # def get_actor(self):
    #     return self.actor_net

    # def get_critic(self):
    #     return self.critic_net

    def form_actor_input(self, stage, ds_hist, ys_hist):
        """
        A function to form the inputs of actor network.

        Parameters
        ----------
        stage : int
            The stage index. 0 <= stage <= n_stage - 1.
        ds_hist : numpy.ndarray of size (n_traj or 1, stage, n_design)
            n_traj sequences of designs before stage "stage".
        ys_hist : numpy.ndarray of size (n_traj or 1, stage, n_obs)
            n_traj sequences of observations before stage "stage". 

        Returns
        -------
        A torch.Tensor of size (n_traj, dimn_actor_input).
        """
        n_traj = max(len(ds_hist), len(ys_hist))
        # Inputs.
        X = torch.zeros(n_traj, self.backend_input_dimn)
        # Index of experiments.
        X[:, stage] = 1
        # Historical designs.
        begin = self.n_stage
        end = begin + ds_hist.shape[1:].numel()
        X[:, begin:end] = ds_hist.reshape(len(ds_hist), end - begin)
        # Historical observations.
        begin = self.n_stage + (self.n_stage - 1) * self.n_design
        end = begin + ys_hist.shape[1:].numel()
        X[:, begin:end] = ys_hist.reshape(len(ys_hist), end - begin)
        return X

    def form_critic_input(self, stage, ds_hist, ys_hist, ds):
        """
        A function to form the inputs of critic network.

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
        A torch.Tensor of size (n_traj, critic_input_dimn).
        """
        n_traj = max(len(ds_hist), len(ys_hist))
        X = torch.zeros(n_traj, self.backend_input_dimn + self.n_design)
        X[:, :self.backend_input_dimn] = self.form_actor_input(stage, ds_hist, ys_hist)
        X[:, -self.n_design:] = ds
        return X

    def get_designs(self, stage, ds_hist, ys_hist, xps_hist, 
        return_all_stages=False, use_target=False):
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
        if not self.use_encoder:
            assert ds_hist.shape[1] == stage and ys_hist.shape[1] == stage
            X = self.form_actor_input(stage, ds_hist, ys_hist)
            if not use_target:
                designs = self.actor_net(X).detach()
            else:
                designs = self.actor_target_net(X).detach()
        else:
            states = get_encoded_states(self.encoder_actor_net, ds_hist, ys_hist, xps_hist, return_all_stages)
            if use_target:
                designs = self.actor_target_net(states)
            else:
                designs = self.actor_net(states)
        return designs

    def get_action_values(self, stage, ds_hist, ys_hist, xps_hist, ds, 
        return_all_stages=False, use_target=False, state_no_grad=False):
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
        if not self.use_encoder:
            assert ds_hist.shape[1] == stage and ys_hist.shape[1] == stage
            X = self.form_critic_input(stage, ds_hist, ys_hist, ds)
            if not use_target:
                values = self.critic_net(X).detach()
            else:
                values = self.critic_target_net(X).detach()
        else:
            if return_all_stages:
                assert ds.shape[1] == xps_hist.shape[1]
            else:
                assert len(ds.shape) == 2
            states = get_encoded_states(self.encoder_critic_net, ds_hist, ys_hist, xps_hist, return_all_stages)
            if state_no_grad:
                states = states.detach()
            X = torch.cat([states, ds], dim=-1)
            if use_target:
                values = self.critic_target_net(X)
            else:
                values = self.critic_net(X)
        return values[..., 0]



    def train(self, 
              n_update=int(1e4), 
              n_newtraj=1000, 
              n_batch=10000,
              n_buffer_init=20000,
              n_buffer_max=int(1e5), 
              buffer_device=torch.device('cpu'),
              discount=1,
              encoder_actor_optimizer=None,
              encoder_actor_lr_scheduler=None,
              encoder_critic_optimizer=None,
              encoder_critic_lr_scheduler=None,
              actor_optimizer=None,
              actor_lr_scheduler=None,
              n_critic_update=5,
              critic_optimizer=None,
              critic_lr_scheduler=None,
              n_post_approx_update=5,
              lr_target=0.05,
              design_noise_scale=None, 
              design_noise_decay=0.9999,
              on_policy=False,
              use_PCE=None,
              use_PCE_incre=None,
              n_contrastive_sample=10000,
              transition=5000,
              frozen=-1,
              log_every=1,
              dowel=None,
              save_every=1000,
              save_path=None,
              restart=False):
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
        t0 = time.time()
        assert n_update >= self.update
        assert n_buffer >= n_newtraj
        assert n_buffer >= n_batch
        self.n_update = n_update
        if self.update == 0 or restart:
            self.n_newtraj = n_newtraj
            self.n_batch = n_batch
            self.n_buffer_init = n_buffer_init
            self.n_buffer_max = n_buffer_max
            self.buffer_device = buffer_device
            self.discount = discount
            self.n_critic_update = n_critic_update
            self.n_post_approx_update = n_post_approx_update
            self.lr_target = lr_target
            if design_noise_scale is None:
                self.design_noise_scale = (self.design_bounds[:, 1] -  self.design_bounds[:, 0]) / 20                
                self.design_noise_scale[self.design_noise_scale == float('inf')] = 5
            elif isinstance(design_noise_scale, (list, tuple)):
                self.design_noise_scale = torch.tensor(design_noise_scale)
            else:
                self.design_noise_scale = design_noise_scale
            assert design_noise_decay > 0 and design_noise_decay <= 1
            self.design_noise_decay = design_noise_decay
            if isinstance(use_PCE, bool):
                self.use_PCE = use_PCE
                if isinstance(use_PCE_incre, bool):
                    self.use_PCE_incre = use_PCE_incre
                else:
                    self.use_PCE_incre = False
            else:
                self.use_PCE = self.use_PCE_incre = False
            self.n_contrastive_sample = n_contrastive_sample
            self.transition = transition
            self.frozen = frozen
            self.log_every = log_every
            self.dowel = dowel
            self.save_every = save_every
            self.save_path = save_path

        if self.use_encoder:
            if self.encoder_actor_optimizer is None:
                self.encoder_actor_optimizer = encoder_actor_optimizer
            if self.encoder_actor_lr_scheduler is None:
                self.encoder_actor_lr_scheduler = encoder_actor_lr_scheduler
            if self.encoder_critic_optimizer is None:
                self.encoder_critic_optimizer = encoder_critic_optimizer
            if self.encoder_critic_lr_scheduler is None:
                self.encoder_critic_lr_scheduler = encoder_critic_lr_scheduler
        if actor_optimizer is None:
            if self.actor_optimizer is None:
                self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=3e-4)
        else:
            self.actor_optimizer = actor_optimizer
        if actor_lr_scheduler is None:
            if self.actor_lr_scheduler is None:
                self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.actor_optimizer, gamma=0.9999)
        else:
            self.actor_lr_scheduler = actor_lr_scheduler
        if critic_optimizer is None:
            if self.critic_optimizer is None:
                self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-4)
        else:
            self.critic_optimizer = critic_optimizer
        if critic_lr_scheduler is None:
            if self.critic_lr_scheduler is None:
                self.critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(
                    self.critic_optimizer, gamma=0.9999)
        else:
            self.critic_lr_scheduler = critic_lr_scheduler

        # Initialize replay buffer
        buffer = {}
        self.asses(self.n_buffer_init, self.design_noise_scale, True, self.use_PCE, self.use_PCE_incre, self.n_contrastive_sample)
        buffer['thetas'] = self.thetas.to(self.buffer_device)
        buffer['ds_hist'] = self.ds_hist.to(self.buffer_device)
        buffer['ys_hist'] = self.ys_hist.to(self.buffer_device)
        buffer['xps_hist'] = self.xps_hist.to(self.buffer_device)
        if self.use_PCE:
            buffer['rewards_hist'] = self.rewards_hist.to(self.buffer_device)
        self.update_hist = []
        p_max = torch.arange(1, self.n_buffer_max + 1).to(torch.float32)
        p = []

        critic_loss_fun = nn.MSELoss()
            
        while self.update < self.n_update:
            t1 = time.time()
            self.asses(self.n_newtraj, self.design_noise_scale, True, self.use_PCE, self.use_PCE_incre, self.n_contrastive_sample)
            self.dowel.logger.push_prefix(f'epoch #{self.update} | ')
            self.dowel.tabular.clear()
            rewards = self.rewards_hist.sum(-1)
            self.dowel.tabular.record('Epoch', self.update)
            self.dowel.tabular.record('Reward/MeanReward', rewards.mean().item())
            self.dowel.tabular.record('Reward/StdReward', rewards.std().item())
            self.dowel.tabular.record('Design/MeanDesign', self.dcs_hist.mean(dim=(0, 1)).tolist())
            self.dowel.tabular.record('Design/StdDesign', self.dcs_hist.std(dim=(0, 1)).tolist())
            self.dowel.tabular.record('ReplayBuffer/buffer_size', len(buffer['thetas']))
            log(self.dowel.tabular, self.dowel, self.update, self.log_every)
            self.update_hist.append(self.averaged_reward)   
            # Update the replay buffer
            l_buffer = len(buffer['thetas'])
            idx_left = -min(l_buffer, n_buffer_max - n_newtraj)
            buffer['thetas'] = torch.cat([buffer['thetas'][idx_left:], self.thetas.to(self.buffer_device)], 0)
            buffer['ds_hist'] = torch.cat([buffer['ds_hist'][idx_left:], self.ds_hist.to(self.buffer_device)], 0)
            buffer['ys_hist'] = torch.cat([buffer['ys_hist'][idx_left:], self.ys_hist.to(self.buffer_device)], 0)
            buffer['xps_hist'] = torch.cat([buffer['xps_hist'][idx_left:], self.xps_hist.to(self.buffer_device)], 0)
            if self.use_PCE:
                buffer['rewards_hist'] = torch.cat([buffer['rewards_hist'][idx_left:], self.rewards_hist.to(self.buffer_device)], 0)
            l_buffer = len(buffer['thetas'])
            if len(p) < self.n_buffer_max:
                p = p_max[:l_buffer]
                p /= p.sum()
            idxs_pick = torch.multinomial(p, self.n_batch, replacement=False)
            self.thetas = buffer['thetas'][idxs_pick].to(self.thetas.device)
            self.ds_hist = buffer['ds_hist'][idxs_pick].to(self.ds_hist.device)
            self.ys_hist = buffer['ys_hist'][idxs_pick].to(self.ys_hist.device)
            self.xps_hist = buffer['xps_hist'][idxs_pick].to(self.xps_hist.device)
            if self.use_PCE:
                self.rewards_hist = buffer['rewards_hist'][idxs_pick].to(self.rewards_hist.device)
            ############################################
            # Train the post approxer
            if self.post_approx is not None and not self.use_PCE:
                self.rewards_hist = torch.zeros(self.n_batch, self.n_stage + 1)
                self.post_approx.train(self.ds_hist, self.ys_hist, self.xps_hist, self.thetas, self.update, self.n_post_approx_update)
                for k in range(self.n_stage + 1):
                    self.rewards_hist[:, k] = self.get_rewards(k,
                                                               self.ds_hist[:, :k+1],
                                                               self.ys_hist[:, :k+1],
                                                               self.xps_hist[:, :k+2],
                                                               self.thetas,
                                                               True)
            ############################################
            if self.update <= frozen:
                self.dowel.logger.pop_prefix()
                self.update += 1
                continue
            # Form the inputs and target values of critic network, and form the 
            # inputs of the actor network.
            with torch.no_grad():
                g_critic = torch.zeros(self.n_batch, self.n_stage)
                if not self.use_encoder:
                    X_critic = torch.zeros(self.n_batch, self.n_stage, self.backend_input_dimn + self.n_design)
                factor = 1 if self.transition <= 0 else min(self.update / self.transition, 1)

                if factor < 1:
                    for k in range(self.n_stage):
                        rewards = self.rewards_hist[:, k:]
                        discounts = self.discount ** torch.arange(rewards.shape[1])
                        discounts[-1] = discounts[-2]
                        g_critic[:, k] = (rewards * discounts).sum(-1) * (1 - factor)
                if self.use_encoder:
                    if on_policy:
                        ds_next = self.ds_hist[:, 1:]
                        ds_next = torch.cat([torch.zeros(len(ds_next), 1, self.n_design), ds_next], dim=1)
                        values_next = self.critics(self.ds_hist[:, :-1], self.ys_hist[:, :-1], 
                            self.xps_hist[:, :-1], ds_next)[:, 1:]
                    else:
                        ds_next = self.actors_target(self.ds_hist[:, :-2], self.ys_hist[:, :-2], 
                            self.xps_hist[:, :-2])
                        ds_next = torch.cat([torch.zeros(len(ds_next), 1, self.n_design),
                            ds_next], dim=1)
                        values_next = self.critics_target(self.ds_hist[:, :-1], self.ys_hist[:, :-1], 
                            self.xps_hist[:, :-1], ds_next)[:, 1:]
                    for k in range(self.n_stage - 1):
                        g_critic[:, k] += (self.discount * values_next[:, k] 
                        + self.rewards_hist[:, k]) * factor
                else:
                    for k in range(self.n_stage):
                        X = self.form_critic_input(k,
                                                   self.ds_hist[:, :k],
                                                   self.ys_hist[:, :k],
                                                   self.ds_hist[:, k])
                        X_critic[:, k, :] = X
                        if on_policy:
                            ds_next = self.ds_hist[:, k + 1]
                            next_action_value = self.critic(k + 1, 
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None,
                                ds_next)
                        else:
                            ds_next = self.actor_target(k + 1, 
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None)
                            next_action_value = self.critic_target(k + 1, 
                                self.ds_hist[:, :k + 1],
                                self.ys_hist[:, :k + 1],
                                None,
                                ds_next)
                        g_critic[:, k] += (self.discount * next_action_value 
                        + self.rewards_hist[:, k]) * factor
            # Train critic.
            for _ in range(n_critic_update):
                if not use_encoder:
                    y_critic = self.critic_net(X_critic)
                else:
                    y_critic = self.critics(self.ds_hist[:, :-1], self.ys_hist[:, :-1], self.xps_hist[:, :-1], self.ds_hist)
                    self.encoder_critic_optimizer.zero_grad()
                loss = critic_loss_fun(g_critic, y_critic)
                self.critic_optimizer.zero_grad()
                loss.backward()
                if self.use_encoder: self.encoder_critic_optimizer.step()
                self.critic_optimizer.step()

            if self.use_encoder: self.encoder_critic_lr_scheduler.step()
            self.critic_lr_scheduler.step()

            # Train actor
            if not self.use_encoder:
                X_actor = X_critic[..., :-self.n_design]
                designs = self.actor_net(X_actor)
                X = torch.cat([X_actor, designs], dim=-1)
                output = -self.critic_net(X).mean()
            else:
                designs = self.actors(self.ds_hist[:, :-1], self.ys_hist[:, :-1], self.xps_hist[:, :-1])
                output = -self.critics(self.ds_hist[:, :-1], self.ys_hist[:, :-1], self.xps_hist[:, :-1], 
                    designs, state_no_grad=True).mean()
                self.encoder_actor_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            output.backward()
            self.actor_optimizer.step()
            self.actor_lr_scheduler.step()
            if self.use_encoder: 
                self.encoder_actor_optimizer.step()
                self.encoder_actor_lr_scheduler.step()

            for param, target_param in zip(self.actor_net.parameters(), self.actor_target_net.parameters()):
                target_param.data.copy_(self.lr_target * param.data + (1 - self.lr_target) * target_param.data)
            for param, target_param in zip(self.critic_net.parameters(), self.critic_target_net.parameters()):
                target_param.data.copy_(self.lr_target * param.data + (1 - self.lr_target) * target_param.data)

            self.design_noise_scale *= design_noise_decay
            self.update += 1
            if self.update % self.save_every and self.save_path is not None:
                joblib.dump(self, save_path + f'/itr_{self.update}.pkl')
            log(f'Checkpoint saved', self.dowel, 1, 1)

            t2 = time.time()
            log(f'Total time {t2 - t0:.2f} s', self.dowel, self.update, self.log_every)
            log(f'Epoch time {t2 - t1:.2f} s', self.dowel, self.update, self.log_every)
            self.dowel.logger.pop_prefix()

        if self.dowel:
            self.dowel.logger.remove_all()
            
    def asses(self, n_traj=10000, design_noise_scale=None, in_training=False,
              use_PCE=True, use_PCE_incre=True, n_contrastive_sample=10000,
              return_nkld_rewards=False, return_all=False, 
              theta_samples=None, save_path=None, dowel=None):
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
        
        if design_noise_scale is None:
            design_noise_scale = torch.zeros(self.n_design)
        elif isinstance(design_noise_scale, (int, float)):
            design_noise_scale = torch.ones(self.n_design) * design_noise_scale
        elif isinstance(design_noise_scale, (list, tuple)):
            assert (isinstance(design_noise_scale, (list, tuple))
                    and len(design_noise_scale) == self.n_design)
            design_noise_scale = torch.tensor(design_noise_scale)

        if theta_samples is None:
            thetas = self.prior_rvs(n_traj)
        else:
            assert len(theta_samples) == n_traj
            thetas = theta_samples

        if use_PCE:
            contrastive_thetas = self.prior_rvs(n_contrastive_sample)
            
        dcs_hist = torch.zeros(n_traj, self.n_stage, self.n_design)
        ds_hist = torch.zeros(n_traj, self.n_stage, self.n_design)
        ys_hist = torch.zeros(n_traj, self.n_stage, self.n_obs)
        
        # Store n_stage + 1 physical states.
        xps_hist = torch.zeros(n_traj, self.n_stage + 1, self.n_xp)
        xps_hist[:, 0] = self.init_xp
        if return_nkld_rewards:
            nkld_rewards_hist = torch.zeros(n_traj, self.n_stage + 1)
        else:
            nkld_rewards_hist = 0
        rewards_hist = torch.zeros(n_traj, self.n_stage + 1)

        for k in range(self.n_stage + 1):
            if k < self.n_stage:
                # Get clean designs.
                with torch.no_grad():
                    dcs = self.actor(k, ds_hist[:, :k], ys_hist[:, :k], xps_hist[:, :k+1])
                
                dcs_hist[:, k, :] = dcs
                # Add design noise for exploration. 
                ds = dcs + torch.randn(dcs.shape) * design_noise_scale
                ds = torch.maximum(ds, self.design_bounds[:, 0])
                ds = torch.minimum(ds, self.design_bounds[:, 1])
                ds_hist[:, k, :] = ds
                
                # Run the forward model to get observations.
                ys = self.m_f(k, thetas, ds, xps_hist[:, k, :])
                ys_hist[:, k, :] = ys
                
                # Update physical state.
                xps = self.xp_f(k, xps_hist[:, k], ds, ys)
                xps_hist[:, k + 1] = xps

                # Get rewards.
                rewards_hist[:, k] += self.get_rewards(k,
                                                       ds_hist[:, :k+1],
                                                       ys_hist[:, :k+1],
                                                       xps_hist[:, :k+2],
                                                       thetas,
                                                       not use_PCE)
            else:
                if use_PCE:
                    rewards_hist[:, k] += self.get_rewards(k,
                                                           ds_hist,
                                                           ys_hist,
                                                           xps_hist,
                                                           thetas,
                                                           False)
                    range_ = range if in_training else trange
                    for i in range_(n_traj):
                        # contrastive_loglikelis = torch.zeros(n_contrastive_sample + 1, self.n_stage)
                        contrastive_loglikelis = None
                        contrastive_samples = torch.cat([thetas[i:i+1], contrastive_thetas], dim=0)
                        for kk in range(self.n_stage):
                            loglikelis = self.loglikeli(kk, ys_hist[i:i+1, kk, :], contrastive_samples, 
                                                        ds_hist[i:i+1, kk, :], xps_hist[i:i+1, kk, :], thetas[i])
                            if contrastive_loglikelis is None:
                                contrastive_loglikelis = loglikelis.reshape(-1, 1) 
                            else:
                                contrastive_loglikelis = torch.cat([contrastive_loglikelis, loglikelis.reshape(-1, 1)], dim=1)
                        contrastive_likelis = torch.exp(contrastive_loglikelis)
                        if not self.use_PCE_incre:
                            contrastive_likelis = contrastive_likelis.prod(dim=-1)
                            contrastive_evid = contrastive_likelis.mean()
                            rewards_hist[i, k] += torch.log(contrastive_likelis[0] / contrastive_evid)
                        else:
                            evid_prev = 1.0
                            for kk in range(self.n_stage):
                                evid = contrastive_likelis[:, :kk + 1].prod(dim=-1).mean()
                                rewards_hist[i, kk] += torch.log(contrastive_likelis[0, kk] / evid * evid_prev)
                                evid_prev = evid
                else:
                    rewards_hist[:, k] += self.get_rewards(k,
                                                           ds_hist,
                                                           ys_hist,
                                                           xps_hist,
                                                           thetas,
                                                           True)

            if return_nkld_rewards:
                nkld_rewards_hist[:, k] += self.get_rewards(k,
                                                            ds_hist[:, :k+1],
                                                            ys_hist[:, :k+1],
                                                            xps_hist[:, :k+2],
                                                            thetas,
                                                            False)
                       
        averaged_reward = rewards_hist.sum(-1).mean().item()
        
        self.averaged_reward = averaged_reward
        self.thetas = thetas
        self.dcs_hist = dcs_hist
        self.ds_hist = ds_hist
        self.ys_hist = ys_hist
        self.xbs = xbs
        self.xps_hist = xps_hist
        self.rewards_hist = rewards_hist

        ret = {
        'averaged_reward': averaged_reward,
        'thetas': thetas,
        'dcs_hist': dcs_hist,
        'ds_hist': ds_hist,
        'ys_hist': ys_hist,
        'xps_hist': xps_hist,
        'nkld_rewards_hist': nkld_rewards_hist,
        'rewards_hist': rewards_hist
        }

        if not in_training:
            if save_path is not None:
                torch.save(ret, save_path)
            if self.dowel is not None:
                self.dowel.logger.log('Evaluating...')
                for k in range(1, rewards_hist.shape[1]):
                    self.dowel.logger.log(f'{k}-horizon averaged reward: {rewards_hist[:, :k].sum(-1).mean().item():.3f}')
                self.dowel.logger.log(f'total averaged reward: {rewards_hist.sum(-1).mean().item():.3f}')
                self.dowel.logger.log(f'total reward se: {rewards_hist.sum(-1).std().item() / math.sqrt(len(rewards_hist)):.3f}')

        if return_all:
            return ret
        else:
            return averaged_reward