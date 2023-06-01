import os,sys,inspect
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from vsOED import VSOED, PGvsOED, GMM_NET, NFs, POST_APPROX
from vsOED.utils import *
from vsOED.models import *

import dowel

def str_to_bool(s):
    if isinstance(s, bool):
        return s
    elif s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError # evil ValueError that doesn't tell you what the wrong value was


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convection-diffusion problem")
    parser.add_argument("--id", default=0, type=int) # From 0 to 9
    parser.add_argument("--n-stage", default=20, type=int) # From 1 to 20
    parser.add_argument("--model", default="uni", type=str) # uni or multi
    parser.add_argument("--include-str-wid", default=False, type=str) # True or False
    parser.add_argument("--include-goal", default=False, type=str) # True or False
    parser.add_argument("--include-cost", default=False, type=str) # True or False
    parser.add_argument("--cost-ratio", default=0.2, type=float)
    parser.add_argument("--model-weight", default=0, type=float) # From 0 to 1
    parser.add_argument("--poi-weight", default=1, type=float) # From 0 to 1
    parser.add_argument("--goal-weight", default=0, type=float) # From 0 to 1
    parser.add_argument("--n-incre", default=1, type=int) # From 1 to n_stage
    parser.add_argument("--share-interm-net", default=False, type=str) # True or False
    parser.add_argument("--post-net-type", default='GMM', type=str) # 'GMM' or 'NFs'
    parser.add_argument("--n-mixture", default=8, type=int) # Number of Gaussian mixture
    parser.add_argument("--post-lr", default=1e-3, type=float) 
    parser.add_argument("--post-gamma", default=0.9999, type=float) 
    parser.add_argument("--actor-lr", default=1e-3, type=float) 
    parser.add_argument("--actor-gamma", default=0.9999, type=float) 
    parser.add_argument("--critic-lr", default=1e-3, type=float) 
    parser.add_argument("--critic-gamma", default=0.9999, type=float) 
    
    parser.add_argument("--n-update", default=10001, type=int) 
    parser.add_argument("--n-newtraj", default=1000, type=int) 
    parser.add_argument("--n-batch", default=10000, type=int) 
    parser.add_argument("--n-buffer-init-batch", default=2, type=int) 
    parser.add_argument("--n-buffer-max", default=int(1e6), type=int) 
    parser.add_argument("--buffer-device", default='cuda', type=str) 
    parser.add_argument("--discount", default=1.0, type=float) 
    parser.add_argument("--n-critic-update", default=5, type=int) 
    parser.add_argument("--n-post-approx-update", default=5, type=int) 
    parser.add_argument("--target-lr", default=0.1, type=float) 
    parser.add_argument("--design-noise-scale", default=0.05, type=float) 
    parser.add_argument("--design-noise-decay", default=0.9999, type=float) 
    parser.add_argument("--transition", default=10000, type=int) 
    parser.add_argument("--frozen", default=-1, type=int) 
    
    parser.add_argument("--save-folder", default='./results/conv_diff/', type=str)
    parser.add_argument("--log-every", default=100, type=int)
    parser.add_argument("--save-every", default=1000, type=int)
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():  
        dev = "cuda:0" 
    else:  
        dev = "cpu"  
    device = torch.device(dev) 
    torch.set_default_device(device)
    dtype = torch.float32
    torch.set_default_dtype(dtype)
    
    n_stage = args.n_stage       # Number of stages.
    n_design = 2                 # Number of design variables.
    n_obs = 1                    # Number of observations.
    include_str_wid = str_to_bool(args.include_str_wid)
    include_goal = str_to_bool(args.include_goal)
    include_cost = str_to_bool(args.include_cost)
    
    if args.model == 'uni':
        n_pois = [4]
        n_nuisps = [4] if include_str_wid else [2]
        n_goals = [1] if include_goal else None
    elif args.model == 'multi':
        n_pois = [2, 4, 6]
        n_nuisps = [4, 4, 4] if include_str_wid else [2, 2, 2]
        n_goals = [1, 1, 1] if include_goal else None
        
    model_weight = args.model_weight
    poi_weight   = args.poi_weight
    goal_weight  = args.goal_weight

    id = args.id
    random_state  = TRAIN_SEEDS[id]
    set_random_seed(random_state)

    model = CONV_DIFF(multimodel=(args.model=='multi'), include_str_wid=include_str_wid, 
                      include_goal=include_goal, include_cost=include_cost, cost_ratio=args.cost_ratio)
    prior = model

    mu_bounds = {}
    mu_bounds['poi'] = []
    for n_poi in n_pois:
        mu_bounds['poi'].append([[-1, 2]] * n_poi)
    if n_goals is not None:
        mu_bounds['goal'] = []
        for n_goal in n_goals:
            mu_bounds['goal'].append([[-15, 3]] * n_goal)
    max_sigmas = {}
    max_sigmas['poi'] = []
    for n_poi in n_pois:
        max_sigmas['poi'].append([1.0] * n_poi)
    if n_goals is not None:
        max_sigmas['goal'] = []
        for n_goal in n_goals:
            max_sigmas['goal'].append([4] * n_goal)
    truncnorm_info = {}
    truncnorm_info['poi'] = []
    for n_poi in n_pois:
        truncnorm_info['poi'].append([[i, 0., 1.] for i in range(n_poi)])
    if n_goals is not None:
        truncnorm_info['goal'] = [None] * len(n_goals)


    n_incre = args.n_incre
    share_interm_net = str_to_bool(args.share_interm_net)

    candidate_lrs = [[args.post_lr] * n_incre, [args.post_lr / n_incre] * (n_incre - 1) + [args.post_lr]]
    candidate_gammas = [[args.post_gamma] * n_incre, [0.999999] * (n_incre - 1) + [args.post_gamma]]
    model_post_lrs = candidate_lrs[share_interm_net]
    model_post_lr_scheduler_gammas = candidate_gammas[share_interm_net]
    poi_post_lrs = candidate_lrs[share_interm_net]
    poi_post_lr_scheduler_gammas = candidate_gammas[share_interm_net]
    goal_post_lrs = candidate_lrs[share_interm_net]
    goal_post_lr_scheduler_gammas = candidate_gammas[share_interm_net]

    use_NFs = args.post_net_type != 'GMM'
    n_mixture = args.n_mixture
    activate = nn.ReLU

    log_every = args.log_every
    save_folder = args.save_folder
    try:
        os.remove(save_folder + 'progress.csv')
    except:
        pass
    try:
        os.remove(save_folder + 'progress.txt')
    except:
        pass
    try:
        dowel.logger.remove_all()
    except:
        pass
    dowel.logger.add_output(dowel.StdOutput())
    dowel.logger.add_output(dowel.CsvOutput(save_folder + '/progress.csv'))
    dowel.logger.add_output(dowel.TextOutput(save_folder + '/progress.txt'))

    try:
        for _ in range(100):
            dowel.logger.pop_prefix()
    except:
        pass
    dowel.logger.push_prefix(f'[CONV-DIFF-{n_stage}stage] ')
    dowel.logger.log('Experiment id: ' + str(id))
    dowel.logger.log('Stored at ' + save_folder)
    dowel.logger.log('Random seed:  ' + str(random_state))
    dowel.logger.log('Device: ' + str(device))
    dowel.logger.log('dtype: ' + str(dtype))
    dowel.logger.log('args: ' + str(args))

    post_approx_params = {
        'n_stage': n_stage, 
        'n_design': n_design, 
        'n_obs': n_obs, 
        'n_pois': n_pois,
        'n_nuisps': n_nuisps,
        'n_goals': n_goals,
        'model_weight': model_weight,
        'poi_weight': poi_weight,
        'goal_weight': goal_weight,
        'mu_bounds': mu_bounds,
        'max_sigmas': max_sigmas,
        'truncnorm_info': truncnorm_info,
        'n_mixture': n_mixture,
        'activate': activate,
        'prior': prior,
        'n_incre': n_incre,
        'share_interm_net': share_interm_net,
        'model_post_lrs': model_post_lrs,
        'model_post_lr_scheduler_gammas': model_post_lr_scheduler_gammas,
        'poi_post_lrs': poi_post_lrs,
        'poi_post_lr_scheduler_gammas': poi_post_lr_scheduler_gammas,
        'goal_post_lrs': goal_post_lrs,
        'goal_post_lr_scheduler_gammas': goal_post_lr_scheduler_gammas,
        'use_NFs': use_NFs,
        'log_every': log_every,
        'dowel': dowel}

    dowel.logger.log('Post_approx_params: ' + str(post_approx_params))

    dowel.logger.dump_all()

    post_approx = POST_APPROX(**post_approx_params)
    dowel.logger.dump_all()
    
    n_param = len(n_pois) > 1
    try:
        n_param += max(n_pois)
    except:
        pass
    try:
        n_param += max(n_nuisps)
    except:
        pass
    try:
        n_param += max(n_goals)
    except:
        pass

    design_bounds = [(0, 1), (0, 1)] # lower and upper bounds of design variables.
    nkld_reward_fun = model.reward_fun
    kld_reward_fun = post_approx.reward_fun
    # Physical state
    n_xp = 2
    init_xp = [0.5, 0.5]
    phys_state_info = (n_xp, init_xp, model.xp_f)
    post_approx = post_approx
    encoder_dimns = None
    backend_dimns = None
    actor_dimns = [256, 256, 256]
    critic_dimns = [256, 256, 256]

    vsoed_params = {
        'n_stage': n_stage,
        'n_param': n_param,
        'n_design': n_design,
        'n_obs': n_obs,
        'model': model,
        'prior': prior,
        'design_bounds': design_bounds,
        'nkld_reward_fun': nkld_reward_fun,
        'kld_reward_fun': kld_reward_fun,
        'phys_state_info': phys_state_info,
        'post_approx': post_approx,
        'encoder_dimns': encoder_dimns,
        'backend_dimns': backend_dimns,
        'actor_dimns': actor_dimns,
        'critic_dimns': critic_dimns,
        'activate': activate
    }
    dowel.logger.log('vsoed_params: ' + str(vsoed_params))

    vsoed = PGvsOED(**vsoed_params)
    
    actor_lr = args.actor_lr
    actor_lr_scheduler_gamma = args.actor_gamma
    critic_lr = args.critic_lr
    critic_lr_scheduler_gamma = args.critic_gamma

    n_update = args.n_update
    n_newtraj = args.n_newtraj
    n_batch = args.n_batch
    n_buffer_init = n_batch * args.n_buffer_init_batch
    n_buffer_max = args.n_buffer_max
    buffer_device = torch.device(args.buffer_device)
    discount = args.discount
    encoder_actor_optimizer = None
    encoder_actor_lr_scheduler = None
    encoder_critic_optimizer = None
    encoder_critic_lr_scheduler = None
    actor_optimizer = optim.Adam(vsoed.actor_net.parameters(), lr=actor_lr)
    actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(actor_optimizer, gamma=actor_lr_scheduler_gamma)
    n_critic_update = args.n_critic_update
    critic_optimizer = optim.Adam(vsoed.critic_net.parameters(), lr=critic_lr)
    critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=critic_lr_scheduler_gamma)
    n_post_approx_update = args.n_post_approx_update
    lr_target = args.target_lr
    design_noise_scale = args.design_noise_scale
    design_noise_decay = args.design_noise_decay
    on_policy = False
    use_PCE = False
    use_PCE_incre = None
    n_contrastive_sample = None
    transition = args.transition
    frozen = args.frozen
    save_every = args.save_every
    restart = False

    vsoed_train_params = {
        'n_update': n_update,
        'n_newtraj': n_newtraj,
        'n_batch': n_batch,
        'n_buffer_init': n_buffer_init,
        'n_buffer_max': n_buffer_max,
        'buffer_device': buffer_device,
        'discount': discount,
        'encoder_actor_optimizer': encoder_actor_optimizer,
        'encoder_actor_lr_scheduler': encoder_actor_lr_scheduler,
        'encoder_critic_optimizer': encoder_critic_optimizer,
        'encoder_critic_lr_scheduler': encoder_critic_lr_scheduler,
        'actor_lr': actor_lr,
        'actor_optimizer': actor_optimizer,
        'actor_lr_scheduler_gamma': actor_lr_scheduler_gamma, 
        'actor_lr_scheduler': actor_lr_scheduler,
        'n_critic_update': n_critic_update,
        'critic_lr': critic_lr,
        'critic_optimizer': critic_optimizer,
        'critic_lr_scheduler_gamma': critic_lr_scheduler_gamma,
        'critic_lr_scheduler': critic_lr_scheduler,
        'n_post_approx_update': n_post_approx_update,
        'lr_target': lr_target,
        'design_noise_scale': design_noise_scale,
        'design_noise_decay': design_noise_decay,
        'on_policy': on_policy,
        'use_PCE': use_PCE,
        'use_PCE_incre': use_PCE_incre,
        'n_contrastive_sample': n_contrastive_sample,
        'transition': transition,
        'frozen': frozen,
        'log_every': log_every,
        'dowel': dowel,
        'save_every': save_every,
        'save_path': save_folder,
        'restart': restart
    }

    dowel.logger.log('vsoed_train_params: ' + str(vsoed_train_params))

    del vsoed_train_params['actor_lr']
    del vsoed_train_params['actor_lr_scheduler_gamma']
    del vsoed_train_params['critic_lr']
    del vsoed_train_params['critic_lr_scheduler_gamma']
    
    try:
        for _ in range(100):
            dowel.logger.pop_prefix()
    except:
        pass
    dowel.logger.push_prefix(f'[CONV-DIFF-{n_stage}stage] ')
    vsoed.train(**vsoed_train_params)
    torch.save(torch.tensor(vsoed.update_hist), save_folder + '/update_hist.pt')
    set_random_seed(EVAL_SEEDS[id])
    dowel.logger.log('Evaluating using post approx...')
    vsoed.asses(int(1e6), use_PCE=False, return_all=True, dowel=dowel,return_nkld_rewards=True, save_path=save_folder + '/evaluation_with_post_approx.pt')