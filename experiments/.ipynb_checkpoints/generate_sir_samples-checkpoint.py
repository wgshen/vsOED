import os
import argparse
import sys
import time

import torch
import torchsde

import numpy as np
import random

from tqdm import trange


# needed for torchsde
sys.setrecursionlimit(1500)


class SIR_SDE(torch.nn.Module):

    noise_type = "general"
    sde_type = "ito"

    def __init__(self, params, population_size):

        super().__init__()
        # parameters: (beta, gamma)
        self.params = params
        self.N = population_size

    # For efficiency: implement drift and diffusion together
    def f_and_g(self, t, x):
        with torch.no_grad():
            x.clamp_(0.0, self.N)

        p_inf = self.params[:, 0] * x.prod(-1) / self.N
        p_inf_sqrt = torch.sqrt(p_inf)
        p_rec = self.params[:, 1] * x[:, 1]

        f_term = torch.stack([-p_inf, p_inf - p_rec], dim=-1)
        g_term = torch.stack(
            [
                torch.stack([-p_inf_sqrt, p_inf_sqrt], dim=-1),
                torch.stack([torch.zeros_like(p_rec), -torch.sqrt(p_rec)], dim=-1),
            ],
            dim=-1,
        )
        return f_term, g_term


def solve_sir_sdes(
    num_samples,
    num_ys,
    device,
    grid=10000,
    savegrad=False,
    save=False,
    foldername="/scratch/xhuan_root/xhuan1/wgshen/vsOED/SIR/sir_sde_data/train_data",
    filename="sir_sde_data.pt",
    params=None,
    theta_loc=None,
    theta_covmat=None,
):
    if params is None:
        ####### Change priors here ######
        if theta_loc is None or theta_covmat is None:
            theta_loc = torch.tensor([0.5, 0.1], device=device).log()
            theta_covmat = torch.eye(2, device=device) * 0.5 ** 2

        prior = torch.distributions.MultivariateNormal(theta_loc, theta_covmat)
        log_params = prior.sample(torch.Size([num_samples]))
        log_probs = prior.log_prob(log_params)
        params = log_params.exp()
        if num_ys > 1:
            params = params.reshape(num_samples, 1, -1)
            params = params.expand(-1, num_ys, -1)
            params = params.reshape(num_samples * num_ys, -1)
        #################################

    T0, T = 0.0, 100.0  # initial and final time
    GRID = grid  # time-grid

    population_size = 500.0
    initial_infected = 2.0  # initial number of infected

    ## [non-infected, infected]
    y0 = torch.tensor(
        num_samples * num_ys * [[population_size - initial_infected, initial_infected]],
        device=device,
    )  # starting point
    ts = torch.linspace(T0, T, GRID, device=device)  # time grid

    sde = SIR_SDE(
        population_size=torch.tensor(population_size, device=device), params=params,
    ).to(device)

    print("Simulation Start")
    start_time = time.time()
    ys = torchsde.sdeint(sde, y0, ts)  # solved sde
    end_time = time.time()
    print("Simulation Time: %s seconds" % (end_time - start_time))

    save_dict = dict()
    # save_dict["ts"] = ts.cpu()
    save_dict["dt"] = (ts[1] - ts[0]).cpu()  # delta-t (time grid)
    
    if num_ys == 1:
        idx_good = torch.where(ys[:, :, 1].mean(0) >= 1)[0]

        save_dict["prior_samples"] = params[idx_good].cpu()
        save_dict["log_prior_samples"] = save_dict["prior_samples"].log()
        save_dict["log_probs"] = log_probs[idx_good].cpu()
        # drop 0 as it's not used (saves space)
        save_dict["ys"] = ys[:, idx_good, 1].T.cpu()
    else:
        save_dict["prior_samples"] = params.reshape(num_samples, num_ys, -1)[:, 0, :].cpu()
        save_dict["log_prior_samples"] = save_dict["prior_samples"].log()
        save_dict["log_probs"] = log_probs.cpu()
        save_dict["ys"] = ys[:, :, 1].T.cpu()

    # # grads can be calculated in backward pass (saves space)
    # if savegrad:
    #     # central difference
    #     grads = (ys[2:, ...] - ys[:-2, ...]) / (2 * save_dict["dt"])
    #     save_dict["grads"] = grads[:, idx_good, :].cpu()

    # meta data
    save_dict["N"] = population_size
    save_dict["I0"] = initial_infected
    save_dict["num_samples"] = save_dict["prior_samples"].shape[0]

    if save:
        print("Saving data.", end=" ")
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        torch.save(save_dict, f"{foldername}/{filename}")

    print("DONE.")
    return save_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIR: solve SIR equations")
    parser.add_argument("--num-train-sample", default=120000, type=int)
    parser.add_argument("--num-train-batch", default=10, type=int)
    parser.add_argument("--num-test-sample", default=12000, type=int)
    parser.add_argument("--num-test-batch", default=30, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--save-folder", default="./SIR/", type=str)

#     if not os.path.exists("data"):
#         os.makedirs("data")

    args = parser.parse_args()
    device = args.device
    parent_folder = args.save_folder
    
    grid = 10001
    
    # Generate training samples
    num_samples = args.num_train_sample
    num_batch = args.num_train_batch
    print(f'Generating {num_batch} training batches, each with {num_samples} trials')
    foldername = parent_folder + "/train_data"
    seed = 3128564
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for i in range(num_batch):
        print(f"{i}-th batch")
        save_dict = solve_sir_sdes(num_samples, 1, device, grid=grid,
                                   save=True, foldername=foldername,
                                   filename=f"sir_train_data_{i}.pt")
        
    # Generate testing samples
    num_samples = args.num_test_sample
    num_batch = args.num_test_batch
    print(f'Generating {num_batch} testing batches, each with {num_samples} trials')
    foldername = parent_folder + "/test_data"
    seed = 84189743
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for i in range(num_batch):
        print(f"{i}-th batch")
        save_dict = solve_sir_sdes(num_samples, 1, device, grid=grid,
                                   save=True, foldername=foldername,
                                   filename=f"sir_test_data_{i}.pt")
    