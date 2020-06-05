import warnings

import numpy as np
import torch

from otbar import Distribution, GridBarycenter


def create_distribution_2d(images):
    """Transform images to sparse format for free support barycenters."""
    n_samples, width, _ = images.shape
    images = images.reshape(n_samples, -1)
    images /= images.sum(axis=1)[:, None]
    pre_supp = [np.where(x)[0] for x in images]
    pre_weights = [x[s] for x, s in zip(images, pre_supp)]
    X = torch.linspace(0., 1., width)
    Y = torch.linspace(0., 1., width)
    X, Y = torch.meshgrid(X, Y)
    X = X.reshape(X.shape[0] ** 2)
    Y = Y.reshape(Y.shape[0] ** 2)

    distributions = []
    for i in range(len(pre_supp)):
        supp = torch.zeros((pre_supp[i].shape[0], 2))
        supp[:, 0] = X[pre_supp[i]]
        supp[:, 1] = Y[pre_supp[i]]
        weights = torch.tensor(pre_weights[i])
        distributions.append(Distribution(supp, weights))
    return distributions


def create_distribution_3d(images, device="cpu"):
    """Transform images to sparse format for free support barycenters."""
    n_samples, width, _, _ = images.shape
    images = images.reshape(n_samples, -1)
    images /= images.sum(axis=1)[:, None]
    pre_supp = [np.where(x)[0] for x in images]
    pre_weights = [x[s] for x, s in zip(images, pre_supp)]
    X = torch.linspace(-1, 1., width)
    Y = torch.linspace(-1, 1., width)
    Z = torch.linspace(-1, 1., width)

    X, Y, Z = torch.meshgrid(X, Y, Z)
    X = X.reshape(X.shape[0] ** 3)
    Y = Y.reshape(Y.shape[0] ** 3)
    Z = Z.reshape(Z.shape[0] ** 3)

    distributions = []
    for i in range(len(pre_supp)):
        supp = torch.zeros((pre_supp[i].shape[0], 3))
        supp[:, 0] = X[pre_supp[i]]
        supp[:, 1] = Y[pre_supp[i]]
        supp[:, 2] = Z[pre_supp[i]]
        weights = torch.tensor(pre_weights[i])
        distributions.append(Distribution(supp, weights, device=device))
    return distributions


def create_distribution_1d(data, bounds=None, device="cpu"):
    """Transform images to sparse format for free support barycenters."""
    n_hists, n_features = data.shape
    if bounds is None:
        bounds = -5, 5
    pre_supp = [np.where(x)[0] for x in data]
    pre_weights = [x[s] for x, s in zip(data, pre_supp)]
    X = torch.linspace(bounds[0], bounds[1], n_features)

    distributions = []
    for ii in range(n_hists):
        supp = X[pre_supp[ii]]
        weights = torch.tensor(pre_weights[ii])
        distributions.append(Distribution(supp[:, None], weights[:, None],
                             device=device))
    return distributions


def distribution_to_density(dist, grid):
    support = dist.support.flatten()
    order = np.argsort(support)
    weights = dist.weights.flatten()[order]
    support = support[order]
    density = torch.zeros_like(grid)
    distances = (grid[:, None] - support[None, :]) ** 2
    positions = distances.argmin(dim=0)
    for ii, position in enumerate(positions):
        density[position] += weights[ii]
    return density


def barycenter_free(distributions, grid_step, epsilon, init=None,
                    maxiter=2000, tol=1e-4, sinkhorn_tol=1e-4,
                    fw_steps=10, support_budget=200, weights=None,
                    grid=None, log_domain=True):
    """Free support barycenters with Frank-Wolf's algorithm."""
    dimension = distributions[0].support.shape[-1]
    device = distributions[0].support.device
    log = dict(time=[], err=[])
    if init is None:
        torch.manual_seed(0)
        n_particles = 10
        positions = torch.rand(n_particles, dimension, device=device)
        init_bary = Distribution(positions, device=device).normalize()
    else:
        init_bary = init
    if weights is None:
        weights = torch.ones(len(distributions)) / len(distributions)
        weights = weights.to(device)
    bary = GridBarycenter(distributions, init_bary, grid=grid,
                          support_budget=support_budget,
                          sinkhorn_tol=sinkhorn_tol, grid_step=grid_step,
                          eps=epsilon, mixing_weights=weights,
                          log_domain=log_domain)
    n_iter_max = maxiter // fw_steps
    for ii in range(n_iter_max):
        print("Doing iter {} ..".format(ii + 1))
        n_particles = len(bary.bary.support)
        weights = bary.bary.weights.clone()
        bary.performFrankWolfe(fw_steps)
        if n_particles == len(bary.bary.weights):
            break
    if ii == maxiter - 1:
        warnings.warn("Frank-wolf did not converge to the desired tolerance")

    return bary.bary, log
