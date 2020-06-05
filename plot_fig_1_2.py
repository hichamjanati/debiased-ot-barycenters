import numpy as np
import torch

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from sinkhorn_barycenters import barycenter
from utils import gengaussians

from scipy.optimize import fsolve


params = {"legend.fontsize": 18,
          "axes.titlesize": 16,
          "axes.labelsize": 16,
          "xtick.labelsize": 13,
          "ytick.labelsize": 13,
          "pdf.fonttype": 42}
plt.rcParams.update(params)


def sigma_lebesgue(sigmas, weights, epsilon):
    """Computes the theoretical variance of the blurred-biased barycenter."""
    def func(sigma):
        f = 4 * sigma ** 2 - epsilon
        f -= weights.dot((epsilon ** 2 + 16 * sigma ** 2 * sigmas ** 2) ** 0.5)
        return f
    sigma0 = sigmas.mean()
    sigma = fsolve(func, sigma0, xtol=1e-10)
    return sigma


def sigma_debiased(sigmas, weights, epsilon):
    """Computes the theoretical variance of the debiased barycenter."""
    def func(sigma):
        f = (16 * sigma ** 4 + epsilon ** 2) ** 0.5
        f -= weights.dot((epsilon ** 2 + 16 * sigma ** 2 * sigmas ** 2) ** 0.5)
        return f

    sigma0 = (sigmas ** 2).dot(weights) ** 0.5
    sigma = fsolve(func, sigma0, xtol=1e-15)
    return sigma


if __name__ == "__main__":
    seed = 42
    rng = np.random.RandomState(seed)
    n_hists = 2
    masses = np.ones(n_hists)
    n_features = 500
    epsilons = np.array([50, 500, 5000]) / n_features
    weights = np.ones(n_hists)
    weights = np.array([3., 7.])
    weights /= weights.sum()
    grid = np.linspace(-5, 5, n_features)
    loc = np.array([-2., 2.])
    std = np.array([0.4, 0.7])
    P = gengaussians(grid, n_hists, loc=loc, scale=std) + 1e-5
    M = (grid[:, None] - grid[None, :]) ** 2

    Pt = torch.tensor(P)
    weights_t = torch.tensor(weights)
    bars_lebesgue = []
    bars_lebesgue_theo = []

    bars_debiased = []
    bars_debiased_theo = []

    mu_bar = weights.dot(loc)

    tol = 0.001 / n_features

    for epsilon in epsilons:
        K = np.exp(- M / epsilon)
        Kt = torch.tensor(K)
        bar = barycenter(Pt, Kt, reference="uniform", weights=weights_t,
                         tol=tol)
        bars_lebesgue.append(bar)
        bar_debiased = barycenter(Pt, Kt, reference="debiased",
                                  weights=weights_t, tol=tol)
        bars_debiased.append(bar_debiased)

        sigma_l = sigma_lebesgue(std, weights, epsilon)
        sigma_d = sigma_debiased(std, weights, epsilon)

        bar_lebesgue_theo = gengaussians(grid, n_hists=1, loc=[mu_bar],
                                         scale=[sigma_l]).flatten()
        bar_debiased_theo = gengaussians(grid, n_hists=1, loc=[mu_bar],
                                         scale=[sigma_d]).flatten()
        bars_lebesgue_theo.append(bar_lebesgue_theo)
        bars_debiased_theo.append(bar_debiased_theo)

    colors = ["mediumaquamarine", "indianred"]
    styles = ["-", "dashed"]

    names_l = [r"$\alpha_{OT^{\mathcal{U}}_{\varepsilon}}$ (IBP)",
               r"Expected by Theorem 1"]
    names_d = [r"$\alpha_{S_{\varepsilon}}$ (proposed)",
               r"Expected by Theorem 3"]
    names = [names_l, names_d]
    empirical = [bars_lebesgue, bars_debiased]
    theoretical = [bars_lebesgue_theo, bars_debiased_theo]
    titles = ["blur", "div"]
    for name, bar_emp, bar_theo, title in zip(names, empirical, theoretical,
                                              titles):
        legend = [Line2D([0], [0], color=color,
                  label=nn, ls=ls, lw=3)
                  for color, nn, ls in zip(colors, name, styles)]

        f, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
        for ax, bar, bar_t, eps in zip(axes.ravel(), bar_emp,
                                       bar_theo, epsilons):
            ax.plot(grid, P, color="k", alpha=0.7)
            ax.plot(grid, bar, color=colors[0], lw=4, ls=styles[0])
            ax.plot(grid, bar_t, color=colors[1], lw=3, ls=styles[1])
            eps = np.round(eps, 2)
            ax.set_title(r"$\varepsilon$ = %s" % eps)

        plt.savefig("fig/gaussians-%s.pdf" % title, tight_layout=True)
        plt.figure(figsize=(10, 1))
        plt.axis("off")
        plt.legend(handles=legend, ncol=3)
        plt.savefig("fig/gaussians-%s-legend.pdf" % title, tight_layout=True)
