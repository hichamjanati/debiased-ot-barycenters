import numpy as np
from time import time
import warnings

from matplotlib import pyplot as plt

from utils import gengaussians


params = {"legend.fontsize": 15,
          "axes.titlesize": 14,
          "axes.labelsize": 14,
          "xtick.labelsize": 10,
          "ytick.labelsize": 10,
          "pdf.fonttype": 42}
plt.rcParams.update(params)


def barycenter_vs_gaussian(P, K, qstar, debiased, maxiter=10000, tol=1e-8,
                           weights=None, return_log=True):
    """Compute IBP / Debiased bayrenter versus ground truth.
    """
    dim, n_hists = P.shape
    b = np.ones_like(P)
    q = np.ones(dim) / dim
    Kb = K.dot(b)
    log = {'err': [], 'time': []}
    err = 10
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    if debiased:
        c = np.ones_like(q)
    t0 = time()
    for ii in range(maxiter):
        a = P / Kb
        Ka = K.T.dot(a)
        if debiased:
            q = c * np.prod(Ka ** weights[None, :], axis=1)
            c = (c * q / K.dot(c)) ** 0.5
        else:
            q = np.prod(Ka ** weights[None, :], axis=1)
        err = abs(q - qstar).max()
        Q = q[:, None]
        b = Q / Ka
        Kb = K.dot(b)
        t = time()
        log["time"].append(t - t0)
        log["err"].append(err)
        if err < tol:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    return q, log


if __name__ == "__main__":
    seed = 42
    rng = np.random.RandomState(seed)
    n_hists = 2
    masses = np.ones(n_hists)
    n_features = 5000
    epsilon = 0.0015
    weights = np.ones(n_hists)
    # weights = np.array([3., 7.])
    weights /= weights.sum()
    grid = np.linspace(-1, 1., n_features)
    loc = np.array([-0.5, 0.5])
    mubar = [weights.dot(loc)]
    std = np.array([0.1, 0.1])
    P = gengaussians(grid, n_hists, loc=loc, scale=std)

    sigma_debiased = std[:1]
    sigma_biased = (std[:1] ** 2 + epsilon * 0.5) ** 0.5
    bar_debiased = gengaussians(grid, 1, loc=mubar,
                                scale=sigma_debiased).flatten()
    bar_biased = gengaussians(grid, 1, loc=mubar,
                              scale=sigma_biased).flatten()

    M = (grid[:, None] - grid[None, :]) ** 2
    K = np.exp(- M / epsilon)
    bar_ibp, ibp = barycenter_vs_gaussian(P, K, bar_biased,
                                          debiased=False)
    bar_deb, deb = barycenter_vs_gaussian(P, K, bar_debiased,
                                          debiased=True)

    colors = ["steelblue", "forestgreen"]

    names = [r"$\alpha_{OT^{\mathcal{U}}_{\varepsilon}}$ (IBP)",
             r"$\alpha_{S_{\varepsilon}}$ (Alg 1)"]

    lim = 1e-8
    f, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.grid("on")
    x_max = 0
    for name, log, col in zip(names, [ibp, deb], colors):
        dist_to_minimizer = np.array(log["err"])
        dist_to_minimizer[dist_to_minimizer < lim] = 0.
        calendar = log["time"]
        t_max = max(calendar)
        if t_max > x_max:
            x_max = t_max
        ax.set_xticks([1, 2, 3, 4, 5, 6, 7])
        ax.plot(calendar, dist_to_minimizer, color=col, lw=2, label=name)
        ax.set_yscale("log")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\|\|\alpha^{\star} - \alpha^{(l)}\|\|_{\infty}$")
    plt.legend(loc="upper right", fontsize=10)
    plt.subplots_adjust(bottom=0.2, left=0.4)
    # plt.show()
    if 1:
        plt.savefig("fig/convergence.pdf")
