import numpy as np
import torch

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


from sinkhorn_barycenters import barycenter
from utils import gengaussians


params = {"legend.fontsize": 18,
          "axes.titlesize": 16,
          "axes.labelsize": 16,
          "xtick.labelsize": 13,
          "ytick.labelsize": 13,
          "pdf.fonttype": 42}
plt.rcParams.update(params)


if __name__ == "__main__":
    seed = 42
    rng = np.random.RandomState(seed)
    n_hists = 2
    masses = np.ones(n_hists)
    n_features = 500
    epsilons = np.array([100, 200, 500]) / n_features
    grid = np.linspace(-5, 5, n_features)
    loc = np.array([-3, 3])
    std = np.array([0.4, 0.4])
    P = gengaussians(grid, n_hists, loc=loc, scale=std) + 1e-10
    M = (grid[:, None] - grid[None, :]) ** 2

    Pt = torch.tensor(P)
    bars = []
    bars_div = []
    bars_prod = []
    tol = 0.001 / n_features
    for epsilon in epsilons:
        K = np.exp(- M / epsilon)
        Kt = torch.tensor(K)
        bar = barycenter(Pt, Kt, reference="uniform", tol=tol)
        bar_deb = barycenter(Pt, Kt, reference="debiased", tol=tol)
        bar_prod = barycenter(Pt, Kt, reference="product", tol=tol)
        bars.append(bar)
        bars_div.append(bar_deb)
        bars_prod.append(bar_prod)

    colors = ["darkblue", "salmon", "mediumaquamarine"]
    names = [r"$\alpha_{OT^{\mathcal{U}}_{\varepsilon}}$ (IBP)",
             r"$\alpha_{OT^{\otimes}_{\varepsilon}}$",
             r"$\alpha_{S_{\varepsilon}}$ (proposed)"]
    styles = ["-", "-", "-"]
    legend = [Line2D([0], [0], color=color,
              label=name, linewidth=2, ls=ls)
              for color, name, ls in zip(colors, names, styles)]

    f, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    for ax, bar, bar_prod, bar_div, eps in zip(axes.ravel(), bars,
                                               bars_prod, bars_div, epsilons):
        ax.plot(grid, P, color="k", alpha=0.7)
        ax.plot(grid, bar_prod, color="salmon", lw=2, ls=styles[1])
        ax.plot(grid, bar, color="darkblue", lw=2, ls=styles[0])
        ax.plot(grid, bar_div, color="mediumaquamarine", lw=3, ls=styles[2])

        eps = np.round(eps, 2)
        ax.set_title(r"$\varepsilon$ = %s" % eps)
    ax.set_ylim([0., 0.04])
    plt.savefig("fig/gaussians.pdf", tight_layout=True)

    plt.figure(figsize=(10, 1))
    plt.axis("off")
    plt.legend(handles=legend, ncol=3)
    # plt.show()
    plt.savefig("fig/gaussians-legend.pdf", tight_layout=True)
