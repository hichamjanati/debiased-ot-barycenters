import time

import numpy as np
from scipy.io import savemat, loadmat

import torch

from matplotlib import pyplot as plt

from sinkhorn_barycenters import barycenter

from sharp_barycenter import sharp_barycenter_img
from free_barycenter import barycenter_free, create_distribution_2d
from make_ellipse import make_nested_ellipses


device = "cpu"
# We ran this experiment on CPU
# if torch.cuda.device_count():
#     device = "cuda:0"

seed = 42
n_samples = 10
width = 60
n_features = width ** 2
imgs_np = make_nested_ellipses(width, n_samples, seed=seed)
imgs_np /= imgs_np.sum((1, 2))[:, None, None]
savemat("data/ellipses.mat", dict(ellipses=imgs_np))

imgs = torch.tensor(imgs_np, dtype=torch.float64, device=device,
                    requires_grad=False)
dists = create_distribution_2d(imgs_np)
imgs = imgs + 1e-10
imgs /= imgs.sum((1, 2))[:, None, None]
epsilon = 0.002

grid = torch.arange(width).type(torch.float64)
grid /= width
M = (grid[:, None] - grid[None, :]) ** 2
M_large = M[:, None, :, None] + M[None, :, None, :]
M_large = M_large.reshape(n_features, n_features)
M_large = M_large.to(device)

K = torch.exp(- M / epsilon)
K = K.to(device)

print("Doing IBP ...")
time_ibp = time.time()
bar_ibp = barycenter(imgs, K, reference="uniform")
time_ibp = time.time() - time_ibp

print("Doing Debiased ...")
time_deb = time.time()
bar_deb = barycenter(imgs, K, reference="debiased")
time_deb = time.time() - time_deb

print("Doing product ...")
time_prod = time.time()
bar_prod = barycenter(imgs, K, reference="product")
time_prod = time.time() - time_prod


print("Doing Sharp ...")
time_sharp = time.time()
bar_sharp, log_sharp = sharp_barycenter_img(imgs, K, M_large, epsilon)
time_sharp = time.time() - time_sharp

print("Doing Free ...")
time_free = time.time()
bar_free, log_free = barycenter_free(distributions=dists, grid_step=width,
                                     epsilon=epsilon)
time_free = time.time() - time_free

weights = bar_free.weights.flatten()
support = bar_free.support
x, y = support.T.numpy()


compute_maaipm = False
if compute_maaipm:
    import matlab.engine
    print("Doing MAAIPM ...")
    eng = matlab.engine.start_matlab()
    eng.mellipses(nargout=0)

maaipm_data = loadmat("data/barycenter.mat")
bar_maaipm = maaipm_data["barycenter"]
time_maaipm = maaipm_data["t"][0][0]
bar_maaipm = torch.tensor(bar_maaipm)

titles = [r"$\alpha_{S_{\varepsilon}}$ (proposed)",
          r"$\alpha_{IBP}$",
          r"$\alpha_{{OT^{\otimes}}_{\varepsilon}}$",
          r"$\alpha_{A_{\varepsilon}}$",
          r"$\alpha_{S_{\varepsilon}}^{F}$ (Free support)",
          r"$\alpha_{W}$"]
bars = [bar_deb, bar_ibp, bar_prod, bar_sharp, bar_free, bar_maaipm]
times = [time_deb, time_ibp, time_prod, time_sharp, time_free, time_maaipm]


rc = {"legend.fontsize": 14,
      "axes.titlesize": 23,
      "axes.labelsize": 20,
      "xtick.labelsize": 15,
      "ytick.labelsize": 18,
      "pdf.fonttype": 42}
plt.rcParams.update(rc)

f, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.ravel()):
    time_value = times[i]
    name = titles[i]
    tt = " Ran in %s s" % np.round(time_value, 2)

    if i == 4:
        ax.hist2d(y, x, bins=grid, weights=weights, cmap="hot_r")
        ax.set_xlabel(tt)
    else:
        ax.imshow(bars[i], cmap="hot_r")
        if not torch.isnan(bars[i]).any():
            if i == 0:
                ax.set_xlabel(tt, color="green")
            else:
                ax.set_xlabel(tt)

    ax.set_xticks([])
    ax.set_yticks([])
    if i == 0:
        for spine in ax.spines.values():
            spine.set_edgecolor('green')
            spine.set_linewidth(5)
        ax.set_title(name, color='green')
    else:
        ax.set_title(name)
plt.subplots_adjust(hspace=0.4)
plt.savefig("fig/barycenter-ellipses.pdf", bbox_inches="tight")
