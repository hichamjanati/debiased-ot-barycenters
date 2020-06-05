import numpy as np
from time import time
import torch

from sinkhorn_barycenters import barycenter_3d


def rescale_data(mesh, scale):
    mesh.points -= np.array(mesh.center)[None, :]
    diameter = 0.5 * (mesh.points.max() - mesh.points.min())
    mesh.points *= scale / diameter
    return mesh


if __name__ == "__main__":
    import pickle
    import pyvista as pv
    from pyvista import examples
    torch.set_default_dtype(torch.float32)
    device = "cpu"
    if torch.cuda.device_count():
        device = "cuda:0"

    beta = pv.ParametricTorus()
    beta = pv.PolyData(beta)

    alpha = examples.download_bunny()
    # rotate the bunny
    alpha.rotate_x(100)
    alpha.rotate_z(140)
    alpha.rotate_y(-20)
    alpha = alpha.smooth(100, relaxation_factor=0.1)
    beta = beta.smooth(100, relaxation_factor=0.1)

    alpha = rescale_data(alpha, 0.95)
    beta = rescale_data(beta, 0.95)
    beta.rotate_y(90)

    width = 200
    n_features = width ** 3

    hist_grid = torch.linspace(-1., 1., width + 1)
    grid = torch.linspace(-1., 1., width)
    X, Y, Z = torch.meshgrid(grid, grid, grid)
    alpha_hist = np.histogramdd(alpha.points,
                                bins=[hist_grid, hist_grid, hist_grid])[0]
    beta_hist = np.histogramdd(beta.points,
                               bins=[hist_grid, hist_grid, hist_grid])[0]

    alpha_hist /= alpha_hist.sum()
    beta_hist /= beta_hist.sum()
    epsilon = 0.01
    M = (grid[:, None] - grid[None, :]) ** 2
    K = torch.exp(- M / epsilon)
    K = K.to(device)
    hists = np.stack((alpha_hist, beta_hist))
    hists += 1e-10
    hists /= hists.sum(axis=(1, 2, 3))[:, None, None, None]
    hists = torch.tensor(hists).type(torch.float32)
    hists = hists.to(device)
    ws = [0., 0.25,  0.5, 0.75,  1.]
    data = dict(ibp=dict(times=[], bars=[]),
                deb=dict(times=[], bars=[]))
    bars = []
    for ii, w in enumerate(ws):
        print("->>> Doing weight {} ... ".format(ii + 1))
        weights = torch.tensor([1. - w, w], device=device)
        t0 = time()
        bar_deb = barycenter_3d(hists, K, weights=weights, debiased=True)
        t1 = time()
        print("Debiased done in ", t1 - t0)
        bar_ibp = barycenter_3d(hists, K, weights=weights, debiased=False)
        t2 = time()
        print("IBP done in ", t2 - t1)

        data["deb"]["times"].append(t1 - t0)
        data["ibp"]["times"].append(t2 - t1)
        data["deb"]["bars"].append(bar_deb.cpu())
        data["ibp"]["bars"].append(bar_ibp.cpu())

    with open("data/interpolation-data.pkl", "wb") as ff:
        pickle.dump(data, ff)
