import warnings
import numpy as np

import torch

from convol import convol_imgs


def convol_huge_imgs(imgs, K):
    n, m, dimension, dimension = imgs.shape
    out = convol_imgs(imgs.reshape(n * m, dimension, dimension), K)
    out = out.reshape(n, m, dimension, dimension)
    return out


def barycenter_wbc(P, K, logweights, Kb=None, c=None, debiased=False,
                   maxiter=1000, tol=1e-4):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    n_hists, width, _ = P.shape
    if Kb is None:
        b = torch.ones_like(P)[None, :]
        Kb = convol_huge_imgs(b, K)
    if c is None:
        c = torch.ones(1, width, width, device=P.device)
    q = c.clone()
    logweights.requires_grad = True
    err = 1
    weights = torch.softmax(logweights, dim=1)[:, :, None, None]
    for ii in range(maxiter):
        with torch.no_grad():
            qold = q.detach().clone()
        a = P[None, :] / Kb
        Ka = convol_huge_imgs(a, K.t())
        q = c * torch.prod((Ka) ** weights, dim=1)
        if debiased:
            Kc = convol_imgs(c, K.t())
            c = (c * q / Kc) ** 0.5
        Q = q[:, None, :, :]
        b = Q / Ka
        Kb = convol_huge_imgs(b, K)
        if torch.isnan(q).any():
            warnings.warn("Numerical Errors ! Stopped early in debiased = %s" % debiased)

            break
        with torch.no_grad():
            err = abs(q - qold).max()
            if err < tol and ii > 5:
                break

    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    return q


def projection(dataset, Q, K, maxiter_adam=1000, tol=1e-5, **args):
    """Perform OT barycentric projection of q over a dataset."""
    n_hists = dataset.shape[0]
    n_projections = len(Q)
    logw = torch.zeros(n_projections, n_hists, device=dataset.device)
    opt = torch.optim.Adam([logw], lr=0.1, betas=(0.9, 0.9999))
    loss = 1e10
    for ii in range(maxiter_adam):
        with torch.no_grad():
            oldloss = loss
        bar = barycenter_wbc(dataset, K, logw, **args)
        if torch.isnan(bar).any():
            warnings.warn("Numerical Errors ! Stopped early.")
            break
        loss = ((bar - Q) ** 2).sum()
        with torch.no_grad():
            opt.zero_grad()
            loss.backward()
            opt.step()
            err = abs(loss - oldloss).item() / max(loss, oldloss, 1.)
            if err < tol:
                break

    if ii == maxiter_adam - 1:
        warnings.warn("Projection did not converge ! err = {} ***".format(err))
    weights = torch.softmax(logw, dim=1).detach().cpu().numpy()
    return weights


def encode_dataset(dataset, train_ratio=0.5, verbose=True, batch_size=100,
                   **args):
    """Encode dataset using WBC."""
    n_samples = len(dataset)
    n_train = int(train_ratio * n_samples)
    training_set = dataset[:n_train]
    to_encode = dataset[n_train:]
    n_projections = len(to_encode)
    n_batches = int(np.ceil(n_projections / batch_size))
    weights = []
    for ii in range(n_batches):
        start = ii * batch_size
        end = start + batch_size
        weights_batch = projection(training_set, to_encode[start:end], **args)
        weights.append(weights_batch)
    weights = np.concatenate(weights)
    return weights
