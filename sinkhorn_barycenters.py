# -*- coding: utf-8 -*-
"""
Debiased Sinkhorn barycenters.
"""
#
# License: MIT License

import torch
import numpy as np

import warnings


def convol_imgs(imgs, K):
    kx = torch.einsum("...ij,kjl->kil", K, imgs)
    kxy = torch.einsum("...ij,klj->kli", K, kx)
    return kxy


def convol_3d(cloud, K):
    kx = torch.einsum("ij,rjlk->rilk", K, cloud)
    kxy = torch.einsum("ij,rkjl->rkil", K, kx)
    kxyz = torch.einsum("ij,rlkj->rlki", K, kxy)
    return kxyz


def barycenter_3d(P, K, Kb=None, c=None, maxiter=1000, tol=1e-7,
                  debiased=False, weights=None, return_log=False):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    n_hists, width, _, _ = P.shape
    b = torch.ones_like(P, requires_grad=False)
    q = torch.ones((width, width, width), device=P.device, dtype=P.dtype)
    if Kb is None:
        Kb = convol_3d(b, K)
    if c is None:
        c = q.clone()
    log = {'err': [], 'a': [], 'b': [], 'q': []}
    err = 10
    if weights is None:
        weights = torch.ones(n_hists, device=P.device, dtype=P.dtype) / n_hists
    for ii in range(maxiter):
        if torch.isnan(q).any():
            break
        qold = q.clone()
        a = P / Kb
        Ka = convol_3d(a, K.t())
        q = c * torch.prod((Ka) ** weights[:, None, None, None], dim=0)
        if debiased:
            Kc = convol_3d(c[None, :], K).squeeze()
            c = (c * q / Kc) ** 0.5
        Q = q[None, :]
        b = Q / Ka
        Kb = convol_3d(b, K)
        err = abs(q - qold).max()

        if err < tol and ii > 10:
            break
    print("Barycenter 3d | err = ", err)
    if return_log:
        log["err"].append(err)
        log["a"] = a
        log["q"] = q
        log["b"] = b

    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    if return_log:
        return q, log
    return q


def barycenter_debiased_1d(P, K, maxiter=5000, tol=1e-5,
                           weights=None, return_log=False):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    dim, n_hists = P.shape
    bold = torch.ones_like(P, device=P.device)
    b = bold.clone()
    c, q = torch.ones((2, dim), dtype=P.dtype, device=P.device)
    Kb = K.mm(b)
    log = {'err': [], 'a': [], 'b': [], 'c': [], 'q': []}
    err = 10
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=P.device) / n_hists
    for ii in range(maxiter):
        qold = q.clone()
        a = P / Kb
        Ka = K.t().mm(a)
        q = c * torch.prod((Ka) ** weights[None, :], dim=1)
        c = (c * q / K.mv(c)) ** 0.5
        Q = q[:, None]
        b = Q / Ka
        Kb = K.mm(b)
        # err = abs(a * Kb - P).mean()
        err = abs(q - qold).max()
        if return_log:
            log["err"].append(err)
            log["a"].append(a)
            log["q"].append(q)
            log["c"].append(c)
            log["b"].append(b)
        if err < tol and ii > 10:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    if return_log:
        return q, log
    return q


def ot_diag_2d(q, K, maxiter=100, tol=1e-5):
    """Computes Auto-correlation potential for 2d distributions."""
    c = torch.ones_like(q)
    for ii in range(maxiter):
        Kc = K.t().mm(K.mm(c).t()).t()
        c_new = (c * q / Kc) ** 0.5
        err = abs(c - c_new).max()
        err /= max(c.max(), c_new.max(), 1.)
        c = c_new.clone()
        if err < tol and ii > 3:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Auto-correlation potential "
                      "did not converge ! err = {} ***".format(err))

    return c


def ot_diag_1d(c, q, K, maxiter=100, tol=1e-5):
    """Computes Auto-correlation potential for 2d distributions."""
    c = torch.ones_like(q)
    for ii in range(maxiter):
        Kc = K.mv(c)
        c_new = (c * q / Kc) ** 0.5
        err = abs(c - c_new).max()
        err /= max(c.max(), c_new.max(), 1.)
        c = c_new.clone()
        if err < tol and ii > 3:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Auto-correlation potential "
                      "did not converge ! err = {} ***".format(err))

    return c


def ot_diag_2d_np(c, q, K, maxiter=100, tol=1e-5):
    """Computes Auto-correlation potential for 2d distributions."""
    for ii in range(maxiter):
        Kc = K.T.dot(K.dot(c).T).T
        c_new = (c * q / Kc) ** 0.5
        err = abs(c - c_new).max()
        err /= max(c.max(), c_new.max(), 1.)
        c = c_new.copy()
        if err < tol and ii > 3:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Auto-correlation potential "
                      "did not converge ! err = {} ***".format(err))

    return c


def ot_diag_1d_np(q, K, maxiter=100, tol=1e-5):
    """Computes Auto-correlation potential for 2d distributions."""
    c = np.ones_like(q)
    for ii in range(maxiter):
        c_new = (c * q / K.dot(c)) ** 0.5
        err = abs(c - c_new).max()
        err /= max(c.max(), c_new.max(), 1.)
        c = c_new.copy()
        if err < tol and ii > 3:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Auto-correlation potential "
                      "did not converge ! err = {} ***".format(err))

    return c


def barycenter_1d(P, K, maxiter=5000, tol=1e-5,
                  weights=None, return_log=False):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    dim, n_hists = P.shape
    b = torch.ones_like(P, device=P.device)
    q = torch.ones(dim, dtype=P.dtype, device=P.device)
    Kb = K.mm(b)
    err = 1
    log = {'err': [err], 'a': [], 'b': [], 'c': [], 'q': []}

    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=P.device) / n_hists
    for ii in range(maxiter):
        qold = q.clone()
        a = P / Kb
        Ka = K.t().mm(a)
        q = torch.prod((b * Ka) ** weights[None, :], dim=1)
        Q = q[:, None]
        b = Q / Ka
        Kb = K.mm(b)
        # err = abs(a * Kb - P).mean()
        err = abs(q - qold).max()

        if err < tol and ii > 10:
            break

        if return_log:
            log["err"].append(err)
            log["a"].append(a)
            log["q"].append(q)
            log["b"].append(b)

    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    if return_log:
        return q, log
    return q


def barycenter_debiased_2d(P, K, Kb=None, c=None, maxiter=5000, tol=1e-5,
                           weights=None, return_log=False):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    n_hists, width, _ = P.shape
    b = torch.ones_like(P, requires_grad=False)
    q = torch.ones((width, width), dtype=P.dtype, device=P.device)
    if Kb is None:
        Kb = convol_imgs(b, K)
    if c is None:
        c = q.clone()
    log = {'err': [], 'a': [], 'b': [], 'q': []}
    err = 10
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=P.device) / n_hists
    for ii in range(maxiter):
        qold = q.clone()
        a = P / Kb
        Ka = convol_imgs(a, K.t())
        q = c * torch.prod((Ka) ** weights[:, None, None], dim=0)
        for kk in range(10):
            Kc = K.t().mm(K.mm(c).t()).t()
            c = (c * q / Kc) ** 0.5
        Q = q[None, :, :]
        b = Q / Ka
        Kb = convol_imgs(b, K)
        # err = abs(a * Kb - P).mean()
        err = abs(q - qold).max()

        if err < tol and ii > 10:
            break
    if return_log:
        log["err"].append(err)
        log["a"] = a
        log["q"] = q
        log["b"] = b

    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    if return_log:
        return q, log
    return q


def barycenter_2d(P, K, Kb=None, maxiter=5000, tol=1e-5,
                  weights=None, return_log=False):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    n_hists, width, _ = P.shape
    b = torch.ones_like(P, requires_grad=False)
    q = torch.ones((width, width), dtype=P.dtype, device=P.device)
    if Kb is None:
        Kb = convol_imgs(b, K)
    log = {'err': [], 'a': [], 'b': [], 'q': []}
    err = 10
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=P.device) / n_hists
    for ii in range(maxiter):
        qold = q.clone()
        a = P / Kb
        Ka = convol_imgs(a, K.t())
        q = torch.prod((b * Ka) ** weights[:, None, None], dim=0)
        Q = q[None, :, :]
        b = Q / Ka
        Kb = convol_imgs(b, K)
        err = abs(q - qold).max()

        if err < tol and ii > 10:
            break
    if return_log:
        log["err"].append(err)
        log["a"] = a
        log["q"] = q
        log["b"] = b

    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    if return_log:
        return q, log
    return q


def barycenter(P, K, reference="debiased", **kwargs):
    """Compute OT barycenter."""
    ndim = P.ndimension()
    if ndim > 3 or ndim <= 1:
        raise ValueError("Data dimension must be 2 for 1d distributions"
                         " or 3 for 2d distributions.")
    if reference == "debiased":
        if ndim == 2:
            func = barycenter_debiased_1d
        elif ndim == 3:
            func = barycenter_debiased_2d

    elif reference == "uniform":
        if ndim == 2:
            func = barycenter_1d
        elif ndim == 3:
            func = barycenter_2d

    elif reference == "product":
        if ndim == 2:
            func = barycenter_ref_1d
        else:
            func = barycenter_ref_2d

    return func(P, K, **kwargs)


def barycenter_np_debiased_1d(P, K, maxiter=5000, tol=1e-5,
                              weights=None, return_log=True):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    dim, n_hists = P.shape
    bold = np.ones_like(P)
    b = bold.copy()
    q, c = np.ones((2, dim))
    Kb = K.dot(b)
    log = {'err': [], 'a': [], 'b': [], 'c': [], 'q': []}
    err = 10
    if weights is None:
        weights = np.ones(n_hists) / n_hists

    for ii in range(maxiter):
        qold = q.copy()
        a = P / Kb
        Ka = K.T.dot(a)
        q = c * np.prod(Ka ** weights[None, :], axis=1)
        Q = q[:, None]
        b = Q / Ka
        Kb = K.dot(b)
        c = (c * q / K.dot(c)) ** 0.5
        err = abs(q - qold).max()
        if return_log:
            log["err"].append(err)
            log["a"].append(a)
            log["q"].append(q)
            log["c"].append(c)
            log["b"].append(b)
        if err < tol and ii > 10:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    if return_log:
        return q, log
    return q


def barycenter_np_1d(P, K, maxiter=5000, tol=1e-5,
                     weights=None, return_log=True):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    dim, n_hists = P.shape
    bold = np.ones_like(P)
    b = bold.copy()
    q = np.ones(dim)
    Kb = K.dot(b)
    log = {'err': [], 'a': [], 'b': [], 'c': [], 'q': []}
    err = 10
    if weights is None:
        weights = np.ones(n_hists) / n_hists

    for ii in range(maxiter):
        qold = q.copy()
        a = P / Kb
        Ka = K.T.dot(a)
        q = np.prod((b * Ka) ** weights[None, :], axis=1)
        Q = q[:, None]
        # err = abs(Ka * b).std(axis=1).mean()
        b = Q / Ka
        Kb = K.dot(b)
        err = abs(q - qold).max()

        if return_log:
            log["err"].append(err)
            log["a"].append(a)
            log["q"].append(q)
            log["b"].append(b)
        if err < tol and ii > 10:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    if return_log:
        return q, log
    return q


def _barycenter_inner_1d_np(P, K, qold=None, bold=None, maxiter=1000,
                            tol=1e-5, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    dim, n_hists = P.shape
    if bold is None:
        bold = np.ones_like(P)
    b = bold.copy()
    if qold is None:
        qold = np.ones(dim) / dim
    Kb = K.dot(b)
    err = 10
    if weights is None:
        weights = np.ones(n_hists) / n_hists
    for ii in range(maxiter):
        a = P / Kb
        Ka = K.T.dot(a)
        q = qold * np.prod((Ka) ** weights[None, :], axis=1)
        Q = q[:, None]
        err = abs(Ka * b).std(axis=1).mean()
        b = Q / Ka
        Kb = K.dot(b)

        if err < tol and ii > 10:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    return q, b


def _barycenter_inner_1d(P, K, qold=None, bold=None, maxiter=1000,
                         tol=1e-4, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    dim, n_hists = P.shape
    if bold is None:
        bold = torch.ones_like(P)
    b = bold.clone()
    if qold is None:
        qold = torch.ones(dim) / dim
    Kb = K.mm(b)
    err = 10
    if weights is None:
        weights = torch.ones(n_hists) / n_hists
    q = qold.clone()
    for ii in range(maxiter):
        qold_inner = q.clone()
        a = P / Kb
        Ka = K.t().mm(a)
        q = qold * torch.prod((Ka) ** weights[None, :], dim=1)
        Q = q[:, None]
        b = Q / Ka
        Kb = K.mm(b)
        err = abs(q - qold_inner).max()
        if err < tol and ii > 10:
            break
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    return q, b


def _barycenter_inner_2d(P, K, qold=None, bold=None, maxiter=1000,
                         tol=1e-4, weights=None):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    n_hists, width, _ = P.shape
    if bold is None:
        bold = torch.ones_like(P, requires_grad=False)
    b = bold.clone()
    Kb = convol_imgs(b, K)
    if weights is None:
        weights = torch.ones(n_hists, dtype=P.dtype, device=P.device) / n_hists
    if qold is None:
        qold = torch.ones_like(P[0]) / (width ** 2)
    q = qold.clone()
    for ii in range(maxiter):
        qlocal = q.clone()
        a = P / Kb
        Ka = convol_imgs(a, K.t())
        q = qold * torch.prod(Ka ** weights[:, None, None], dim=0)
        Q = q[None, :, :]
        b = Q / Ka
        Kb = convol_imgs(b, K)
        err = abs(q - qlocal).max()
        if err < tol and ii > 10:
            break

    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    return q, b


def barycenter_ref_1d(P, K, maxiter=500, tol=1e-5,
                      weights=None, return_log=True):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    dim, n_hists = P.shape
    q = torch.ones(dim) / dim
    b = torch.ones_like(P)
    for ii in range(maxiter):
        qold = q.clone()
        q, b = _barycenter_inner_1d(P, K, qold=q, bold=b)
        err = abs(q - qold).max()
        if err < tol:
            break
    return q


def barycenter_ref_2d(P, K, maxiter=500, tol=1e-5,
                      weights=None, return_log=True):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    n_hists, width, _ = P.shape
    q = torch.ones((width, width), device=P.device, dtype=P.dtype)
    b = torch.ones_like(P)
    for ii in range(maxiter):
        qold = q.clone()
        q, b = _barycenter_inner_2d(P, K, qold=q, bold=b)
        err = abs(q - qold).max()
        if err < tol:
            break
    return q


def barycenter_ref_1d_np(P, K, maxiter=500, tol=1e-5,
                         weights=None, return_log=True):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    dim, n_hists = P.shape
    q = np.ones(dim) / dim
    b = np.ones_like(P)
    for ii in range(maxiter):
        qold = q.copy()
        q, b = _barycenter_inner_1d_np(P, K, qold=q, bold=b)
        err = abs(q - qold).max()
        if err < tol:
            break
    return q


def barycenter_np(P, K, debiased=True, **kwargs):
    if debiased:
        func = barycenter_np_debiased_1d
    else:
        func = barycenter_np_1d
    return func(P, K, **kwargs)
