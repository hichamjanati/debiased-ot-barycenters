import torch
import warnings

from convol import convol_imgs


def get_potentials_img(P, q, K, b=None, exact_grad=False, maxiter=1000,
                       tol=1e-6):
    """Compute the Wasserstein divergence barycenter between histograms.
    """
    n_hists, width, _ = P.shape
    if b is None:
        b = torch.ones_like(P, dtype=P.dtype, device=P.device)
    Kb = convol_imgs(b, K)

    err = 10
    if not exact_grad:
        q.requires_grad = True
    for ii in range(maxiter):
        a = P / Kb
        Ka = convol_imgs(a, K.t())
        b = q / Ka
        Kb = convol_imgs(b, K)
        err = abs(a * Kb - P).mean()
        if err < tol and ii > 10:
            break

    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))

    return a, b


def sharp_barycenter_img(P, K, M_large, epsilon, init=None, t0=1.,
                         maxiter=2000, tol=1e-6, n_ls=20):
    """Compute Sharp barycenter using AGD."""
    n_hists, width, _ = P.shape
    K_large = torch.exp(- M_large / epsilon)
    KM = K_large * M_large
    if init is None:
        q = torch.ones((1, width, width), dtype=K.dtype, device=P.device)
    else:
        q = init.clone().reshape(1, width, width)
    b = torch.ones_like(P, dtype=P.dtype, device=P.device, requires_grad=True)
    bold = b.clone()
    log = dict(bars=[q], err=[], success=False, loss=[], loss_=[])
    q /= q.sum()
    qhat = q.clone()
    q.requires_grad = True
    grad = torch.zeros_like(q)
    loss = 1e10
    t = t0
    beta = 1
    for ii in range(maxiter):
        beta = (t + 1) / 2
        with torch.no_grad():
            lossold = loss
            bold = b.reshape(n_hists, width, width).clone()
            qold = qhat.clone()
        for jj in range(n_ls):
            with torch.no_grad():
                if ii:
                    qtild = qold * torch.exp(- t0 * beta * grad)
                    qtild = qtild / qtild.sum()
                    qhat = (1 - 1 / beta) * qold + qtild / beta
                    q = (1 - 1 / beta) * qold + qtild / beta
            a, b = get_potentials_img(P, q, K, bold)
            if torch.isnan(b).any():
                print("Numerical Errors !", ii)
                nan = torch.ones((width, width), dtype=P.dtype) * float('nan')
                return nan, log

            a = a.reshape(n_hists, -1)
            b = b.reshape(n_hists, -1)

            loss = torch.einsum("ki,ij,kj->k", a, KM, b).sum()

            loss = loss / n_hists
            b = b.reshape(n_hists, width, width)
            log["loss"].append(loss)
            loss.backward()
            grad = q.grad.clone()
            q.grad.zero_()
            if loss <= lossold + tol:
                break
            else:
                t = t / 2
        if jj == n_ls - 1:
            print("ABNORMAL TERMINATION IN LINESEARCH")
            break
        err = abs(qhat - qold).max()
        log["err"].append(err)
        if err < tol and ii > 10:
            break
    log["a"] = a
    log["b"] = b
    if ii == maxiter - 1:
        warnings.warn("*** Maxiter reached ! err = {} ***".format(err))
    qhat = qhat.cpu().detach().squeeze()
    log["success"] = True
    return qhat, log
