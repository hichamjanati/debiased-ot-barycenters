import torch


def convol_imgs(imgs, K):
    kx = torch.einsum("...ij,kjl->kil", K, imgs)
    kxy = torch.einsum("...ij,klj->kli", K, kx)
    return kxy


def convol_3d(cloud, K):
    kx = torch.einsum("...ij,rjlk->rilk", K, cloud)
    kxy = torch.einsum("...ij,rkjl->rkil", K, kx)
    kxyz = torch.einsum("...ij,rlkj->rlki", K, kxy)
    return kxyz


def convol_old(imgs, K):
    kxy = torch.zeros_like(imgs)
    for i, img in enumerate(imgs):
        kxy[i] = K.mm(K.mm(img).t()).t()
    return kxy


def convol_huge_imgs(imgs, K):
    n, m, dimension, dimension = imgs.shape
    out = convol_imgs(imgs.reshape(n * m, dimension, dimension), K)
    out = out.reshape(n, m, dimension, dimension)
    return out


def convol_huge(imgs, K):
    dimension, n, m = imgs.shape
    out = K.mm(imgs.reshape(dimension, n * m))
    out = out.reshape(dimension, n, m)
    return out


def convol_huge_log(imgs, C):
    dimension, n, m = imgs.shape
    imgs = imgs.reshape(dimension, n * m)
    out = torch.logsumexp(C[:, :, None] + imgs[None, :], dim=1)
    out = out.reshape(dimension, n, m)
    return out


def convol_imgs_log(imgs, C):
    """Compute log separable kernal application."""
    n, dimension, dimension = imgs.shape
    x = (torch.logsumexp(C[None, None, :, :] + imgs[:, :, None], dim=-1))
    x = torch.logsumexp(C.t()[None, :, :, None] + x[:, :, None], dim=1)
    return x.reshape(n, dimension, dimension)


def convol_huge_imgs_log(imgs, C):
    """Compute log separable kernal application."""
    n, m, dimension, dimension = imgs.shape
    imgs = imgs.reshape(n * m, dimension, dimension)
    x = (torch.logsumexp(C[None, None, :, :] + imgs[:, :, None], dim=-1))
    x = torch.logsumexp(C.t()[None, :, :, None] + x[:, :, None], dim=1)
    return x.reshape(n, m, dimension, dimension)
