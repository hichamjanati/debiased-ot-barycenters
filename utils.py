import numpy as np

from scipy.stats import norm as gaussian


def gengaussians(grid, n_hists, loc=None, scale=None, normed=True,
                 mass=None):
    """Generate random gaussian histograms.

    Parameters
    ----------
    n_features: int.
        dimensionality of histograms.
    n_hists: int.
        number of histograms.
    loc: array-like (n_hists,)
        Gaussian means vector, zeros by default.
    scale: array-like (n_hists,)
        Gaussian std vector, ones by default.
    normed: boolean.
        if True, each measure is normalized to sum to 1.
    xlim: tuple of floats.
        delimiters of the grid on which the gaussian density is computed.
    mass: array-like (n_hists,)
        positive mass of each measure (ones by default)
        overruled if `normed` is True.
    seed: int.
        random state.

    Returns
    -------
    array-like (n_features, n_hists).
fr
    """
    if loc is None:
        loc = np.zeros(n_hists)
    if scale is None:
        scale = np.ones(n_hists)
    if mass is None:
        mass = np.ones(n_hists)

    coefs = np.empty((len(grid), n_hists))
    for i, (l, s) in enumerate(zip(loc, scale)):
        coefs[:, i] = gaussian.pdf(grid, loc=l, scale=s)

    if normed:
        coefs /= coefs.sum(axis=0)
    coefs *= mass

    return coefs
