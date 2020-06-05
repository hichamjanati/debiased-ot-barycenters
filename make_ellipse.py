import numpy as np

from matplotlib import pyplot as plt


def make_ellipse(width=100, mean=None, semimaj=0.3,
                 semimin=0.1, phi=np.pi / 3):
    """
    Generate ellipse.
    The function creates a 2D ellipse in polar coordinates then transforms
    to cartesian coordinates.

    semi_maj : float
        length of semimajor axis (always taken to be some phi (-90<phi<90 deg)
        from positive x-axis!)

    semi_min : float
        length of semiminor axis

    phi : float
        angle in radians of semimajor axis above positive x axis

    mean : array,
        coordinates of the center.

    n_samples : int
        Number of points to sample along ellipse from 0-2pi

    """
    if mean is None:
        mean = [width // 2, width // 2]
    semimaj *= width
    semimin *= width
    mean = np.asarray(mean)
    # Generate data for ellipse structure
    n_samples = 1e6
    theta = np.linspace(0, 2 * np.pi, n_samples)
    r = 1 / np.sqrt((np.cos(theta)) ** 2 + (np.sin(theta)) ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    data = np.array([x, y])
    S = np.array([[semimaj, 0], [0, semimin]])
    R = np.array([[np.cos(phi), -np.sin(phi)],
                 [np.sin(phi), np.cos(phi)]])
    T = np.dot(R, S)
    data = np.dot(T, data)
    data += mean[:, None]
    data = np.round(data).astype(int)
    data = np.clip(data, 0, width - 1)
    return data


def make_nested_ellipses(width, n_ellipses=1, centers=None, seed=None,
                         max_radius=0.3, smoothing=0.):
    """Creates array of random nested ellipses."""
    rng = np.random.RandomState(seed)
    if smoothing:
        grid = np.arange(width)
        kernel = np.exp(- (grid[:, None] - grid[None, :]) ** 2 / smoothing)
    ellipses = []
    if centers is None:
        centers = np.linspace(width // 3, 2 * width // 3, n_ellipses)
        centers = np.vstack([centers, centers]).T.astype(float)
    for ii in range(n_ellipses):
        mean = centers[ii]
        semimaj = rng.rand() * max_radius + 0.25
        semimin = rng.rand() * 0.15 + 0.05

        phi = rng.rand() * np.pi - np.pi / 2
        x1, y1 = make_ellipse(width, mean, semimaj, semimin, phi)
        mean = mean + rng.rand(2) * semimaj
        semimaj = rng.rand() * 0.05 + 0.1
        semimin = rng.rand() * 0.1 + 0.05
        phi = rng.rand() * np.pi - np.pi / 2
        x2, y2 = make_ellipse(width, mean, semimaj, semimin, phi)

        img = np.zeros((width, width))
        img[x1, y1] = 1.

        img[x2, y2] = 1.

        if smoothing:
            img = kernel.dot(kernel.dot(img).T).T
        ellipses.append(img)
    ellipses = np.array(ellipses)
    return ellipses


if __name__ == "__main__":
    import os

    overleaf_path = "~/Documents/work/phd/overleaf/icml20/icml20/img/"
    overleaf_path = os.path.expanduser(overleaf_path)

    width = 60
    n_ellipses = 10
    ellipses = make_nested_ellipses(width, n_ellipses, seed=42)

    # plot 5 ellipses
    f, axes = plt.subplots(1, 5, figsize=(40, 8))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(ellipses[::2][i], cmap="hot_r")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("fig/ellipses.pdf")
    plt.savefig(overleaf_path + "ellipses.pdf", bbox_inches="tight")

    # plot all 10 ellipses
    f, axes = plt.subplots(1, 10, figsize=(30, 3))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(ellipses[i], cmap="hot_r")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig("fig/all-ellipses.pdf")
    plt.savefig(overleaf_path + "all-ellipses.pdf", bbox_inches="tight")
