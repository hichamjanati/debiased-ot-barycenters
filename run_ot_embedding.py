import pickle
from time import time

import torch
import numpy as np
from wbc import encode_dataset
from torchvision.datasets import MNIST

torch.set_default_dtype(torch.float32)

device = "cpu"
if torch.cuda.device_count():
    device = "cuda:0"

torch.manual_seed(42)
rng = np.random.RandomState(42)
mnist = MNIST("data/", download=True)
imgs = mnist.data.type(torch.float32)
targets = mnist.targets
n_classes = torch.unique(targets).size()[0]
n_samples, width, _ = imgs.shape
n_features = width ** 2
classes = [0, 1, 2, 3, 4]

n_samples = 500

selection = torch.where(targets <= classes[-1])[0][:n_samples]
targets = targets[selection]
imgs = imgs[selection] + 1e-6
imgs /= imgs.sum(dim=(1, 2))[:, None, None]
imgs = imgs.to(device)
n_permutations = 20
train_ratio = 0.1
batch_size = 100
epsilons = [0.2, 0.16, 0.14, 0.12, 0.1, 0.08]
grid = torch.linspace(0., 1., width)
M = (grid[:, None] - grid[None, :]) ** 2

n_train = int(train_ratio * n_samples)

results = dict(epsilons=epsilons, train_ratio=train_ratio,
               n_permutations=n_permutations, n_samples=n_samples)

for epsilon in epsilons:
    t = time()
    print(">>>> Doing epsilon = ", epsilon)
    K = torch.exp(- M / epsilon)
    K = K.to(device)
    results[epsilon] = dict(ibp=[], deb=[], targets=[])
    for kk in range(n_permutations):
        t0 = time()
        print("- shuffle {} / {}".format(kk + 1, n_permutations))
        ints = np.arange(n_samples)
        permutation = np.random.permutation(ints)
        permutation = torch.tensor(permutation)
        targets_shuffled = targets[permutation][n_train:]
        imgs_shuffled = imgs[permutation]
        w_deb = encode_dataset(imgs_shuffled, train_ratio=train_ratio, K=K,
                               debiased=True, batch_size=batch_size)
        w_ibp = encode_dataset(imgs_shuffled, train_ratio=train_ratio, K=K,
                               debiased=False, batch_size=batch_size)
        results[epsilon]["ibp"].append(w_ibp)
        results[epsilon]["deb"].append(w_deb)
        results[epsilon]["targets"].append(targets_shuffled)
        t0 = time() - t0
        print("Time for inner loop:", t0)
    t = time() - t
    print("###### Time for outer loop:", t)

with open("data/wbc-mnist.pkl", "wb") as ff:
    pickle.dump(results, ff)
