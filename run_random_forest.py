import numpy as np
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


with open("data/wbc-mnist.pkl", "rb") as ff:
    results = pickle.load(ff)
epsilons = results["epsilons"]
n_samples = results["n_samples"]
ratio = results["train_ratio"]
n_train = int(n_samples * ratio)
cv = StratifiedKFold(10)

n_permutations = len(results[epsilons[0]]["ibp"])
scores_deb, scores_ibp = np.zeros((2, len(epsilons), n_permutations))
rf = RandomForestClassifier(100, random_state=0)
for jj, eps in enumerate(epsilons):
    print("Doing eps ... {} / {}".format(jj + 1, len(epsilons)))
    res_esp = results[eps]
    zip_arg = zip(res_esp["ibp"], res_esp["deb"], res_esp["targets"])
    for kk, (data_ibp, data_deb, labels) in enumerate(zip_arg):
        labels = labels.numpy()
        scores_ibp[jj, kk] = cross_val_score(rf, data_ibp, labels,
                                             cv=cv).mean()
        scores_deb[jj, kk] = cross_val_score(rf, data_deb, labels,
                                             cv=cv).mean()

ibp = np.swapaxes(scores_ibp, 0, 1).flatten()
deb = np.swapaxes(scores_deb, 0, 1).flatten()

scores = np.r_[ibp, deb]
n_scores = len(ibp)
models = n_scores * [r"$\alpha_{OT^{\mathcal{U}}_{\varepsilon}}$ (IBP)"]
models += n_scores * [r"$\alpha_{S_{\varepsilon}}$ (proposed)"]
all_eps = np.array(epsilons)[:, None] * \
    np.ones(n_forests * n_permutations)[None, :]
all_eps = all_eps.flatten()
all_eps = np.r_[all_eps, all_eps]
data = dict(score=scores, model=models, eps=all_eps)
df = pd.DataFrame(data)

colors = ["indianred", "cornflowerblue"]
hue_order = [r"$\alpha_{OT^{\mathcal{U}}_{\varepsilon}}$ (IBP)",
             r"$\alpha_{S_{\varepsilon}}$ (proposed)"]
df = df[df.eps < 0.2]
df.to_csv("data/mnist-embedding.csv")
