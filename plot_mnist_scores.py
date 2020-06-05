from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

params = {"legend.fontsize": 18,
          "axes.titlesize": 16,
          "axes.labelsize": 16,
          "xtick.labelsize": 13,
          "ytick.labelsize": 13,
          "pdf.fonttype": 42}
plt.rcParams.update(params)
sns.set_context("paper", rc=params)

df = pd.read_csv("data/mnist-embedding.csv")

colors = ["indianred", "cornflowerblue"]
hue_order = [r"$\alpha_{OT^{\mathcal{U}}_{\varepsilon}}$ (IBP)",
             r"$\alpha_{S_{\varepsilon}}$ (proposed)"]
df = df[df.eps < 0.2]
df.to_csv("data/mnist-embedding.csv")
f, ax = plt.subplots(1, 1)
sns.boxplot(y="score", x="eps", hue="model", data=df, ax=ax,
              palette=colors, hue_order=hue_order)
ax.set_ylabel("Cross-validation score")
ax.legend(ncol=2, bbox_to_anchor=(1.05, +1.2),
          frameon=False)
ax.grid("on")
ax.set_xlabel(r"$\varepsilon$")
plt.savefig("fig/mnist-cv.pdf")
