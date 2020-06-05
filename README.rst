Debiased Sinkhorn barycenters
-----------------------------

Guide to reproduce the results of (Janati et al, 2020, Debiased Sinkhorn barycenters)
available at http://arxiv.org/abs/2006.02575.
All experiments can be ran
on CPU or GPUs. In the results reported in the paper, this is our used config
with the computation time:

- Theorem illustrations + convergence plot (CPU) a few seconds
- Ellipses (CPU): 3 minutes
- Barycenters of 3D shapes (GPU): 15 seconds
- OT barycentric embedding: ot embedding on (GPU) (1h) + Random forest
    training on CPU (5 minutes)


All figures are saved in the fig/ folder.


Dependencies
------------

Please make sure you have a
miniconda environment installed and the following
necessary dependencies (available through pip or conda):

- numpy
- matplotlib
- scikit-learn
- torch
- pandas



1. Ellipses Experiment
----------------------
Moreover, to reproduce the Ellipse experiment, you will need:

1. to install the free support barycenter code of (G. Luise, 2019) that is
shipped in the folder otbar. Inside the folder otbar, run:
```
    python setup.py develop
```

2. to have an installed version of matlab 2019b to reproduce the MAAIPM barycenter
of (Dongdong, 2019). And to install the Matlab engine API for Python.
See https://fr.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html


3. run `python plot_ellipse_bar.py`



2. Barycenters of 3D shapes
---------------------------

run `python run_barycenter_3d.py`
run `python plot_barycenter_3d.py`

The images are saved in the fig/3d folder.



3. MNIST barycentric Encoding
-----------------------------

run `python run_ot_embedding.py`
run `python run_random_forest.py`
run `python plot_mnist_scores.py`
