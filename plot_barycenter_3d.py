import pickle
import numpy as np
import torch
import pyvista as pv
from pyvista import examples
from run_barycenter_3d import rescale_data


with open("data/interpolation-data.pkl", "rb") as ff:
    data = pickle.load(ff)


beta = pv.ParametricTorus()
beta = pv.PolyData(beta)

# rotate the tore
beta.rotate_y(90)

alpha = examples.download_bunny()
# rotate the bunny
alpha.rotate_x(100)
alpha.rotate_z(140)
alpha.rotate_y(-20)
alpha = alpha.smooth(100, relaxation_factor=0.1)
beta = beta.smooth(100, relaxation_factor=0.1)

alpha = rescale_data(alpha, 0.95)
beta = rescale_data(beta, 0.95)
width = 200
n_features = width ** 3


hist_grid = torch.linspace(-1., 1., width + 1)
grid = torch.linspace(-1., 1., width)
X, Y, Z = torch.meshgrid(grid, grid, grid)
threshold = 1e-7


plotter = pv.Plotter(off_screen=True, point_smoothing=True)
plotter.set_background("w")
plotter.add_mesh(alpha, color="r")
plotter.show(screenshot="fig/3d/rabbit")

plotter = pv.Plotter(off_screen=True, point_smoothing=True)
plotter.set_background("w")
plotter.add_mesh(beta, color="r")
plotter.show(screenshot="fig/3d/tore")


cpos = [(3.3, 3.3, 3.3), (0.0, 0.005, 0.0), (0.0, 0.0, 1.0)]
meshes = []
for key in ["ibp", "deb"]:
    meshes.append([])
    bars = data[key]["bars"]
    for ii, hist in enumerate(bars):
        print("->> creating mesh {} ... ".format(ii + 1))
        support = torch.where(hist > threshold)
        weights = hist[support].numpy()
        cloud = torch.stack((X[support], Y[support], Z[support])).t()
        mesh = pv.PolyData(cloud.numpy())
        mesh.add_field_array(1 - weights, "weights")
        meshes[-1].append(mesh)
        plotter = pv.Plotter(off_screen=True, point_smoothing=True)
        plotter.set_background("w")
        plotter.camera_position = cpos
        plotter.add_mesh(mesh, scalars="weights", opacity="weights",
                         use_transparency=True, cmap="hot_r",
                         style="surface", show_scalar_bar=False)
        plotter.show(screenshot="fig/3d/%s-%d.png" % (key, ii))
        plotter.close()
