"""
Illustration of the volume to cortical surface sampling schemes
===============================================================

"""

import numpy as np

import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from nilearn.plotting import surf_plotting


######################################################################
# Build a mesh (of a cylinder)
######################################################################

N_Z = 5
N_T = 10
u, v = np.mgrid[:N_T, :N_Z]
triangulation = matplotlib.tri.Triangulation(u.flatten(), v.flatten())
angles = u.flatten() * 2 * np.pi / N_T
x, y = np.cos(angles), np.sin(angles)
z = v.flatten() * 2 / N_Z

mesh = [np.asarray([x, y, z]).T, triangulation.triangles]


######################################################################
# Get the locations from which surf_plotting.niimg_to_surf_data
# would draw its samples
######################################################################

line_sample_points = surf_plotting._line_sample_locations(
    mesh, np.eye(4), segment_half_width=.2, n_points=6)

ball_sample_points = surf_plotting._ball_sample_locations(
    mesh, np.eye(4), ball_radius=.15, n_points=20)


######################################################################
# Plot the mesh and the sample locations
######################################################################

for sample_points in [line_sample_points, ball_sample_points]:
    fig = plt.figure()
    ax = plt.subplot(projection='3d')
    ax.set_aspect(1)

    ax.plot_trisurf(x, y, z, triangles=triangulation.triangles)

    ax.scatter(*sample_points.T, color='r')

plt.show()
