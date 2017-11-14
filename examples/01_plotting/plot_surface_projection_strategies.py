"""
Illustration of the volume to cortical surface sampling schemes
===============================================================

In nilearn, plotting.niimg_to_surf_data allows us to measure
values of a 3d volume at the nodes of a cortical mesh, transforming it into
surface data. This data can then be plotted with
plotting.plot_surf_stat_map for example.

This script shows, on a toy example, where samples are drawn around each mesh
vertex. Image values are interpolated at each sample location, then these
samples are averaged to produce a value for the vertex.

Two strategies are available to choose sample locations: they can be spread
along the normal to the mesh, or inside a ball around the vertex. Don't worry
too much about choosing one or the other: they take a similar amount of time
and give almost identical results for most images.

"""

import numpy as np

import matplotlib
from matplotlib import pyplot as plt

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


#########################################################################
# Get the locations from which niimg_to_surf_data would draw its samples
#########################################################################

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
