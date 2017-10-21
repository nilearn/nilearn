"""
Illustration of the volume to surface sampling scheme
=====================================================

When using the 'ball' kind of sampling for cortical surface projections, image
values are interpolated at points uniformly spread in a ball around each vertex
location.

This example illustrates this strategy in 2d by showing some images, vertex
locations, sample points and interpolated values.

"""

import numpy as np
import matplotlib.patches
from matplotlib import pyplot as plt
from nilearn.plotting import surf_plotting


np.random.seed(13)

# get some piecewise constant images
w, h = 7, 10
small_image = np.random.uniform(0, 1, size=(w, h))
small_image -= small_image.min()
small_image /= small_image.max()
image = np.empty((10 * w, 10 * h))
for i in range(w):
    for j in range(h):
        image[10 * i:10 * i + 10, 10 * j:10 * j + 10] = small_image[i, j]
images = [image, image > .5, image < .5, image < .8]


# get a mesh
n_nodes = 7
x = np.random.uniform(0, image.shape[0], size=(n_nodes, ))
y = np.random.uniform(0, image.shape[1], size=(n_nodes, ))
mesh = np.asarray([x, y]).T, None


# get interpolated values for each point of the mesh
ball_radius = 10
n_points = 7
vertices = mesh[0]
all_values, sample_points = surf_plotting._sampling(
    images, mesh, np.eye(3), n_points=n_points,
    radius=ball_radius, kind='ball')

# plot the images, the mesh, and the sample points
fig, axes = plt.subplots(2, 2)
axes = axes.ravel()
for image, values, ax in zip(images, all_values, axes):
    ax.imshow(image, cmap='gray', vmin=0, vmax=1)
    ax.add_patch(matplotlib.patches.Rectangle(
        (-.5, -.5), image.shape[1], image.shape[0], fill=False,
        linestyle='dashed'))
    ax.scatter(
        vertices[:, 1], vertices[:, 0], c=values, s=300, cmap='gray',
        edgecolors='r', vmin=0, vmax=1, zorder=2)
    for sp, mp in zip(sample_points, vertices):
        ax.scatter(sp[:, 1], sp[:, 0], marker='x', color='blue', zorder=3)
        for s in sp:
            ax.plot(
                [mp[1], s[1]], [mp[0], s[0]], color='red', alpha=.5, zorder=1)

plt.show()
