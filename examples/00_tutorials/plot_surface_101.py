"""
Working with Surface images
===========================

Here we explain how surface images are represented within Nilearn and how you
can plot, save and load them.
"""

# %%
# What is a surface image?
# ------------------------
#
# Within the context of neuroimaging, a surface image is an alternative way of
# representing MRI data as opposed to a volumetric image.
#
# While volumetric images are 3D grids of voxels, surface images consist of
# points (vertices) in 3D space connected to represent the surface of the
# brain.
#
# Practically, this means that the main difference between the two is the basic
# unit that holds the data. For volumetric images, that basic unit is a voxel,
# while for surface images it is a :term:`vertex`.
#
# The goal of this tutorial is to show how to work with surface images in
# Nilearn. For more existential questions like why surface images are useful,
# how they are created etc., `Andy Jahn's blog
# <https://andysbrainbook.readthedocs.io/en/latest/FreeSurfer/FreeSurfer_Introduction.html>`_
# is a good starting point.

# %%
# GIFTI data format
# -----------------
#
# Surface images are typically stored in the GIFTI format (``.gii`` files)
# which can be read via Nilearn.
#
# Nilearn divides surface images into two main components:
#  1. The :term:`mesh`, which is the geometry of the surface.
#  2. The data, which is the information stored at each vertex of the mesh.

# %%
# Mesh
# ----
#
# The :term:`mesh` can be defined by two arrays:
#  1. The coordinates of the vertices.
#  2. Which vertices need to be connected to form :term:`faces`.
#
# .. note:: This representation of a mesh is known as `Face-Vertex
#           <https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes>`_
#           representation.
#
# For brain surfaces we typically have two meshes: one for the left hemisphere
# and one for the right hemisphere. Nilearn represents this as a
# :class:`~nilearn.experimental.surface.PolyMesh` object with two ``parts``:
# ``left`` and ``right``.
#
# So you can define your own :term:`mesh`, say, for the left part a tetrahedron
# and for the right part a pyramid, using numpy arrays and create a
# :class:`~nilearn.experimental.surface.PolyMesh` object as follows:
import numpy as np

from nilearn.experimental.surface import InMemoryMesh, PolyMesh

# for the tetrahedron
left_coords = np.asarray(
    [
        [0.0, 0, 0],  # vertex 0
        [1, 0, 0],  # vertex 1
        [0, 1, 0],  # vertex 2
        [0, 0, 1],  # vertex 3
    ]
)
left_faces = np.asarray(
    [
        [1, 0, 2],  # face created by connecting vertices 1, 0, 2
        [0, 1, 3],  # face created by connecting vertices 0, 1, 3
        [0, 3, 2],  # face created by connecting vertices 0, 3, 2
        [1, 2, 3],  # face created by connecting vertices 1, 2, 3
    ]
)
# for the pyramid
right_coords = (
    np.asarray([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]]) + 2.0
)
right_faces = np.asarray(
    [
        [0, 1, 4],
        [0, 3, 1],
        [1, 3, 2],
        [1, 2, 4],
        [2, 3, 4],
        [0, 4, 3],
    ]
)
# put the two parts together
mesh = PolyMesh(
    left=InMemoryMesh(left_coords, left_faces),
    right=InMemoryMesh(right_coords, right_faces),
)

# %%
# Data
# ----
# The data is the information stored at each :term:`vertex` of the
# :term:`mesh`. This can be anything from the thickness of the cortex to the
# activation level at that :term:`vertex`.
#
# For this example, let's create some random data for the vertices of the
# :term:`mesh`:
rng = np.random.default_rng(0)
left_data = rng.random(mesh.parts["left"].n_vertices)
right_data = rng.random(mesh.parts["right"].n_vertices)
# put them together in a dictionary
data = {"left": left_data, "right": right_data}

# %%
# Creating a surface image
# ------------------------
# Now we can create a surface image by combining the :term:`mesh` and the data
# using the :class:`~nilearn.experimental.surface.SurfaceImage` class:
from nilearn.experimental.surface import SurfaceImage

surface_image = SurfaceImage(mesh=mesh, data=data)

# %%
# Plotting the surface image
# --------------------------
# The surface image can be plotted using the different functions from the
# :mod:`nilearn.plotting` module. Here we will show how to use the
# :func:`~nilearn.experimental.plotting.view_surf` function:
from nilearn.experimental import plotting

# %%
# Plot the left part
plotting.view_surf(
    surf_mesh=surface_image.mesh, surf_map=surface_image, hemi="left"
)

# %%
# Plot the right part
plotting.view_surf(
    surf_mesh=surface_image.mesh, surf_map=surface_image, hemi="right"
)

# %%
# Save the surface image
# ----------------------
# You can save the :term:`mesh` and the data separately as GIFTI files:
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_surface_101"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")
surface_image.mesh.to_filename(output_dir / "surface_image_mesh.gii")
surface_image.data.to_filename(output_dir / "surface_image_data.gii")

# %%
# You will see that this creates four files in total -- two for the
# :term:`mesh` and two for the data. The files ending with ``_hemi-L.gii`` are
# correspond to the left part and those ending with ``_hemi-R.gii`` correspond
# to the right part.

# %%
# Load the surface image
# ----------------------
# You can load the saved files back into Nilearn using the
# :class:`~nilearn.experimental.surface.SurfaceImage` object:

mesh = {
    "left": "surface_image_mesh_hemi-L.gii.gz",
    "right": "surface_image_mesh_hemi-R.gii.gz",
}
data = {
    "left": "surface_image_data_hemi-L.gii.gz",
    "right": "surface_image_data_hemi-R.gii.gz",
}

surface_image_loaded = SurfaceImage(mesh=mesh, data=data)

# %%
# You can now plot the loaded surface image:
plotting.view_surf(
    surf_mesh=surface_image_loaded.mesh,
    surf_map=surface_image_loaded,
    hemi="left",
)
