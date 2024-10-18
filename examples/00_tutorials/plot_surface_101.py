"""
Working with Surface images
===========================

Here we explain how surface images are represented within Nilearn. We will
also show how to work with them.
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
# while for surface images it is a vertex.
#
# The goal of this tutorial is to show how to work with surface images in
# Nilearn. For more existential questions like why surface images are useful,
# how they are created etc., [Andy Jahn's blog](https://andysbrainbook.readthedocs.io/en/latest/FreeSurfer/FreeSurfer_Introduction.html)
# is a good starting point.

# %%
# GIFTI data format
# -----------------
#
# Surface images are typically stored in the GIFTI format (`.gii` files) which
# can be read via Nilearn.
#
# Nilearn divides surface images into two main components:
#  1. The mesh, which is the geometry of the surface.
#  2. The data, which is the information stored at each vertex of the mesh.

# %%
# Mesh
# ----
#
# The mesh is the geometry of the surface and can be defined by two arrays:
#  1. The coordinates of the vertices.
#  2. Which vertices need to be connected to form faces.
#
# For brain surfaces we typically have two meshes: one for the left hemisphere
# and one for the right hemisphere. Nilearn represents this as a `PolyMesh`
# object with two `parts`: `left` and `right`.
#
# So you can define your own mesh, say, for the left part a tetrahedron and for
# the right part a pyramid, using numpy arrays and create a `PolyMesh` object
# as follows:
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
# The data is the information stored at each vertex of the mesh. This can be
# anything from the thickness of the cortex to the activation level of a
# at that vertex.
#
# For this example, let's create some random data for the vertices of the mesh:
rng = np.random.default_rng(0)
left_data = rng.random(mesh.parts["left"].n_vertices)
right_data = rng.random(mesh.parts["right"].n_vertices)
# put them together in a dictionary
data = {"left": left_data, "right": right_data}

# %%
# Creating a surface image
# ------------------------
# Now we can create a surface image by combining the mesh and the data:
from nilearn.experimental.surface import SurfaceImage

surface_image = SurfaceImage(mesh=mesh, data=data)

# %%
# Plotting the surface image
# --------------------------
# The surface image can be plotted using the `view_surf` function:
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
# You can save the mesh and the data separately as GIFTI files:
surface_image.mesh.to_filename("surface_image_mesh.gii")

surface_image.data.to_filename("surface_image_data.gii")

# %%
# You will see that this creates four files in total -- two for the mesh and
# two for the data. The files ending with `_hemi-L.gii` are for the left
# hemisphere and those ending with `_hemi-R.gii` are for the right hemisphere.
