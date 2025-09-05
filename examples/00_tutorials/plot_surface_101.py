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
# unit that holds the data.
# For volumetric images, that basic unit is a voxel,
# while for surface images it is a :term:`vertex`.
#
# The goal of this tutorial is to show you how to work with surface images in
# Nilearn. For more existential questions like why surface images are useful,
# how they are created etc., `Andy Jahn's blog
# <https://andysbrainbook.readthedocs.io/en/latest/FreeSurfer/FreeSurfer_Introduction.html>`_
# is a good starting point.
#
# Surface images have two main components:
#
# 1. The :term:`mesh`, which is the geometry of the surface.
# 2. The data, which is the information stored at each vertex of the mesh.

# %%
# Mesh
# ----
#
# A :term:`mesh` can be defined by two arrays:
#
# 1. The coordinates of the vertices.
# 2. Which vertices need to be connected to form :term:`faces`.
#
# .. note:: This representation of a mesh is known as `Face-Vertex
#           <https://en.wikipedia.org/wiki/Polygon_mesh#Face-vertex_meshes>`_
#           representation.
#
# For brain surfaces we typically have two meshes: one for the left hemisphere
# and one for the right hemisphere.
# Nilearn represents this as a
# :class:`~nilearn.surface.PolyMesh` object with two ``parts``:
# ``left`` and ``right``.
#
# So you can define your own :term:`mesh`, say, for the left part a tetrahedron
# and for the right part a pyramid, using numpy arrays and create a
# :class:`~nilearn.surface.PolyMesh` object as follows:
import numpy as np

from nilearn.surface import InMemoryMesh, PolyMesh

# for the tetrahedron
left_coords = np.asarray(
    [
        [0, 0, 0],  # vertex 0
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
    np.asarray(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    + 2
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
#
# The data is the information stored at each :term:`vertex`
# of the :term:`mesh`.
# This can be anything from the thickness of the cortex
# to the activation level at that :term:`vertex`.
#
# For this example, let's create some random data
# for the vertices of the :term:`mesh`:
rng = np.random.default_rng(0)
left_data = rng.random(mesh.parts["left"].n_vertices)
right_data = rng.random(mesh.parts["right"].n_vertices)
# put them together in a dictionary
data = {
    "left": left_data,
    "right": right_data,
}

# %%
# Creating a surface image
# ------------------------
#
# Now we can create a surface image by combining the :term:`mesh` and the data
# using the :class:`~nilearn.surface.SurfaceImage` class:
from nilearn.surface import SurfaceImage

surface_image = SurfaceImage(mesh=mesh, data=data)

# %%
# Plotting the surface image
# --------------------------
#
# The surface image can be plotted using the different functions
# from the :mod:`~nilearn.plotting` module.
# Here we will show how to use the
# :func:`~nilearn.plotting.view_surf` function:
from nilearn.plotting import view_surf

# %%
# Plot the left part
view_surf(surf_map=surface_image, hemi="left", darkness=None)


# %%
# Plot the right part
view_surf(surf_map=surface_image, hemi="right", darkness=None)

# %%
# Data format
# -----------
#
# Brain-related surface data are typically stored in the GIFTI format
# (``.gii`` files) which can be saved to and loaded from via Nilearn.

# %%
# Save the surface image
# ----------------------
#
# You can save the :term:`mesh` and the data separately as GIFTI files:
from pathlib import Path

output_dir = Path.cwd() / "results" / "plot_surface_101"
output_dir.mkdir(exist_ok=True, parents=True)
print(f"Output will be saved to: {output_dir}")
surface_image.mesh.to_filename(output_dir / "surface_image_mesh.gii")
surface_image.data.to_filename(output_dir / "surface_image_data.gii")

# %%
# You will see that this creates four files in total -- two for the
# :term:`mesh` and two for the data.
# The files ending with ``_hemi-L.gii``
# correspond to the left part and those ending with ``_hemi-R.gii`` correspond
# to the right part.

# %%
# Load the surface image
# ----------------------
#
# You can load the saved files back into Nilearn using the
# :class:`~nilearn.surface.SurfaceImage` object:
mesh = {
    "left": output_dir / "surface_image_mesh_hemi-L.gii",
    "right": output_dir / "surface_image_mesh_hemi-R.gii",
}
data = {
    "left": output_dir / "surface_image_data_hemi-L.gii",
    "right": output_dir / "surface_image_data_hemi-R.gii",
}

surface_image_loaded = SurfaceImage(
    mesh=mesh,
    data=data,
)

# %%
# You can now plot the loaded surface image:
view_surf(surf_map=surface_image_loaded, hemi="left", darkness=None)

# %%
# And that's it! Now you know how to create, plot, save and load surface images
# with Nilearn.
#
# Further reading
# ---------------
#
# Most things that can be done with volumetric images can also be done with
# surface images.
# See following examples for more details:
#
# * For plotting statistical maps on the surface, see
#   :ref:`sphx_glr_auto_examples_01_plotting_plot_surf_stat_map.py`
#
# * For performing GLM analysis on surface data organized in BIDS
#   format, see
#   :ref:`sphx_glr_auto_examples_07_advanced_plot_surface_bids_analysis.py`
#
# * For performing first-level GLM analysis on surface data,
#   see this example
#   :ref:`sphx_glr_auto_examples_04_glm_first_level\
#   _plot_localizer_surface_analysis.py`.
