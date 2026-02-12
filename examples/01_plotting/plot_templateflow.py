"""

Using brain templates from TemplateFlow
=======================================

In this example, we see how to fetch and use brain templates
from `TemplateFlow <https://https://www.templateflow.org/>`_.

We first see how to fetch and use the Desikan-Killiany (DK) atlas
in order to make surface plots.

Then, we see how to get the proper template
for the Harvard-Oxford volumetric atlas.
"""

# %%
# Fetch DK Surface template
# -------------------------
# TemplateFlow allows for the retrieval of image files, annotations, and
# metadata for various brain templates.
# Specific files are accessed via the ``get`` function
# by providing the ``template`` name
# and filtering by file name 'entities'
# such as resolution (``res``),
# ``suffix``, ``atlas``, hemisphere (``hemi``), ``space``,
# description (``desc``)...
#
# .. admonition:: Tips
#
#   TemplateFlow files follow a `BIDS filename template <https://bids.neuroimaging.io/getting_started/folders_and_files/files.html#filename-template>`_
#   and should more less be organized as `BIDS template <https://bids-specification.readthedocs.io/en/latest/derivatives/atlas.html>`_.
#
# If a search matches multiple files,
# the function returns a list of all matching paths.
#
# Available resources can be explored in the
# `TemplateFlow browser <https://www.templateflow.org/browse/>`_.

import templateflow.api as tflow

from nilearn.surface import SurfaceImage, load_surf_data

template = "fsaverage"

fetched_files = tflow.get(
    template,
    extension="surf.gii",
    suffix="pial",
)
print(fetched_files)

# %%
# In this example, the Desikan-Killiany (DK) atlas is fetched.
# DK is defined with reference to the _FreeSurfer average_ (fsaverage) template,
# which is provided at multiple densities (number of vertices per hemisphere):
# 3k, 10k, 41k, 164k
# Here, we build a SurfaceImage containing the labels for both hemispheres
# for the 3k and 10k densities.

used_densities = ["3k", "10k"]
# uncomment the following line in case you want to see all densities
# used_densities = ["3k", "10k", "41k", "164k"]

desikan_dict = {}
for density in ["3k", "10k"]:
    mesh = {}
    data = {}
    for hemi in ["left", "right"]:
        mesh[hemi] = tflow.get(
            template,
            extension="surf.gii",
            suffix="pial",
            density=density,
            hemi=hemi[0].upper(),
        )

        desc = "curated" if density == "164k" else None
        roi_data = load_surf_data(
            tflow.get(
                template,
                atlas="Desikan2006",
                density=density,
                hemi=hemi[0].upper(),
                extension="label.gii",
                desc=desc,
            )
        )
        if density == "164k":
            roi_data[roi_data < 0] = 0

        data[hemi] = roi_data

    desikan_dict[density] = SurfaceImage(mesh=mesh, data=data)


# %%
# Plotting the DK atlas
# ---------------------
# To plot the DK atlas, we also fetch the lookup table (LUT) from TemplateFlow,
# to map region indices to the original region names and colors.
# We also use the background sulcal depth map
# to add more detail to the surface plot.
# To do this we map the density
# to the corresponding fsaverage data that Nilearn can fetch directly.


import matplotlib.pyplot as plt

from nilearn.datasets import load_fsaverage_data
from nilearn.plotting import plot_surf_roi, show

_, axes = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

fs_density = {
    "3k": "fsaverage4",
    "10k": "fsaverage5",
    "41k": "fsaverage6",
    "164k": "fsaverage",
}

lut = tflow.get(
    template,
    atlas="Desikan2006",
    suffix="dseg",
    extension="tsv",
)

for ax, density in zip(axes, ["3k", "10k"], strict=True):
    desikan = desikan_dict[density]
    sulcal_depth_map = load_fsaverage_data(
        mesh=fs_density[density], data_type="sulcal"
    )
    plot_surf_roi(
        roi_map=desikan,
        cmap=lut,
        bg_map=sulcal_depth_map,
        bg_on_data=True,
        title=f"DK atlas ({density})",
        axes=ax,
    )
    if density == "10k":
        ax._colorbars[0].remove()

show()


# %%
# Getting the right template for volumetric atlases
# -------------------------------------------------
# The template used by nilearn to plot volumetric images is ICBM152 2009,
# release a.
# However, some atlases are defined on different templates,
# this is the case for the Harvard-Oxford atlas.
# This can lead to some imprecisions
# when plotting volumetric atlases, with regions encompassing non-brain areas.

from nilearn.datasets import fetch_atlas_harvard_oxford
from nilearn.plotting import plot_roi

plotting_params = {
    "view_type": "contours",
    "display_mode": "x",
    "cut_coords": [-4, -2, 0],
    "black_bg": False,
}

harvard_oxford_sub = fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")


plot_roi(
    harvard_oxford_sub.filename,
    title="Harvard-Oxford atlas | sub-cortical | ICBM152 2009",
    **plotting_params,
)


# %%
# This is because the template of the Harvard-Oxford atlas is:

print(f"Harvard-Oxford atlas template: {harvard_oxford_sub.template}")

# %%
# Getting a template
# ------------------
# If you want to visualize the Harvard-Oxford atlas on the proper template,
# you can get it from templateFlow.
# This template is also used by FSL.

template = "MNI152NLin6Asym"
resolution = "01"

MNI152NLin6Asym_template_img = tflow.get(
    template,
    resolution=resolution,
    suffix="T1w",
    desc="brain",
    extension="nii.gz",
)

print(f"{MNI152NLin6Asym_template_img=}")

plot_roi(
    harvard_oxford_sub.filename,
    title="Harvard-Oxford atlas | sub-cortical | MNI152NLin6Asym",
    bg_img=MNI152NLin6Asym_template_img,
    **plotting_params,
)

show()
