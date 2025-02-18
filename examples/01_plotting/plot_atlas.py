"""
Basic Atlas plotting
====================

Plot the regions of reference atlases.
"""

# %%
# Retrieving the atlas data
# -------------------------

from nilearn import datasets

dataset_ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_ho_filename = dataset_ho.filename
print(f"Atlas ROIs are located at: {atlas_ho_filename}")

dataset_ju = datasets.fetch_atlas_juelich("maxprob-thr0-1mm")
atlas_ju_filename = dataset_ju.filename
print(f"Atlas ROIs are located at: {atlas_ju_filename}")

# %%
# Visualizing the Harvard-Oxford atlas
# ------------------------------------

from nilearn.plotting import plot_roi, show

plot_roi(atlas_ho_filename, title="Harvard Oxford atlas")

# %%
# Visualizing the Juelich atlas
# -----------------------------

plot_roi(atlas_ju_filename, title="Juelich atlas")

# %%
# Visualizing the Harvard-Oxford atlas with contours
# --------------------------------------------------
plot_roi(
    atlas_ho_filename,
    view_type="contours",
    title="Harvard Oxford atlas in contours",
)
show()

# %%
# Visualizing the Juelich atlas with contours
# -------------------------------------------
plot_roi(
    atlas_ju_filename, view_type="contours", title="Juelich atlas in contours"
)
show()


# %%
# Visualizing an atlas with its own colormap
# ------------------------------------------
# Some atlases come with a look-up table
# that determines the color to use to represent each of its regions.
#
# You can pass this look-up table
# as a pandas dataframe to the ``cmap`` argument
# to use its colormap.
#
# .. admonition:: Control via commit message
#    :class: tip
#
#    The look-up table must be formatted according to the BIDS standard.
#    and that the colors must be in ``color`` column using hexadecimal values.
#
#    If an invalid look-up table is passed,
#    a warning will be thrown and the ``plot_roi`` function
#    will fall back to using its default colormap.
#

# %%
# Here we are using the Yeo atlas
# that comes with a predefined color
dataset_yeo = datasets.fetch_atlas_yeo_2011(n_networks=17)

plot_roi(
    dataset_yeo.maps,
    title="Yeo atlas",
    colorbar=True,
)

print(dataset_yeo.lut)

plot_roi(
    dataset_yeo.maps,
    title="Yeo atlas with its own colors",
    cmap=dataset_yeo.lut,
    colorbar=True,
)

show()

# sphinx_gallery_dummy_images=1
