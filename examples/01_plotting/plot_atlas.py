"""
Basic Atlas plotting
====================

Plot the regions of a reference atlas (Harvard-Oxford and Juelich atlases).
"""

##########################################################################
# Retrieving the atlas data
# -------------------------

from nilearn import datasets

dataset_ho = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
dataset_ju = datasets.fetch_atlas_juelich("maxprob-thr0-1mm")

atlas_ho_filename = dataset_ho.filename
atlas_ju_filename = dataset_ju.filename

print(f"Atlas ROIs are located at: {atlas_ho_filename}")
print(f"Atlas ROIs are located at: {atlas_ju_filename}")

###########################################################################
# Visualizing the Harvard-Oxford atlas
# ------------------------------------

from nilearn import plotting

plotting.plot_roi(atlas_ho_filename, title="Harvard Oxford atlas")

###########################################################################
# Visualizing the Juelich atlas
# -----------------------------

plotting.plot_roi(atlas_ju_filename, title="Juelich atlas")

###########################################################################
# Visualizing the Harvard-Oxford atlas with contours
# --------------------------------------------------
plotting.plot_roi(
    atlas_ho_filename,
    view_type="contours",
    title="Harvard Oxford atlas in contours",
)
plotting.show()

###########################################################################
# Visualizing the Juelich atlas with contours
# -------------------------------------------
plotting.plot_roi(
    atlas_ju_filename, view_type="contours", title="Juelich atlas in contours"
)
plotting.show()
