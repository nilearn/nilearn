"""Example to show how fetch and use templates from templateflow."""

import pandas as pd
from templateflow import api as tflow

from nilearn.datasets import (
    fetch_atlas_harvard_oxford,
    fetch_development_fmri,
    load_sample_motor_activation_image,
)
from nilearn.image import mean_img
from nilearn.maskers import NiftiLabelsMasker, NiftiMapsMasker
from nilearn.plotting import plot_roi, plot_stat_map, show

# %%
# Let's have a look at the Harvard-Oxford deterministic atlas
# that is provided with Nilearn.

harvard_oxford = fetch_atlas_harvard_oxford("cort-maxprob-thr25-1mm")

black_bg = False

plot_roi(
    harvard_oxford.filename, title="Harvard-Oxford atlas", black_bg=black_bg
)

# %%
# Looks fine, no?
# Well actually,
# if we plot only contours and
# check the occipital regions on a few coronal slices,
# you can see that a region seems to include some non-brain data.

plotting_params = {
    "view_type": "contours",
    "display_mode": "x",
    "cut_coords": [-4, -2, 0],
    "black_bg": black_bg,
}

plot_roi(
    harvard_oxford.filename,
    title="Harvard-Oxford atlas - contour",
    **plotting_params,
)

# %%
# This is a bit more obvious with the sub-cortical Harvard-Oxford atlas.

harvard_oxford_sub = fetch_atlas_harvard_oxford("sub-maxprob-thr25-1mm")


plot_roi(
    harvard_oxford_sub.filename,
    title="Harvard-Oxford atlas - sub-cortical",
    **plotting_params,
)

# This is because Nilearn, by default, plots images
# on the MNI template ICBM152 2009 (release a asymmetrical) .
# where as the template of the Harvard-Oxford atlas is...

print(f"{harvard_oxford.template=}")
print(f"{harvard_oxford_sub.template=}")

#  %%
# Getting a template
# ------------------
# If you want to visualize the Harvard-Oxford atlas on the proper template,
# you can get it from templateFlow.

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
    harvard_oxford.filename,
    title="Harvard-Oxford atlas on MNI152NLin6Asym",
    bg_img=MNI152NLin6Asym_template_img,
    **plotting_params,
)

plot_roi(
    harvard_oxford_sub.filename,
    title="Harvard-Oxford atlas sub-cortical on MNI152NLin6Asym",
    bg_img=MNI152NLin6Asym_template_img,
    **plotting_params,
)

show()

# %%
# Using the wrong template

n_subjects = 50

data = fetch_development_fmri(n_subjects=n_subjects)
print(data.func[0])

# From the filename we can see that this data was standardized
# to the MNI152NLin2009cAsym template (the default used by fMRIprep).

# Let's see what the default Harvard-Oxford atlas look like when projected
# on the mean functional image of the first subject.

bg_img = mean_img(data.func[0])

plot_roi(
    harvard_oxford.filename,
    title="Atlas on mean functional subject 1",
    bg_img=bg_img,
    view_type="contours",
)


# Now let's get the Atlas on the MNI152NLin2009cAsym template.

template = "MNI152NLin2009cAsym"
resolution = "01"

MNI152NLin2009cAsym_harvard_oxford = tflow.get(
    template,
    resolution=resolution,
    atlas="HOCPA",
    suffix="dseg",
    desc="th25",
    extension="nii.gz",
)

print(f"{MNI152NLin2009cAsym_harvard_oxford=}")

plot_roi(
    MNI152NLin2009cAsym_harvard_oxford,
    title="Atlas in MNI152NLin2009cAsym template on mean functional subject 1",
    bg_img=bg_img,
    view_type="contours",
)

show()

import numpy as np

from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import MultiNiftiLabelsMasker
from nilearn.plotting import plot_matrix

matrices = []

for tpl, name in zip(
    [harvard_oxford.filename, MNI152NLin2009cAsym_harvard_oxford],
    ["MNI152NLin6Asym", "MNI152NLin2009cAsym"],
    strict=False,
):
    masker = MultiNiftiLabelsMasker(
        labels_img=tpl,
        standardize="zscore_sample",
        standardize_confounds=True,
        memory="nilearn_cache",
        n_jobs=2,
        lut=harvard_oxford.lut,
    )

    time_series = masker.fit_transform(data.func)

    connectome_measure = ConnectivityMeasure(
        kind="correlation",
        standardize="zscore_sample",
    )

    correlation_matrices = connectome_measure.fit_transform(time_series)

    mean_correlation_matrix = connectome_measure.mean_

    np.fill_diagonal(mean_correlation_matrix, 0)

    plot_matrix(
        mean_correlation_matrix,
        figure=(10, 8),
        vmax=0.8,
        vmin=-0.8,
        title=name,
        labels=masker.region_names_.values(),
        # reorder=True
    )

    matrices.append(mean_correlation_matrix)


plot_matrix(
    matrices[0] - matrices[1],
    figure=(10, 8),
    vmax=0.1,
    vmin=-0.1,
    title="difference",
    labels=masker.region_names_.values(),
    # reorder=True
)

show()


#  %%
# Discrete segmentation: labels
# -----------------------------
#
# We get the Schaefer atlas discrete segmentation
# for with the "MNI152NLin6Asym" template
#

labels_img = tflow.get(
    template,
    desc="100Parcels7Networks",
    atlas="Schaefer2018",
    resolution=resolution,
    suffix="dseg",
    extension="nii.gz",
)


plot_roi(labels_img, title="Schaefer2018 - 100 Parcels - 7 Networks")

show()

lut = tflow.get(
    template,
    desc="100Parcels7Networks",
    atlas="Schaefer2018",
    suffix="dseg",
    extension="tsv",
)

masker = NiftiLabelsMasker(labels_img, lut=lut)

stat_map = load_sample_motor_activation_image()

plot_stat_map(stat_map)

show()

masker.fit(stat_map)

report = masker.generate_report()

report.open_in_browser()

#  %%
# Probabilistic segmentation: maps
# --------------------------------
#
#


maps_img = tflow.get(
    template,
    desc="64dimensions",
    atlas="DiFuMo",
    resolution="02",
    suffix="probseg",
    extension="nii.gz",
)

assert maps_img, maps_img

lut = tflow.get(
    template,
    desc="64dimensions",
    atlas="DiFuMo",
    suffix="probseg",
    extension="tsv",
)

assert lut, lut

masker = NiftiMapsMasker(maps_img)

masker.fit(stat_map)

report = masker.generate_report(displayed_maps=[1, 3])

report.open_in_browser()

lut_df = pd.read_csv(lut, sep="\t")

print(lut_df)
