"""
Extracting signals from brain regions using the NiftiLabelsMasker
=================================================================

This simple example shows how to extract signals from functional
:term:`fMRI` data and brain regions defined through an atlas.
More precisely, this example shows how to use the
:class:`~nilearn.maskers.NiftiLabelsMasker` object to perform this
operation in just a few lines of code.

"""

from nilearn._utils.helpers import check_matplotlib

check_matplotlib()


# %%
# Retrieve the brain development functional dataset
# -------------------------------------------------
#
# We start by fetching the brain development functional dataset
# and we restrict the example to one subject only.
from nilearn.datasets import fetch_atlas_harvard_oxford, fetch_development_fmri

dataset = fetch_development_fmri(n_subjects=1)
func_filename = dataset.func[0]

# print basic information on the dataset
print(f"First functional nifti image (4D) is at: {func_filename}")

# %%
# Load an atlas
# -------------
#
# We then load the Harvard-Oxford atlas to define the brain regions
# and the first label correspond to the background.
#

atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
print(f"The atlas contains {len(atlas.labels) - 1} non-overlapping regions")

# %%
# Instantiate the mask and visualize atlas
# ----------------------------------------
#
# Instantiate the masker with label image and label values
#
from nilearn.maskers import NiftiLabelsMasker

masker = NiftiLabelsMasker(atlas.maps, lut=atlas.lut, verbose=1)

# %%
# Visualize the atlas
# -------------------
#
# We need to call fit prior to generating the mask.
# We can then generate a report to visualize the atlas.
#
# .. include:: ../../../examples/report_note.rst
#
masker.fit()

report = masker.generate_report()
report

# %%
# Fitting the masker on data and generating a report
# --------------------------------------------------
#
# We can again generate a report, but this time,
# the provided functional image is displayed with the ROI of the atlas.
# The report also contains a summary table giving the region sizes in mm3.
#
masker.fit(func_filename)

report = masker.generate_report()
report

# %%
# Process the data with the NiftiLablesMasker
# -------------------------------------------
#
# In order to extract the signals, we need to call transform on the
# functional data.
signals = masker.transform(func_filename)

# signals is a 2D numpy array, (n_time_points x n_regions)
print(f"{signals.shape=}")

# %%
# Output to dataframe and plot
# ----------------------------
#
# You can use 'set_output()' to decide the output format of 'transform'.
# If you want to output to a DataFrame, you an choose pandas and polars.
#
masker.set_output(transform="pandas")
signals_df = masker.transform(func_filename)
print(signals_df.head)

signals_df[["Frontal Pole", "Insular Cortex", "Superior Frontal Gyrus"]].plot(
    title="Signals from 3 regions", figsize=(15, 5)
)
