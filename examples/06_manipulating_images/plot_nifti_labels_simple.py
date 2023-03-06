"""
Extracting signals from brain regions using the NiftiLabelsMasker
=================================================================

This simple example shows how to extract signals from functional
:term:`fMRI` data and brain regions defined through an atlas.
More precisely, this example shows how to use the
:class:`~nilearn.maskers.NiftiLabelsMasker` object to perform this
operation in just a few lines of code.

.. include:: ../../../examples/masker_note.rst

"""

###########################################################################
# Retrieve the brain development functional dataset
#
# We start by fetching the brain development functional dataset
# and we restrict the example to one subject only.

from nilearn import datasets

dataset = datasets.fetch_development_fmri(n_subjects=1)
func_filename = dataset.func[0]

# print basic information on the dataset
print(f"First functional nifti image (4D) is at: {func_filename}")

###########################################################################
# Load an atlas
#
# We then load the Harvard-Oxford atlas to define the brain regions
atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")

# The first label correspond to the background
print(f"The atlas contains {len(atlas.labels) - 1} non-overlapping regions")

###########################################################################
# Instantiate the mask and visualize atlas
#
from nilearn.maskers import NiftiLabelsMasker

# Instantiate the masker with label image and label values
masker = NiftiLabelsMasker(atlas.maps, labels=atlas.labels, standardize=True)

# Visualize the atlas
# Note that we need to call fit prior to generating the mask
masker.fit()

# At this point, no functional image has been provided to the masker.
# We can still generate a report which can be displayed in a Jupyter
# Notebook, opened in a browser using the .open_in_browser() method,
# or saved to a file using the .save_as_html(output_filepath) method.
report = masker.generate_report()
report

##########################################################################
# Fitting the mask and generating a report
masker.fit(func_filename)

# We can again generate a report, but this time, the provided functional
# image is displayed with the ROI of the atlas.
# The report also contains a summary table giving the region sizes in mm3
report = masker.generate_report()
report

###########################################################################
# Process the data with the NiftiLablesMasker
#
# In order to extract the signals, we need to call transform on the
# functional data
signals = masker.transform(func_filename)
# signals is a 2D matrix, (n_time_points x n_regions)
signals.shape

###########################################################################
# Plot the signals
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(15, 5))
ax = fig.add_subplot(111)
for label_idx in range(3):
    ax.plot(
        signals[:, label_idx], linewidth=2, label=atlas.labels[label_idx + 1]
    )  # 0 is background
ax.legend(loc=2)
ax.set_title("Signals for first 3 regions")
plt.show()
