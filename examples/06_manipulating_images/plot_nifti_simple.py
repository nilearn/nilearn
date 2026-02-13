"""
Simple example of NiftiMasker use
=================================

Here is a simple example of automatic mask computation using the nifti masker.
The mask is computed and visualized.

"""

# %%
# Retrieve the brain development functional dataset
# -------------------------------------------------
#
# We fetch the dataset and print some basic information about it.
#

from nilearn.datasets import fetch_development_fmri

dataset = fetch_development_fmri(n_subjects=1)
func_filename = dataset.func[0]

print(f"First functional nifti image (4D) is at: {func_filename}")

# %%
# Compute the mask
# ----------------
#
# As the input image is an EPI image,
# the background is noisy
# and we cannot rely on the ``'background'`` masking strategy.
# We need to use the ``'epi'`` one.
from nilearn.maskers import NiftiMasker

masker = NiftiMasker(
    standardize="zscore_sample",
    mask_strategy="epi",
    memory="nilearn_cache",
    memory_level=1,
    smoothing_fwhm=8,
    verbose=1,
)

# %%
#
# .. include:: ../../../examples/html_repr_note.rst
#
masker

# %%
masker.fit(func_filename)

# %%
# .. note ::
#
#   You can also note that after fitting,
#   the HTML representation of the estimator looks different
#   than before before fitting.
#
masker

# %%
# Visualize the mask
# ------------------
#
# We can quickly get an idea about the estimated mask
# for this functional image by plotting the mask.
#
# We get the estimated mask from the ``mask_img_`` attribute of the masker:
# the final ``_`` ofd this attribute name means it was generated
# by the :meth:`~nilearn.maskers.NiftiMasker.fit` method.
#
# We can then plot it using the :func:`~nilearn.plotting.plot_roi` function
# with the mean functional image as background.
from nilearn.image.image import mean_img
from nilearn.plotting import plot_roi, show

mask_img = masker.mask_img_

mean_func_img = mean_img(func_filename)

plot_roi(mask_img, mean_func_img, display_mode="y", cut_coords=4, title="Mask")

show()

# %%
# Visualize the masker report
# ---------------------------
#
# More information can be obtained about the masker and its mask
# by generating a masker report.
# This can be done using
# the :meth:`~nilearn.maskers.NiftiMasker.generate_report` method.
report = masker.generate_report()

# %%
#
# .. include:: ../../../examples/report_note.rst
#
report

# %%
# Preprocess data with the NiftiMasker
# ------------------------------------
#
# We extract the data from the nifti image and turn it into a numpy array.
#
fmri_masked = masker.transform(func_filename)
print(fmri_masked.shape)

# %%
# ``fmri_masked`` is now a 2D numpy array, (n_voxels x n_time_points).

# %%
# Run an algorithm and visualize the results
# ------------------------------------------
#
# Given that we now have a numpy array,
# we can then pass the data the wide range of algorithm.
# Here we will just do an independent component analysis,
# turned the extracted component back into images
# (using :meth:`~nilearn.maskers.NiftiMasker.inverse_transform`),
# then we will plot the first component.
#
from sklearn.decomposition import FastICA

from nilearn.image import index_img
from nilearn.plotting import plot_stat_map, show

ica = FastICA(n_components=10, random_state=42, tol=0.001, max_iter=2000)
components_masked = ica.fit_transform(fmri_masked.T).T

components = masker.inverse_transform(components_masked)

plot_stat_map(
    index_img(components, 0),
    mean_func_img,
    display_mode="y",
    cut_coords=4,
    title="Component 0",
)

show()
