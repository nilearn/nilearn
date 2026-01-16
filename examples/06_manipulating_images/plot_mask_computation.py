"""
Understanding NiftiMasker and mask computation
==============================================

In this example, the NiftiMasker is used to automatically compute a mask.

* The default strategy is based on the background.

* Another option is to use a template.

* For raw EPI, as in :term:`resting-state` or movie watching time series,
  we need to use the 'epi' strategy of the NiftiMasker.

In addition, we show here how to tweak the different parameters of the
underlying routine that extract masks from EPI
:func:`~nilearn.masking.compute_epi_mask`.

.. include:: ../../../examples/masker_note.rst

"""

# %%
# Computing a mask from the background
# ------------------------------------
#
# The default strategy to compute a mask, like the NiftiMasker,
# is to try to detect the background.
#
# With data that has already been masked this should work well,
# as it relies on a homogeneous background
#

# %%
# Fetch the dataset
# ^^^^^^^^^^^^^^^^^
#
# We fetch do some basic visualization of the image we will be using.
#
from nilearn.datasets import fetch_miyawaki2008
from nilearn.image import mean_img
from nilearn.plotting import plot_epi, show

miyawaki_dataset = fetch_miyawaki2008()
print(
    "First functional nifti image (4D) is located "
    f"at: {miyawaki_dataset.func[0]}"
)

miyawaki_filename = miyawaki_dataset.func[0]
miyawaki_mean_img = mean_img(miyawaki_filename)

plot_epi(miyawaki_mean_img, title="Mean EPI image")

show()

# %%
# A NiftiMasker with the default strategy
# ---------------------------------------
#
# Let's use the NiftiMasker with its defaults parameters.
from nilearn.maskers import NiftiMasker

masker = NiftiMasker(verbose=1)

# %%
#
# .. include:: ../../../examples/html_repr_note.rst
#
masker

# %%
masker.fit(miyawaki_filename)

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
# ^^^^^^^^^^^^^^^^^^
#
# We can quickly get an idea about the estimated mask
# for this functional image by plotting the mask.
#
# We get the estimated mask from the ``mask_img_`` attribute of the masker:
# the final ``_`` of this attribute name means it was generated
# by the :meth:`~nilearn.maskers.NiftiMasker.fit` method.
#
# We can then plot it using the :func:`~nilearn.plotting.plot_roi` function
# with the mean functional image as background.
#
from nilearn.plotting import plot_roi

plot_roi(
    masker.mask_img_, miyawaki_mean_img, title="Mask from already masked data"
)

# display the image
show()

# %%
# View the generated mask
# ^^^^^^^^^^^^^^^^^^^^^^^
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
# Computing a mask from raw :term:`EPI` data
# ------------------------------------------
#
# From raw :term:`EPI` data, there is no uniform background,
# and a different strategy is necessary
#

# %%
# Fetch the dataset
# ^^^^^^^^^^^^^^^^^
# Here we getch the movie watching based brain development fMRI dataset
# and once again do some basic visualization of the data.
#
# Here we only work with the first 100 volumes of the image
# to speed up computation.
from nilearn.datasets import fetch_development_fmri
from nilearn.image import index_img

dataset = fetch_development_fmri(n_subjects=1)
epi_filename = dataset.func[0]

epi_img = index_img(epi_filename, slice(0, 100))

mean_func_img = mean_img(epi_img)

plot_epi(mean_func_img, title="Mean EPI image")

show()

# %%
# Simple mask extraction from :term:`EPI` images
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We need to specify an ``'epi'`` mask_strategy,
# as this is raw :term:`EPI` data

masker = NiftiMasker(mask_strategy="epi", verbose=1)
masker.fit(epi_img)
report = masker.generate_report()
report

# %%
# Generate mask with strong opening
# ---------------------------------
#
# We can fine-tune the outline of the mask
# by increasing the number of opening steps (``opening=10``)
# using the ``mask_args`` argument of the NiftiMasker.
# This effectively performs :term:`erosion<Erosion>`
# and :term:`dilation<Dilation>` operations
# on the outer voxel layers of the mask,
# which can for example remove remaining skull parts in the image.

masker = NiftiMasker(mask_strategy="epi", mask_args={"opening": 10}, verbose=1)
masker.fit(epi_img)
report = masker.generate_report()
report

# %%
# Generate mask with a high lower cutoff
# --------------------------------------
#
# The NiftiMasker calls the :func:`nilearn.masking.compute_epi_mask` function
# to compute the mask from the EPI.
# It has two important parameters: lower_cutoff and upper_cutoff.
# These set the grey-value bounds
# in which the masking algorithm will search for its threshold
# (0 being the minimum of the image and 1 the maximum).
# We will here increase the lower cutoff
# to enforce selection of those voxels
# that appear as bright in the :term:`EPI` image.

masker = NiftiMasker(
    mask_strategy="epi",
    mask_args={"upper_cutoff": 0.9, "lower_cutoff": 0.8, "opening": False},
    verbose=1,
)
masker.fit(epi_img)
report = masker.generate_report()
report

# %%
# Computing the mask from the :term:`MNI` template
# ------------------------------------------------
#
# A mask can also be computed from the :term:`MNI` template.
# In this case, it is resampled to the target image.
# Three options are available:
# ``'whole-brain-template'``, ``'gm-template'``, and ``'wm-template'``
# depending on whether the whole-brain, gray matter,
# or white matter template should be used.

masker = NiftiMasker(mask_strategy="whole-brain-template", verbose=1)
masker.fit(epi_img)
report = masker.generate_report()
report

# %%
# Compute and resample a mask
# ---------------------------
#
# NiftiMasker also allows passing parameters directly
# to :func:`~nilearn.image.resample_img`.
# We can specify a ``target_affine``, a ``target_shape``, or both.
# For more information on these arguments,
# see :doc:`plot_affine_transformation`.
#
# The NiftiMasker report allows us to see the mask before and after resampling.
# Simply hover over the report to see the mask from the original image.
#

import numpy as np

masker = NiftiMasker(
    mask_strategy="epi", target_affine=np.eye(3) * 8, verbose=1
)
masker.fit(epi_img)
report = masker.generate_report()
report

# %%
# After mask computation: extracting time series
# ----------------------------------------------
#
# We extract time series detrended and non-detrended.
trended_data = NiftiMasker(mask_strategy="epi", verbose=1).fit_transform(
    epi_img
)
detrended_data = NiftiMasker(
    mask_strategy="epi", detrend=True, verbose=1
).fit_transform(epi_img)

# %%
# Once extracted,
# the timeseries are numpy arrays, so we can manipulate them with numpy
print(
    f"Trended: mean {np.mean(trended_data):.2f}, "
    f"std {np.std(trended_data):.2f}"
)
print(
    f"Detrended: mean {np.mean(detrended_data):.2f}, "
    f"std {np.std(detrended_data):.2f}"
)

# sphinx_gallery_dummy_images=2
