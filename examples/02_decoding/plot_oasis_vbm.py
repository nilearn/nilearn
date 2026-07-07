"""
Voxel-Based Morphometry on Oasis dataset
========================================

This example uses Voxel-Based Morphometry (:term:`VBM`)
to study the relationship between aging and gray matter density.

The data come from the Open Access Series of Imaging Studies
(`OASIS <https://sites.wustl.edu/oasisbrains/>`_) project.
If you use it, you need to agree with the data usage agreement available
on the website.

It has been run through a standard :term:`VBM` pipeline (using SPM8 and
NewSegment) to create :term:`VBM` maps, which we study here.

Predictive modeling analysis: VBM bio-markers of aging?
-------------------------------------------------------

We run a standard SVM-ANOVA nilearn pipeline to predict age from the VBM data.
We use a subset of the subjects from the OASIS dataset
to limit the memory usage.

.. admonition:: Important

    Note that for an actual predictive modeling study of aging,
    the study should be ran on the full set of subjects.

    Also, all parameters should be set by cross-validation.
    This includes the smoothing applied to the data and the
    number of features selected by the :term:`ANOVA` step.
    Indeed, even these data-preparation parameter
    impact significantly the prediction score.

    Also, parameters such as the smoothing should be applied
    to the data and the number of features selected by the :term:`ANOVA` step
    should be set by nested cross-validation,
    as they impact significantly the prediction score.


.. seealso::

    For more information
    see the :ref:`dataset description <oasis_maps>`.

____
"""

# Use a single variable to control the verbosity of the script.
verbose = 0

# Several of Nilearn's estimators (like the DecoderRegressor we use here)
# accept a ``n_jobs=<some_high_value>``
# to take advantage of a multi-core system.
n_jobs = 2

# %%
# Load Oasis dataset
# ------------------
# We fetch the data and split it into training set and test set.

from sklearn.model_selection import train_test_split

from nilearn.datasets import fetch_oasis_vbm

n_subjects = 200  # more subjects requires more memory

oasis_dataset = fetch_oasis_vbm(n_subjects=n_subjects, verbose=verbose)

print(
    "First gray-matter anatomy image (3D) is located at: "
    f"{oasis_dataset.gray_matter_maps[0]}"
)

gray_matter_map_filenames = oasis_dataset.gray_matter_maps
age = oasis_dataset.ext_vars["age"].to_numpy()
gm_imgs_train, gm_imgs_test, age_train, age_test = train_test_split(
    gray_matter_map_filenames, age, train_size=0.6, random_state=0
)


# %%
# Preprocess data
# ---------------
# The voxels with too low between-subject variance are removed using
# :class:`sklearn.feature_selection.VarianceThreshold`.
#
# Then we convert the data back to the mask image
# in order to use it for decoding process.
#
from sklearn.feature_selection import VarianceThreshold

from nilearn.maskers import NiftiMasker

nifti_masker = NiftiMasker(
    standardize=None,
    smoothing_fwhm=2,
    memory="nilearn_cache",  # cache options
    verbose=verbose,
)
gm_maps_masked = nifti_masker.fit_transform(gm_imgs_train)

variance_threshold = VarianceThreshold(threshold=0.01)
variance_threshold.fit_transform(gm_maps_masked)

mask = nifti_masker.inverse_transform(variance_threshold.get_support())


# %%
# Prediction pipeline with ANOVA and SVR using DecoderRegressor
# -------------------------------------------------------------
#
# In Nilearn we can benefit from the built-in DecoderRegressor object
# to do an :term:`ANOVA` with SVR
# instead of manually defining the whole pipeline.
#
# This estimator also uses cross validation to select best models
# and ensemble them.
#
# To save time (because these are anat images with many voxels),
# we include only the 1-percent voxels most correlated
# with the age variable to fit.
#
# We also want to set mask hyperparameter
# to be the mask we just obtained above.
#
# We then fit and predict with the decoder
# and sort test data for better visualization (trend, etc.).
import numpy as np

from nilearn.decoding import DecoderRegressor

decoder = DecoderRegressor(
    mask=mask,
    scoring="neg_mean_absolute_error",
    screening_percentile=1,
    n_jobs=n_jobs,
    verbose=verbose,
)
decoder.fit(gm_imgs_train, age_train)

perm = np.argsort(age_test)[::-1]
age_test = age_test[perm]
gm_imgs_test = np.array(gm_imgs_test)[perm]
age_pred = decoder.predict(gm_imgs_test)

prediction_score = -np.mean(decoder.cv_scores_["beta"])

print(f"explained variance for the cross-validation: {prediction_score:f}")


# %%
# Visualization
# -------------
import matplotlib.pyplot as plt

from nilearn.image import mean_img
from nilearn.plotting import plot_stat_map, show

bg_filename = mean_img(gray_matter_map_filenames)
z_slice = 0

weight_img = decoder.coef_img_["beta"]

display = plot_stat_map(
    weight_img,
    title="SVM weights",
    bg_img=bg_filename,
    display_mode="z",
    cut_coords=[z_slice],
    figure=plt.figure(figsize=(5.5, 7.5), facecolor="k"),
)
show()

# %%
# Visualize the quality of predictions
# ------------------------------------
linewidth = 3

plt.figure(figsize=(6, 4.5))
plt.suptitle(f"Decoder: Mean Absolute Error {prediction_score:.2f} years")
plt.plot(age_test, label="True age", linewidth=linewidth)
plt.plot(age_pred, "--", c="g", label="Predicted age", linewidth=linewidth)
plt.ylabel("age")
plt.xlabel("subject")
plt.legend(loc="best")

plt.figure(figsize=(6, 4.5))
plt.plot(
    age_test - age_pred, label="True age - predicted age", linewidth=linewidth
)
plt.xlabel("subject")
plt.legend(loc="best")

show()


# %%
# Brain mapping with mass univariate
# ----------------------------------
#
# :term:`SVM` weights are very noisy,
# partly because heavy smoothing is detrimental for the prediction here.
# A standard analysis using mass-univariate :term:`GLM`
# (here permuted to have exact correction for multiple comparisons)
# gives a much clearer view of the important regions.
#
from nilearn.image import get_data
from nilearn.mass_univariate import permuted_ols

gm_maps_masked = NiftiMasker(
    standardize=None,
    memory="nilearn_cache",  # cache options
    verbose=verbose,
).fit_transform(gray_matter_map_filenames)
data = variance_threshold.fit_transform(gm_maps_masked)

output = permuted_ols(
    age,
    data,  # + intercept as a covariate by default
    n_perm=2000,  # 10 000 would be better
    verbose=verbose,
    n_jobs=n_jobs,
)

# %%
# Show results
neg_log_pvals = output["logp_max_t"]
t_scores_original_data = output["t"]
signed_neg_log_pvals = neg_log_pvals * np.sign(t_scores_original_data)
signed_neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    variance_threshold.inverse_transform(signed_neg_log_pvals)
)
threshold = -np.log10(0.1)  # 10% corrected

n_detections = (get_data(signed_neg_log_pvals_unmasked) > threshold).sum()

title = (
    "Negative $\\log_{10}$ p-values"
    "\n(Non-parametric + max-type correction)"
    f"\n{int(n_detections)} detections"
)

plot_stat_map(
    signed_neg_log_pvals_unmasked,
    threshold=threshold,
    title=title,
    bg_img=bg_filename,
    display_mode="z",
    cut_coords=[z_slice],
    figure=plt.figure(figsize=(5.5, 7.5), facecolor="k"),
)
show()

# sphinx_gallery_dummy_images=2
