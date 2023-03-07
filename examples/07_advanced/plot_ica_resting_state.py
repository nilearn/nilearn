"""
Multivariate decompositions: Independent component analysis of fMRI
===================================================================

This example is meant to demonstrate nilearn as a low-level tools used to
combine feature extraction with a multivariate decomposition algorithm
for movie-watching.

This example is a toy. To apply ICA to fmri timeseries data, it is advised
to look at the example
:ref:`sphx_glr_auto_examples_03_connectivity_plot_compare_decomposition.py`.

The example here applies the scikit-learn :term:`ICA` to movie watching
timeseries data. Note that following the code in the example, any unsupervised
decomposition model, or other latent-factor models, can be applied to
the data, as the scikit-learn API enables to exchange them as almost
black box (though the relevant parameter for brain maps might no longer
be given by a call to fit_transform).

.. include:: ../../../examples/masker_note.rst

"""

#####################################################################
# Load movie watching dataset
from nilearn import datasets

# Here we use only single subject to get faster-running code.
dataset = datasets.fetch_development_fmri(n_subjects=1)
func_filename = dataset.func[0]

# print basic information on the dataset
print(f"First subject functional nifti image (4D) is at: {dataset.func[0]}")

#####################################################################
# Preprocess
from nilearn.maskers import NiftiMasker

# This is fmri timeseries data: the background has not been removed yet,
# thus we need to use mask_strategy='epi' to compute the mask from the
# EPI images
masker = NiftiMasker(
    smoothing_fwhm=8,
    memory="nilearn_cache",
    memory_level=1,
    mask_strategy="epi",
    standardize=True,
)
data_masked = masker.fit_transform(func_filename)

#####################################################################
# Apply ICA
from sklearn.decomposition import FastICA

n_components = 10
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(data_masked.T).T

# Normalize estimated components, for thresholding to make sense
components_masked -= components_masked.mean(axis=0)
components_masked /= components_masked.std(axis=0)

# Threshold
import numpy as np

components_masked[np.abs(components_masked) < 0.8] = 0

# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)

#####################################################################
# Visualize the results
from nilearn import image
from nilearn.plotting import plot_stat_map, show

# Show some interesting components

# Use the mean as a background
mean_img = image.mean_img(func_filename)

plot_stat_map(image.index_img(component_img, 0), mean_img)

plot_stat_map(image.index_img(component_img, 1), mean_img)

show()
