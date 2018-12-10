"""
Multivariate decompositions: Independent component analysis of fMRI
===================================================================


This example is meant to demonstrate nilearn as a low-level tools used to
combine feature extraction with a multivariate decomposition algorithm
for resting state.

This example is a toy. To apply ICA to resting-state data, it is advised
to look at the example
:ref:`sphx_glr_auto_examples_03_connectivity_plot_canica_resting_state.py`.

The example here applies the scikit-learn ICA to resting-state data.
Note that following the code in the example, any unsupervised
decomposition model, or other latent-factor models, can be applied to
the data, as the scikit-learn API enables to exchange them as almost
black box (though the relevant parameter for brain maps might no longer
be given by a call to fit_transform).

"""

#####################################################################
# Load ADHD dataset
from nilearn import datasets
# Here we use only 3 subjects to get faster-running code. For better
# results, simply increase this number
# XXX: must get the code to run for more than 1 subject
dataset = datasets.fetch_adhd(n_subjects=1)
func_filename = dataset.func[0]

# print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      dataset.func[0])  # 4D data


#####################################################################
# Preprocess
from nilearn.input_data import NiftiMasker

# This is resting-state data: the background has not been removed yet,
# thus we need to use mask_strategy='epi' to compute the mask from the
# EPI images
masker = NiftiMasker(smoothing_fwhm=8, memory='nilearn_cache', memory_level=1,
                     mask_strategy='epi', standardize=True)
data_masked = masker.fit_transform(func_filename)

# Concatenate all the subjects
# fmri_data = np.concatenate(data_masked, axis=1)
fmri_data = data_masked


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
components_masked[np.abs(components_masked) < .8] = 0

# Now invert the masking operation, going back to a full 3D
# representation
component_img = masker.inverse_transform(components_masked)

#####################################################################
# Visualize the results

# Show some interesting components
from nilearn import image
from nilearn.plotting import plot_stat_map, show

# Use the mean as a background
mean_img = image.mean_img(func_filename)

plot_stat_map(image.index_img(component_img, 0), mean_img)

plot_stat_map(image.index_img(component_img, 1), mean_img)

show()
