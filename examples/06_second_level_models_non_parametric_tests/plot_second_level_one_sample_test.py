"""
Second-level fMRI model: one sample test
========================================

Full step-by-step example of fitting a GLM to perform a second-level analysis (one-sample test)
and visualizing the results.

More specifically:

1. A sequence of subject fMRI button press contrasts is downloaded.
2. a mask of the useful brain volume is computed
3. A one-sample t-test is applied to the brain maps

We focus on a given contrast of the localizer dataset: the motor response to left versus right button press. Both at the ndividual and group level, this is expected to elicit activity in the motor cortex (positive in the right hemisphere, negative in the left hemisphere).

"""

import numpy as np

#########################################################################
# Fetch dataset
# --------------
# We download a list of left vs right button press contrasts from a
# localizer dataset. Note that we fetc individual t-maps that represent the Bold activity estimate divided by the uncertainty about this estimate.
from nilearn.datasets import fetch_localizer_contrasts
n_subjects = 16
data = fetch_localizer_contrasts(["left vs right button press"], n_subjects,
                                 get_tmaps=True)

###########################################################################
# Display subject t_maps
# ----------------------
# We plot a grid with all the subjects t-maps thresholded at t = 2 for
# simple visualization purposes. The button press effect is visible among
# all subjects
from nilearn import plotting
import matplotlib.pyplot as plt
subjects = [subject_data[0] for subject_data in data['ext_vars']]
fig, axes = plt.subplots(nrows=4, ncols=4)
for cidx, tmap in enumerate(data['tmaps']):
    plotting.plot_glass_brain(tmap, colorbar=False, threshold=2.0,
                              title=subjects[cidx],
                              axes=axes[int(cidx / 4), int(cidx % 4)],
                              plot_abs=False, display_mode='z')
fig.suptitle('subjects t_map left-right button press')
plt.show()

############################################################################
# Estimate second level model
# ---------------------------
# We define the input maps and the design matrix for the second level model
# and fit it.
import pandas as pd
second_level_input = data['cmaps']
design_matrix = pd.DataFrame([1] * len(second_level_input),
                             columns=['intercept'])

############################################################################
# Model specification and fit
from nistats.second_level_model import SecondLevelModel
second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
second_level_model = second_level_model.fit(second_level_input,
                                            design_matrix=design_matrix)

##########################################################################
# To estimate the contrast is very simple. We can just provide the column
# name of the design matrix.
z_map = second_level_model.compute_contrast(output_type='z_score')

##########################################################################
from nilearn.input_data import NiftiMasker
from scipy.stats import norm
masker = NiftiMasker(smoothing_fwhm=8.0).fit(z_map)
stats = np.ravel(masker.transform(z_map))
n_voxels = np.size(stats)
pvals = 2 * norm.sf(np.abs(stats))
pvals_corr = np.minimum(1, pvals * n_voxels)
neg_log_pvals = - np.log10(pvals_corr)
neg_log_pvals_unmasked = masker.inverse_transform(neg_log_pvals)

###########################################################################
# We threshold the second level contrast at uncorrected p < 0.001 and plot
from scipy.stats import norm
cut_coords = [0]
display = plotting.plot_glass_brain(
    neg_log_pvals_unmasked, colorbar=True, display_mode='z', plot_abs=False,
    cut_coords=cut_coords)
plotting.show()

###########################################################################
from nistats.second_level_model import non_parametric_inference
neg_log_pvals_permuted_ols_unmasked = \
    non_parametric_inference(second_level_input,
                             design_matrix=design_matrix,
                             model_intercept=True, n_perm=1000,
                             two_sided_test=True,
                             smoothing_fwhm=8.0, n_jobs=1)

###########################################################################
#Let us plot the second level contrast
from nilearn import plotting
display = plotting.plot_glass_brain(
    neg_log_pvals_permuted_ols_unmasked, colorbar=True,
    display_mode='z', plot_abs=False,
    cut_coords=cut_coords)
plotting.show()
