"""
Massively univariate analysis of a calculation task from the Localizer dataset
==============================================================================

This example shows how to use the Localizer dataset in a basic analysis.
A standard Anova is performed (massively univariate F-test) and the resulting
Bonferroni-corrected p-values are plotted.
We use a calculation task and 20 subjects out of the 94 available.

The Localizer dataset contains many contrasts and subject-related
variates.  The user can refer to the
`plot_localizer_mass_univariate_methods.py` example to see how to use these.


"""
# Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, May. 2014
import numpy as np
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.input_data import NiftiMasker

### Load Localizer contrast ###################################################
n_samples = 20
dataset_files = datasets.fetch_localizer_calculation_task(n_subjects=n_samples)
tested_var = np.ones((n_samples, 1))

### Mask data #################################################################
nifti_masker = NiftiMasker(
    smoothing_fwhm=5,
    memory='nilearn_cache', memory_level=1)  # cache options
fmri_masked = nifti_masker.fit_transform(dataset_files.cmaps)

### Anova (parametric F-scores) ###############################################
from nilearn._utils.fixes import f_regression
_, pvals_anova = f_regression(fmri_masked, tested_var,
                              center=False)  # do not remove intercept
pvals_anova *= fmri_masked.shape[1]
pvals_anova[np.isnan(pvals_anova)] = 1
pvals_anova[pvals_anova > 1] = 1
neg_log_pvals_anova = - np.log10(pvals_anova)
neg_log_pvals_anova_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals_anova)

### Visualization #############################################################

# Various plotting parameters
picked_slice = 32  # plotted slice
vmin = - np.log10(0.1)  # 10% corrected
vmax = np.amax(neg_log_pvals_anova)

# Here, we should use a structural image as a background, when available.

# Plot Anova p-values
plt.figure(figsize=(5.5, 5.5))
masked_pvals = np.ma.masked_less(neg_log_pvals_anova_unmasked.get_data(), vmin)
plt.imshow(np.rot90(nifti_masker.mask_img_.get_data()[:, :, picked_slice]),
           interpolation='nearest', cmap=plt.cm.gray)
im = plt.imshow(np.rot90(masked_pvals[:, :, picked_slice]),
                interpolation='nearest', cmap=plt.cm.autumn,
                vmin=vmin, vmax=vmax)
plt.axis('off')
plt.colorbar(im)
plt.title(r'Negative $\log_{10}$ p-values'
          + '\n(Parametric + Bonferroni correction)'
          + '\n%d detections' % (~masked_pvals.mask[..., picked_slice]).sum())

plt.show()
