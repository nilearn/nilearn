"""
Voxel-Based Morphometry on Oasis dataset
========================================

This example used Voxel-Based Morphometry (VBM) to study the relationship
between aging and gray matter density.

The data comes from the `OASIS <http://www.oasis-brains.org/>`_ project.
If you use it, you need to agree with the terms of use on the website.

It has been run through a standard VBM pipeline (using SPM8 and
NewSegment) to create VBM maps, which we study here.

Predictive modeling analysis: VBM bio-markers of aging?
--------------------------------------------------------

We run a standard SVM-ANOVA nilearn pipeline to predict age from the VBM
data. We use only 100 subjects from the OASIS dataset to limit the memory
usage.

Note that for an actual predictive modeling study of aging, the study
should be ran on the full set of subjects. Also, parameters such as the
smoothing applied to the data and the number of features selected by the
Anova step should be set by nested cross-validation, as they impact
significantly the prediction score.

Brain mapping with mass univariate
-----------------------------------

SVM weights are very noisy, partly because heavy smoothing is detrimental
for the prediction here. A standard analysis using mass-univariate GLM
(here permute to have exact correction for multiple comparisons) gives a
much clearer view of the important regions.

____

"""
# Authors: Elvis Dhomatob, <elvis.dohmatob@inria.fr>, Apr. 2014
#          Virgile Fritsch, <virgile.fritsch@inria.fr>, Apr 2014
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from nilearn import datasets
from nilearn.input_data import NiftiMasker

n_subjects = 100  # more subjects requires more memory

### Load Oasis dataset ########################################################
dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)

### Preprocess data ###########################################################
nifti_masker = NiftiMasker(
    standardize=False,
    smoothing_fwhm=2,
    memory='nilearn_cache')  # cache options
# remove features with too low between-subject variance
gm_maps_masked = nifti_masker.fit_transform(dataset_files.gray_matter_maps)
gm_maps_masked[:, gm_maps_masked.var(0) < 0.01] = 0.
# final masking
new_images = nifti_masker.inverse_transform(gm_maps_masked)
gm_maps_masked = nifti_masker.fit_transform(new_images)
n_samples, n_features = gm_maps_masked.shape
print n_samples, "subjects, ", n_features, "features"

### Prediction with SVR #######################################################
print "ANOVA + SVR"
### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel
from sklearn.svm import SVR
svr = SVR(kernel='linear')

### Dimension reduction
from sklearn.feature_selection import SelectKBest, f_regression

# Here we use a classical univariate feature selection based on F-test,
# namely Anova.
feature_selection = SelectKBest(f_regression, k=2000)

# We have our predictor (SVR), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svr = Pipeline([('anova', feature_selection), ('svr', svr)])

### Fit and predict
anova_svr.fit(gm_maps_masked, age)
age_pred = anova_svr.predict(gm_maps_masked)

### Visualisation
### Look at the SVR's discriminating weights
coef = svr.coef_
# reverse feature selection
coef = feature_selection.inverse_transform(coef)
# reverse masking
weight_niimg = nifti_masker.inverse_transform(coef)

# We use a masked array so that the voxels at '-1' are displayed
# transparently
weights = np.ma.masked_array(weight_niimg.get_data(),
                             weight_niimg.get_data() == 0)

### Create the figure
background_img = nibabel.load(dataset_files.gray_matter_maps[0]).get_data()
picked_slice = 36
plt.figure(figsize=(5, 5))
data_for_plot = weights[:, :, picked_slice, 0]
vmax = max(np.min(data_for_plot), np.max(data_for_plot)) * 0.5
plt.imshow(np.rot90(background_img[:, :, picked_slice]), cmap=plt.cm.gray,
          interpolation='nearest')
im = plt.imshow(np.rot90(data_for_plot), cmap=plt.cm.Spectral_r,
                interpolation='nearest', vmin=-vmax, vmax=vmax)
plt.axis('off')
plt.colorbar(im)
plt.title('SVM weights')
plt.tight_layout()

### Measure accuracy with cross validation
from sklearn.cross_validation import cross_val_score
cv_scores = cross_val_score(anova_svr, gm_maps_masked, age)

### Return the corresponding mean prediction accuracy
prediction_accuracy = np.mean(cv_scores)
print "=== ANOVA ==="
print "Prediction accuracy: %f" % prediction_accuracy
print

### Inference with massively univariate model #################################
print "Massively univariate model"

### Statistical inference
from nilearn.mass_univariate import permuted_ols
neg_log_pvals, all_scores, _ = permuted_ols(
    age, gm_maps_masked,  # + intercept as a covariate by default
    n_perm=5000,  # In the interest of time; 10000 would be better
    n_jobs=1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals).get_data()[..., 0]

### Show results
# background anat
plt.figure(figsize=(5, 5))
vmin = -np.log10(0.1)  # 10% corrected
masked_pvals = np.ma.masked_less(neg_log_pvals_unmasked, vmin)
print '\n%d detections' % (~masked_pvals.mask[..., picked_slice]).sum()
plt.imshow(np.rot90(background_img[:, :, picked_slice]),
           interpolation='nearest', cmap=plt.cm.gray, vmin=0., vmax=1.)
im = plt.imshow(np.rot90(masked_pvals[:, :, picked_slice]),
                interpolation='nearest', cmap=plt.cm.autumn,
                vmin=vmin, vmax=np.amax(neg_log_pvals_unmasked))
plt.axis('off')
plt.colorbar(im)
plt.title(r'Negative $\log_{10}$ p-values'
          + '\n(Non-parametric + max-type correction)\n')
plt.tight_layout()

plt.show()
