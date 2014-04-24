"""
Voxel-Based Morphometry on Oasis dataset
========================================

Relationship between aging and gray matter density.

"""
# Authors: Elvis Dhomatob, <elvis.dohmatob@inria.fr>, Apr. 2014
#          Virgile Fritsch, <virgile.fritsch@inria.fr>, Apr 2014
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from nilearn import datasets
from nilearn.input_data import NiftiMasker

n_subjects = 50

### Load Oasis dataset ########################################################
dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float)

### Preprocess data ###########################################################
nifti_masker = NiftiMasker(
    standardize=False,
    smoothing_fwhm=5,
    memory='nilearn_cache',
    memory_level=1)  # cache options
# remove features with too low between-subject variance
gm_maps_masked = nifti_masker.fit_transform(dataset_files.gray_matter_maps)
gm_maps_masked[:, gm_maps_masked.var(0) < 0.01] = 0.
# final masking
new_images = nifti_masker.inverse_transform(gm_maps_masked)
gm_maps_masked = nifti_masker.fit_transform(new_images)
n_samples, n_features = gm_maps_masked.shape
print n_samples, "subjects, ", n_features, "features"

### Prediction function #######################################################
### Define the prediction function to be used.
# Here we use a Support Vector Classification, with a linear kernel
from sklearn.svm import SVR
svr = SVR(kernel='linear')

### Dimension reduction #######################################################
from sklearn.feature_selection import SelectKBest, f_classif

### Define the dimension reduction to be used.
# Here we use a classical univariate feature selection based on F-test,
# namely Anova. We set the number of features to be selected to 500
feature_selection = SelectKBest(f_classif, k=5000)

# We have our predictor (SVR), our feature selection (SelectKBest), and now,
# we can plug them together in a *pipeline* that performs the two operations
# successively:
from sklearn.pipeline import Pipeline
anova_svr = Pipeline([('anova', feature_selection), ('svr', svr)])

### Fit and predict ###########################################################
anova_svr.fit(gm_maps_masked, age)
age_pred = anova_svr.predict(gm_maps_masked)

### Visualisation #############################################################
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
mean_img = nibabel.load(dataset_files.gray_matter_maps[0]).get_data()
picked_slice = 36
plt.figure(figsize=(5, 4))
vmax = max(np.min(weights[:, :, picked_slice, 0]),
           np.max(weights[:, :, picked_slice, 0]))
plt.imshow(np.rot90(mean_img[:, :, picked_slice]), cmap=plt.cm.gray,
          interpolation='nearest')
im = plt.imshow(np.rot90(weights[:, :, picked_slice, 0]), cmap=plt.cm.RdBu_r,
                interpolation='nearest', vmin=-vmax, vmax=vmax)
plt.axis('off')
plt.colorbar(im)
plt.title('SVM weights')
plt.tight_layout()
plt.show()

### Cross validation ##########################################################
from sklearn.cross_validation import cross_val_score
cv_scores = cross_val_score(anova_svr, gm_maps_masked, age)

### Print results #############################################################

### Return the corresponding mean prediction accuracy
prediction_accuracy = np.mean(cv_scores)

### Printing the results
print "=== ANOVA ==="
print "Prediction accuracy: %f" % prediction_accuracy
