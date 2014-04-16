"""
Voxel-Based Morphometry on Oasis dataset.
Relationship between aging and gray matter density.

"""
# Authors: Elvis Dhomatob, <elvis.dohmatob@inria.fr>, Apr. 2014
#          Virgile Fritsch, <virgile.fritsch@inria.fr>, Apr 2014
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from nilearn import datasets
from nilearn.input_data import NiftiMasker
from nilearn.mass_univariate import permuted_ols

n_subjects = 50

### Get data
dataset_files = datasets.fetch_oasis_vbm(n_subjects=n_subjects)
age = dataset_files.ext_vars['age'].astype(float).reshape((-1, 1))

### Mask data
nifti_masker = NiftiMasker(
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

### Perform massively univariate analysis with permuted OLS ###################
print "Massively univariate model"
neg_log_pvals, all_scores, _ = permuted_ols(
    age, gm_maps_masked,  # + intercept as a covariate by default
    n_perm=10000,
    n_jobs=1)  # can be changed to use more CPUs
neg_log_pvals_unmasked = nifti_masker.inverse_transform(
    neg_log_pvals).get_data()[..., 0]

### Show results
print "Plotting results"
# background anat
mean_anat = nibabel.load(dataset_files.gray_matter_maps[0]).get_data()
for img in dataset_files.gray_matter_maps[1:]:
    mean_anat += nibabel.load(img).get_data()
mean_anat /= float(len(dataset_files.gray_matter_maps))
picked_slice = 36
vmin = -np.log10(0.1)  # 10% corrected
plt.figure(figsize=(5, 4))
masked_pvals = np.ma.masked_less(neg_log_pvals_unmasked, vmin)
plt.imshow(np.rot90(mean_anat[..., picked_slice]),
           interpolation='nearest', cmap=plt.cm.gray, vmin=0., vmax=1.)
im = plt.imshow(np.rot90(masked_pvals[..., picked_slice]),
                interpolation='nearest', cmap=plt.cm.autumn,
                vmin=vmin, vmax=np.amax(neg_log_pvals_unmasked))
plt.axis('off')
plt.colorbar(im)
plt.subplots_adjust(0., .02, .98, .98)

plt.show()
