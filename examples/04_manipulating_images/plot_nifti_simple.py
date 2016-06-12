"""
Simple example of NiftiMasker use
==================================

Here is a simple example of automatic mask computation using the nifti masker.
The mask is computed and visualized.
"""

###########################################################################
# Retrieve the NYU test-retest dataset

from nilearn import datasets
dataset = datasets.fetch_adhd(n_subjects=1)
func_filename = dataset.func[0]

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' % func_filename)

###########################################################################
# Compute the mask
from nilearn.input_data import NiftiMasker

# As this is raw resting-state EPI, the background is noisy and we cannot
# rely on the 'background' masking strategy. We need to use the 'epi' one
nifti_masker = NiftiMasker(standardize=True, mask_strategy='epi',
                           memory="nilearn_cache", memory_level=2,
                           smoothing_fwhm=8)
nifti_masker.fit(func_filename)
mask_img = nifti_masker.mask_img_

###########################################################################
# Visualize the mask
from nilearn.plotting import plot_roi
from nilearn.image.image import mean_img

# calculate mean image for the background
mean_func_img = mean_img(func_filename)

plot_roi(mask_img, mean_func_img, display_mode='y', cut_coords=4, title="Mask")


###########################################################################
# Preprocess data with the NiftiMasker
nifti_masker.fit(func_filename)
fmri_masked = nifti_masker.transform(func_filename)
# fmri_masked is now a 2D matrix, (n_voxels x n_time_points)

###########################################################################
# Run an algorithm
from sklearn.decomposition import FastICA
n_components = 10
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(fmri_masked.T).T

###########################################################################
# Reverse masking, and display the corresponding map
components = nifti_masker.inverse_transform(components_masked)

# Visualize results
from nilearn.plotting import plot_stat_map, show
from nilearn.image import index_img

plot_stat_map(index_img(components, 0), mean_func_img,
              display_mode='y', cut_coords=4, title="Component 0")

show()
