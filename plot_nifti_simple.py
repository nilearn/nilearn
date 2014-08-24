"""
Simple example of NiftiMasker use
==================================

Here is a simple example of automatic mask computation using the nifti masker.
The mask is computed and visualized.
"""

### Load nyu_rest dataset #####################################################

from nilearn import datasets
from nilearn.input_data import NiftiMasker
dataset = datasets.fetch_nyu_rest(n_subjects=1)

### Compute the mask ##########################################################

# As this is raw resting-state EPI, the background is noisy and we cannot
# rely on the 'background' masking strategy. We need to use the 'epi' one
nifti_masker = NiftiMasker(standardize=False, mask_strategy='epi',
                           memory="nilearn_cache", memory_level=2)
nifti_masker.fit(dataset.func[0])
mask_img = nifti_masker.mask_img_

### Visualize the mask ########################################################
import matplotlib.pyplot as plt
from nilearn.plotting import plot_roi
from nilearn.image.image import mean_img

# calculate mean image for the background
mean_func_img = mean_img(dataset.func[0])

plot_roi(mask_img, mean_func_img, display_mode='y', cut_coords=4, title="Mask")


### Preprocess data ###########################################################
nifti_masker.fit(dataset.func[0])
fmri_masked = nifti_masker.transform(dataset.func[0])

### Run an algorithm ##########################################################
from sklearn.decomposition import FastICA
n_components = 20
ica = FastICA(n_components=n_components, random_state=42)
components_masked = ica.fit_transform(fmri_masked.T).T

### Reverse masking ###########################################################
components = nifti_masker.inverse_transform(components_masked)

### Show results ##############################################################
import nibabel
from nilearn.plotting import plot_stat_map

plot_stat_map(nibabel.Nifti1Image(components.get_data()[:,:,:,0],
                                  components.get_affine()), mean_func_img,
              display_mode='y', cut_coords=4, title="Component 0")

plt.show()
