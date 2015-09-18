# This example shows how to use a 3D statistical map to
# automatically segmented each map into each a seperated brain region.
# For this, we adapted a two step procedure which is summarized below.

# We use localizer t statistic maps which are in 3D and make use of
# an in-built private functions (which can also be adapted to public)
# to first threshold the t statistic images using a function
# "apply_threshold_to_maps" and then to fed them to region extraction
# function denoted as "extract_regions".

# This example motivates to seperate the regions and achieves to focus
# on a particular target of interest to learn functional connectomics
# within those target regions across multiple subjects. For no complexity,
# this example shows only for single subject.

# Load localizer datasets - contrast/t maps
from nilearn import datasets
from nilearn._utils import check_niimg_3d
from nilearn.image import index_img
n_subjects = 1
localizer_path = datasets.fetch_localizer_contrasts(
    ['calculation (auditory cue)'], n_subjects=n_subjects, get_tmaps=True)
localizer_img = check_niimg_3d(localizer_path.tmaps[0])
localizer_data = localizer_img.get_data()

# Mask the data
from nilearn.input_data import NiftiMasker
nifti_masker = NiftiMasker(memory='nilearn_cache',
                           memory_level=1)
masked_data = nifti_masker.fit_transform(localizer_img)
mask_img = nifti_masker.mask_img_

# Step 1: Threshold
from nilearn.region_decomposition import region_extractor
from nilearn.image import new_img_like
# Initialize the parameters to threshold function to keep
# only the interesting features
threshold_value = 0.1
threshold_strategy = 'voxelratio'
map_threshold = region_extractor.apply_threshold_to_maps(
    masked_data, threshold_value, threshold_strategy)
map_thresholded_img = nifti_masker.inverse_transform(map_threshold)
# squeeze the image to 3D
data = map_thresholded_img.get_data()
affine = map_thresholded_img.get_affine()
map_thresholded_img = new_img_like(map_thresholded_img, data[:, :, :, 0], affine)

# Step 2: Region Extraction
# Initialize the parameters
min_size = 200
regions = region_extractor.extract_regions(
    map_thresholded_img, min_size, extract_type='auto',
    smooth_fwhm=6)
n_regions = len(regions)

# Visualization
import matplotlib.pyplot as plt
from nilearn import plotting
# Visualize input t map
plotting.plot_stat_map(localizer_img, title='Input_data: Statistical t-map')
# Visualize thresholded t-map
plotting.plot_stat_map(map_thresholded_img,
                       title='Statistical t-map after threshold')
# Visualize region extraction images
title = 'Regions extracted of the same input t-map'
for index in range(n_regions):
    plotting.plot_stat_map(regions[index],
                           title=title)

plt.show()
