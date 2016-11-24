"""
Regions labeling and extraction using Yeo atlas
===============================================

This example shows how to use
:class:`nilearn.regions.extract_regions_labels_img`
to assigns each separated region of the atlas a unique label.

We use the Yeo atlas as an example for labeling regions,
:func:`nilearn.datasets.fetch_atlas_yeo_2011`

"""

################################################################################
# Fetching the Yeo atlas
from nilearn import datasets

atlas_yeo_2011 = datasets.fetch_atlas_yeo_2011()
atlas_yeo = atlas_yeo_2011.thick_7

################################################################################
# Labeling the atlas regions
from nilearn.regions import extract_regions_labels_img
region_labels = extract_regions_labels_img(atlas_yeo,
                                           min_size=None,
                                           connect_diag=True)

################################################################################
# Counting regions and plotting them

from nilearn.image import load_img
import numpy as np
atlas_yeo_img = load_img(atlas_yeo)
atlas_yeo_data = atlas_yeo_img.get_data()
region_labels_data = region_labels.get_data()

print("The regions of Yeo atlas: ", np.unique(atlas_yeo_data))
print("The labeled regions of Yeo atlas: ", np.unique(region_labels_data))

# Let's plot them
# The original Yeo atlas has 7 regions, that is indicated in the colorbar.
# The colorbar also shows the correspondence to the color and the cluster
from nilearn import plotting

# the original Yeo atlases
pl_yeo = plotting.plot_roi(atlas_yeo, title='Original Yeo atlas',
                           cut_coords=(8, -4, 9),
                           cmap = 'gist_rainbow', colorbar=True)
# labeled Yeo atlas
# Note: the same cluster in original and labeled atlas could have different
# color, so, you can not compare it.
# On the plots we observe the region that belonged to both hemispheres in
# original Yeo atlas and had the one color is slitted in more regions with
# different colors.
pl_yeo_labeled = plotting.plot_roi(region_labels, title='Labeled Yeo atlas',
                                   cut_coords=(8, -4, 9), cmap='gist_rainbow',
                                   colorbar=True)
plotting.show()

# You can save the images using savefig method.
# The images are saving to the current folder. It's possible to specify the
# folder for saving the results, i.e.
# import os
# pl_yeo.savefig(os.path.join(folder_path, 'Original_Yeo_atlas.png'))
pl_yeo.savefig('Original_Yeo_atlas.png')
pl_yeo.close()

pl_yeo_labeled.savefig('Labeled_Yeo_atlas.png')
pl_yeo_labeled.close()

################################################################################
# Labeling with different parameters
# Parameter connect_diag

# Parameter connect_diag=False separate two regions that are connected along
# the diagonal. In consequence, we can get a lot of small regions, to avoid it
# we suggest use connect_diag=True

region_labels_cd_false = extract_regions_labels_img(
    atlas_yeo,
    min_size=None,
    connect_diag=False)

region_labels_cd_false_data = region_labels_cd_false.get_data()

val_cd_false, size_cd_false = np.unique(region_labels_cd_false_data,
                                        return_counts=True)

# Measuring the size of each region of labeled atlas
# printing sorted dictionary of the region and its size
import operator
d_cd_false = dict(zip(val_cd_false[1:], size_cd_false[1:]))
d_cd_false_sorted = sorted(d_cd_false.items(), key=operator.itemgetter(1),
                           reverse=True)

print("The size of the labeled regions with connect_diag=False: ",
      d_cd_false_sorted)

################################################################################
# Parameter min_size

# We get 116 regions with 45 one-voxel regions and another smalls regions.
# min_size parameter helps us to filter these small regions

region_labels_min = extract_regions_labels_img(atlas_yeo, min_size=100,
                                               connect_diag=False)

region_labels_min_data = region_labels_min.get_data()

val_min, size_min = np.unique(region_labels_min_data, return_counts=True)

d_min = dict(zip(val_min[1:], size_min[1:]))
d_min_sorted = sorted(d_min.items(), key=operator.itemgetter(1), reverse=True)

print("The size of the labeled regions with connect_diag=False and min_size",
      d_min_sorted)
