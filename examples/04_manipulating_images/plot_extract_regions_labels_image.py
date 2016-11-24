"""
Regions labeling and extraction using Yeo atlas
=============================================================

This example shows how to use :class:nilearn.regions.extract_regions_labels_img
to assigns each separated region of the atlas a unique label.

We use the Yeo atlas as an example for labeling regiones,
:func:nilearn.datasets.fetch_atlas_yeo_2011

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
# Counting regions and potting them
import nibabel as nb
import numpy as np
atlas_yeo_img = nb.load(atlas_yeo)
atlas_yeo_data = atlas_yeo_img.get_data()
region_labels_data = region_labels.get_data()

print("The regions of Yeo atlas: ", np.unique(atlas_yeo_data))
print("The labeled regions of Yeo atlas: ", np.unique(region_labels_data))

# Let's plot them
# The original Yeo atlas has 7 regions, that is indicated in the colorbar.
# The colorbar also shows the corespondence to the color and the cluster
from nilearn import plotting

# the original Yeo atlases
pl_yeo = plotting.plot_roi(atlas_yeo, title='Original Yeo atlas',
                           cut_coords=(8,-4,9),
                           cmap = 'gist_rainbow', colorbar=True)
plotting.show()

# labeled Yeo atlas
# Note: the same cluster in original and labeled atlas could have different
# color, so, you can not compare it.
# On the plost we observe the region that belonged to both hemispheres in
# original Yeo atlas and had the one color is splitted i more regios with
# different colors.
pl_yeo_labeled = plotting.plot_roi(region_labels, title='Labeled Yeo atlas',
                                   cut_coords=(8,-4,9), cmap = 'gist_rainbow',
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
d_min_sorted = sorted(d_min.items(), key=operator.itemgetter(1),
                           reverse=True)

print("The size of the labeled regions with connect_diag=False and min_size",
      d_min_sorted)










#
# ################################################################################
# # Plotting
# pl_yeo_labeled = plotting.plot_roi(region_labels_cd_false,
#                                    title='Labeled Yeo atlas with connect_diag=False',
#                                    cut_coords=(8,-4,9), cmap = 'gist_rainbow',
#                                    colorbar=True)
# plotting.show()
#
# pl_yeo_labeled = plotting.plot_roi(region_labels_min,
#                                    title='Labeled Yeo atlas with min_size',
#                                    cut_coords=(8,-4,9), cmap = 'gist_rainbow',
#                                    colorbar=True)
# plotting.show()
#
#
# ################################################################################
# # saving
# import os
# folder_results = '/home/darya/Documents/Inria/experiments/nilearn_example_extract_regions_labels'
# region_labels_cd_false.to_filename(os.path.join(folder_results, 'Yeo_region_labels_cd_false.nii'))
#
# region_labels_min.to_filename(os.path.join(folder_results, 'Yeo_region_labels_min.nii'))
#
#
# from scipy import ndimage
#
# a = np.array([0,0,1,1,1,2,2,2,2,13,13])
# #a = np.ones(a.shape)
# alab = [0,0,1,1,1,2,2,2,2,13,13]
# aindex = [0,1,2,13]
#
# a_size = ndimage.measurements.sum(a, labels=alab, index=aindex)
# b_size = ndimage.measurements.sum(b, labels=alab, index=aindex)
# _ , aa_size = np.unique(a, return_counts=True)
#
# a = np.array([0,0,1,1,1,2,2,2,2,13,13])
# cc = _remove_small_regions(a, alab, aindex, 2, search_sorted=False)
# cc
#
# input_data = np.array([0,0,1,1,1,2,2,2,2,13,13])
# mask_data = np.array([0,0,1,1,1,2,2,2,2,13,13])
# index = [0,1,2,13]
# min_size = 2
#
# a = np.ones(a.shape)
# b=[1,1,1,1,1,1,1,1,1,1,1]
#
#
# region_sizes = ndimage.measurements.sum(region_labels_min100_data,
#                                         labels=region_labels_min100_data)
#
#
#
#
# ################################################################################
# # Plotting
#
# from nilearn import plotting
#
# atlases = [atlas_yeo, region_labels, region_labels_connect_diag_false,
#            region_labels_min200]
# atlases_titles = ["Yeo_atlas", "Relabeled_Yeo_atlas",
#                   "Relabeled_Yeo_atlas_connect_diag_false",
#                   "Relabeled_Yeo_atlas_min_size200"]
#
# import matplotlib.pylab as plt
# figure, axes = plt.subplots(len(atlases), 1)
# for axle, atlas, title in zip(axes, atlases, atlases_titles):
#     plotting.plot_roi(atlas, title=title, cut_coords=(8,-4,9), axes=axle,
#                       cmap = 'gist_rainbow', colorbar=True)
#
# plt.show()