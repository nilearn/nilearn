"""
Negate image
============

Here we compute a negative image by multiplying it's voxel values with -1.
"""

from nilearn import datasets, plotting, image

################################################################################
# Fetching AAL atlas regions by loading from datasets.
data = datasets.fetch_atlas_aal()

################################################################################
# Print basic information on the AAL regions.
print('AAL regions nifti image (3D) is located at: %s' % data.regions)

################################################################################
# Multiply voxel values by -1.
result_img = image.math_img("-img", img=data.regions)

plotting.plot_roi(data.regions, cmap='Blues', title="AAL regions")
plotting.plot_roi(result_img, cmap='Blues', title="Negative of AAL regions")
plotting.show()
