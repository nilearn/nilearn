"""
Multiply image
==============

Here we multiply the voxel values of an image by -1.
"""

from nilearn import datasets, plotting, image

data = datasets.fetch_atlas_aal()

# Print basic information on the AAL regions.
print('AAL regions nifti image (3D) is located at: %s' % data.regions)

# Multiply voxel values by -1.
formula = "np.dot(img, -1)"

result_img = image.math_img(formula, img=data.regions)

plotting.plot_epi(result_img, title="AAL regions multiplied by -1.")
plotting.show()
