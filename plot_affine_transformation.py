"""
Visualization of affine resamplings
===================================

This example shows how an affine resampling works in voxel space.
"""


import matplotlib.pyplot as plt
import numpy as np
import nibabel
from nilearn.image import resample_img

# Make an image
grid = np.mgrid[0:192, 0:128]
circle = np.sum(
    (grid - np.array([32, 32])[:, np.newaxis, np.newaxis]) ** 2, axis=0) < 256
diamond = np.sum(np.abs(
        grid - np.array([128, 80])[:, np.newaxis, np.newaxis]), axis=0) < 16
rectangle = np.max(np.abs(
    grid - np.array([64, 96])[:, np.newaxis, np.newaxis]), axis=0) < 16

image = np.zeros_like(circle)
image[16:160, 16:120] = 1.
image = image + 2 * circle + 3 * rectangle + 4 * diamond + 1

vmax = image.max()

source_affine = np.eye(4)
# Use canonical vectors for affine
# Give the affine an offset
source_affine[:2, 3] = np.array([96, 64])

# Rotate it slightly
angle = np.pi / 180 * 15
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                            [np.sin(angle), np.cos(angle)]])
source_affine[:2, :2] = rotation_matrix * 2.0  # 2.0mm voxel size 

niimg = nibabel.Nifti1Image(image[:, :, np.newaxis], affine=source_affine)
niimg_in_mm_space = resample_img(niimg, target_affine=np.eye(4),
                                 target_shape=(512, 512, 1))

target_affine_3x3 = np.eye(3) * 2
target_affine_4x4 = np.eye(4) * 2
target_affine_4x4[3, 3] = 1.
niimg_3d_affine = resample_img(niimg, target_affine=target_affine_3x3)
niimg_4d_affine = resample_img(niimg, target_affine=target_affine_4x4)
target_affine_mm_space_offset_changed = np.eye(4)
target_affine_mm_space_offset_changed[:3, 3] = \
    niimg_3d_affine.get_affine()[:3, 3]

niimg_3d_affine_in_mm_space = resample_img(niimg_3d_affine,
    target_affine=target_affine_mm_space_offset_changed,
    target_shape=(np.array(niimg_3d_affine.shape) * 2).astype(int))

niimg_4d_affine_in_mm_space = resample_img(niimg_4d_affine,
    target_affine=np.eye(4),
    target_shape=(np.array(niimg_4d_affine.shape) * 2).astype(int))
plt.figure()
plt.imshow(image, interpolation="nearest", vmin=0, vmax=vmax)
plt.title("The actual data in voxel space")

plt.figure()
plt.imshow(niimg_in_mm_space.get_data()[:, :, 0], vmin=0, vmax=vmax)
plt.title("The actual data in mm space")

plt.figure()
plt.imshow(niimg_3d_affine_in_mm_space.get_data()[:, :, 0], vmin=0, vmax=vmax)
plt.title("Transformed using a 3x3 affine -\n leads to re-estimation of bounding box")

plt.figure()
plt.imshow(niimg_4d_affine_in_mm_space.get_data()[:, :, 0], vmin=0, vmax=vmax)
plt.title("Transformed using a 4x4 affine -\n Uses affine anchor and estimates bounding box size")

plt.show()
