#! /usr/bin/env python
# TODO
# This follows the singlesubject/single (spm_auditory) session example

###############################################################################
#  Extract the signal from a voxel
#  -------------------------------
#
# We search for the voxel with the larger z-score and plot the signal
# (warning: this is "double dipping")


# Find the coordinates of the peak

from nibabel.affines import apply_affine
values = z_map.get_data()
coord_peaks = np.dstack(np.unravel_index(np.argsort(-values.ravel()),
                                         values.shape))[0, 0, :]
coord_mm = apply_affine(z_map.affine, coord_peaks)

###############################################################################
# We create a masker for the voxel (allowing us to detrend the signal)
# and extract the time course

from nilearn.input_data import NiftiSpheresMasker
mask = NiftiSpheresMasker([coord_mm], radius=3,
                          detrend=True, standardize=True,
                          high_pass=None, low_pass=None, t_r=7.)
sig = mask.fit_transform(fmri_img)

##########################################################
# Let's plot the signal and the theoretical response

plt.plot(frame_times, sig, label='voxel %d %d %d' % tuple(coord_mm))
plt.plot(design_matrix['active'], color='red', label='model')
plt.xlabel('scan')
plt.legend()
plt.show()
