"""
Example of a relatively simple GLM fit on the SPM auditory dataset
===================================================================

The script defines a General Linear Model and fits it to fMRI data.

Author: Bertrand Thirion, dohmatob elvis dopgima, 2015
"""

import os
import numpy as np
import pandas as pd

import nibabel as nib
from nibabel import Nifti1Image
from nilearn.plotting import plot_stat_map, show
from nilearn.image import mean_img

from nistats.first_level_model import FirstLevelModel
from nistats.datasets import fetch_spm_auditory

#########################################################################
# Prepare spm auditory data
# ----------------------------------
# Fetch the dataset
subject_data = fetch_spm_auditory()
fmri_img = nib.concat_images(subject_data.func)
fmri_img = Nifti1Image(fmri_img.get_data(), fmri_img.affine)
#########################################################################
# compute background image unto which activation will be projected
mean_img = mean_img(fmri_img)

#########################################################################
# Construct experimental paradigm
tr = 7.
slice_time_ref = 0.
n_scans = 96
epoch_duration = 6 * tr  # duration in seconds
conditions = ['rest', 'active'] * 8
n_blocks = len(conditions)
duration = epoch_duration * np.ones(n_blocks)
onset = np.linspace(0, (n_blocks - 1) * epoch_duration, n_blocks)
paradigm = pd.DataFrame(
    {'onset': onset, 'duration': duration, 'trial_type': conditions})

#########################################################################
# Construct design matrix
frame_times = np.linspace(0, (n_scans - 1) * tr, n_scans)
drift_model = 'Cosine'
period_cut = 4. * epoch_duration
hrf_model = 'glover + derivative'

#########################################################################
# Perform GLM analysis
# ------------------------------
# Fit GLM
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FirstLevelModel(tr, slice_time_ref, noise_model='ar1',
                           standardize=False, hrf_model=hrf_model,
                           drift_model=drift_model, period_cut=period_cut)
fmri_glm = fmri_glm.fit(fmri_img, paradigm)

#########################################################################
# We could easily specify basic contrasts (Betas of the GLM model)
design_matrix = fmri_glm.design_matrices_[0]
contrast_matrix = np.eye(design_matrix.shape[1])
contrasts = dict([(column, contrast_matrix[i])
                  for i, column in enumerate(design_matrix.columns)])

#########################################################################
# For this example instead lets specify one interesting contrast
contrasts = {'active-rest': contrasts['active'] - contrasts['rest']}

print("Computing contrasts ..")
output_dir = 'results'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

for contrast_id, contrast_val in contrasts.items():
    print("\tcontrast id: %s" % contrast_id)
    z_map = fmri_glm.compute_contrast(contrasts[contrast_id],
                                      output_type='z_score')
    t_map = fmri_glm.compute_contrast(contrasts[contrast_id],
                                      output_type='stat')
    eff_map = fmri_glm.compute_contrast(contrasts[contrast_id],
                                        output_type='effect_size')
    var_map = fmri_glm.compute_contrast(contrasts[contrast_id],
                                        output_type='effect_variance')
    #########################################################################
    # Store maps to disk
    for dtype, out_map in zip(['z', 't', 'effects', 'variance'],
                              [z_map, t_map, eff_map, var_map]):
        map_dir = os.path.join(output_dir, '%s_maps' % dtype)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(map_dir, '%s.nii.gz' % contrast_id)
        nib.save(out_map, map_path)
        print("\t\t%s map: %s" % (dtype, map_path))

    #########################################################################
    # Plot thresholded z scores map
    display = plot_stat_map(z_map, bg_img=mean_img, threshold=3.0,
                            display_mode='z', cut_coords=3, black_bg=True,
                            title=contrast_id)

    show()
