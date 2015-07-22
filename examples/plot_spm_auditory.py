"""
:Module: plot_spm_auditory
Synopsis: Minimal script for preprocessing single-subject data
Author: Bertrand Thirion, dohmatob elvis dopgima, 2015

"""

import os
import numpy as np
import matplotlib.pyplot as plt
import nibabel
from pandas import DataFrame
from nistats.design_matrix import make_design_matrix
from nistats.glm import FMRILinearModel
from nistats.datasets import fetch_spm_auditory
from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img 

# fetch spm auditory data
subject_data = fetch_spm_auditory()
dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(
            subject_data.anat)))
output_dir = 'results'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# construct experimental paradigm
tr = 7.
n_scans = 96
_duration = 6
n_conditions = 2
epoch_duration = _duration * tr
conditions = ['rest', 'active'] * 8
duration = epoch_duration * np.ones(len(conditions))
onset = np.linspace(0, (len(conditions) - 1) * epoch_duration,
                    len(conditions))
paradigm = DataFrame(
    {'onset': onset, 'duration': duration, 'name': conditions})

# construct design matrix
nscans = len(subject_data.func)
frametimes = np.linspace(0, (nscans - 1) * tr, nscans)
drift_model = 'Cosine'
hrf_model = 'Canonical With Derivative'
hfcut = 2 * 2 * epoch_duration
design_matrix = make_design_matrix(
    frametimes, paradigm, hrf_model=hrf_model, drift_model=drift_model,
    hfcut=hfcut)

# plot and save design matrix
ax = design_matrix.show()
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')

# specify contrasts
contrasts = {}
n_columns = len(design_matrix.names)
for i in range(n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

# more interesting contrasts
contrasts['active-rest'] = contrasts['active'] - contrasts['rest']

# fit GLM
print('\r\nFitting a GLM (this takes time) ..')
fmri_glm = FMRILinearModel(nibabel.concat_images(subject_data.func),
                           design_matrix.matrix,
                           mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

# save computed mask
mask_path = os.path.join(output_dir, "mask.nii.gz")
print "Saving mask image %s" % mask_path
nibabel.save(fmri_glm.mask, mask_path)

# compute bg unto which activation will be projected
mean_img = mean_img(subject_data.func)

print "Computing contrasts .."
for contrast_id, contrast_val in contrasts.iteritems():
    print "\tcontrast id: %s" % contrast_id
    z_map, t_map, eff_map, var_map = fmri_glm.contrast(
        contrasts[contrast_id], con_id=contrast_id, output_z=True,
        output_stat=True, output_effects=True, output_variance=True)

    # store stat maps to disk
    for dtype, out_map in zip(['z', 't', 'effects', 'variance'],
                              [z_map, t_map, eff_map, var_map]):
        map_dir = os.path.join(output_dir, '%s_maps' % dtype)
        if not os.path.exists(map_dir):
            os.makedirs(map_dir)
        map_path = os.path.join(map_dir, '%s.nii.gz' % contrast_id)
        nibabel.save(out_map, map_path)
        print "\t\t%s map: %s" % (dtype, map_path)
    # plot activation map
    if contrast_id == 'active-rest':
        display = plot_stat_map(z_map, bg_img=mean_img, threshold=3.0,
                                display_mode='z', cut_coords=3, black_bg=True,
                                title=contrast_id)


plt.show()
