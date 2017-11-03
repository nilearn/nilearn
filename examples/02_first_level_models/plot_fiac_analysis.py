"""
Simple example of GLM fitting in fMRI
======================================

Full step-by-step example of fitting a GLM to experimental data and visualizing
the results. This is done on two runs of one subject of the FIAC dataset.
For details on the data, please see:

Dehaene-Lambertz G, Dehaene S, Anton JL, Campagne A, Ciuciu P, Dehaene
G, Denghien I, Jobert A, LeBihan D, Sigman M, Pallier C, Poline
JB. Functional segregation of cortical language areas by sentence
repetition. Hum Brain Mapp. 2006: 27:360--371.
http://www.pubmedcentral.nih.gov/articlerender.fcgi?artid=2653076#R11

More specifically:

1. A sequence of fMRI volumes are loaded
2. A design matrix describing all the effects related to the data is computed
3. a mask of the useful brain volume is computed
4. A GLM is applied to the dataset (effect/covariance,
   then contrast estimation)

Author: Bertrand Thirion, 2015
"""
from os import mkdir, path, getcwd

import numpy as np
import pandas as pd

from nilearn import plotting
from nilearn.image import mean_img

from nistats.first_level_model import FirstLevelModel
from nistats import datasets


# write directory
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

#########################################################################
# Prepare data and analysis parameters
# --------------------------------------
data = datasets.fetch_fiac_first_level()
fmri_img = [data['func1'], data['func2']]
mean_img_ = mean_img(fmri_img[0])
design_files = [data['design_matrix1'], data['design_matrix2']]
design_matrices = [pd.DataFrame(np.load(df)['X']) for df in design_files]

#########################################################################
# GLM estimation
# ----------------------------------
# GLM specification
fmri_glm = FirstLevelModel(mask=data['mask'], minimize_memory=True)

#########################################################################
# GLM fitting
fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices)

#########################################################################
# compute fixed effects of the two runs and compute related images
n_columns = design_matrices[0].shape[1]


def pad_vector(contrast_, n_columns):
    return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))


contrasts = {'SStSSp_minus_DStDSp': pad_vector([1, 0, 0, -1], n_columns),
             'DStDSp_minus_SStSSp': pad_vector([-1, 0, 0, 1], n_columns),
             'DSt_minus_SSt': pad_vector([-1, -1, 1, 1], n_columns),
             'DSp_minus_SSp': pad_vector([-1, 1, -1, 1], n_columns),
             'DSt_minus_SSt_for_DSp': pad_vector([0, -1, 0, 1], n_columns),
             'DSp_minus_SSp_for_DSt': pad_vector([0, 0, -1, 1], n_columns),
             'Deactivation': pad_vector([-1, -1, -1, -1, 4], n_columns),
             'Effects_of_interest': np.eye(n_columns)[:5]}

print('Computing contrasts...')
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' % (
        index + 1, len(contrasts), contrast_id))
    z_image_path = path.join(write_dir, '%s_z_map.nii' % contrast_id)
    z_map = fmri_glm.compute_contrast(
        contrast_val, output_type='z_score')
    z_map.to_filename(z_image_path)

    # make a snapshot of the contrast activation
    if contrast_id == 'Effects_of_interest':
        display = plotting.plot_stat_map(
            z_map, bg_img=mean_img_, threshold=2.5, title=contrast_id)
        display.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))

print('All the  results were witten in %s' % write_dir)
plotting.show()
