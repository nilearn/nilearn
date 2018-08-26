"""
First level analysis of localizer dataset
=========================================

Full step-by-step example of fitting a GLM to experimental data and visualizing
the results.

More specifically:

1. A sequence of fMRI volumes are loaded
2. A design matrix describing all the effects related to the data is computed
3. a mask of the useful brain volume is computed
4. A GLM is applied to the dataset (effect/covariance,
   then contrast estimation)

"""
from os import mkdir, path

import numpy as np
import pandas as pd
import nilearn
import nistats

from nistats.first_level_model import FirstLevelModel


#########################################################################
# Prepare data and analysis parameters
# -------------------------------------
# Prepare timing
t_r = 2.4
slice_time_ref = 0.5

# Prepare data
from nistats.datasets import fetch_localizer_first_level
data = fetch_localizer_first_level()
paradigm_file = data.paradigm
paradigm = pd.read_csv(paradigm_file, sep=' ', header=None, index_col=None)
paradigm.columns = ['session', 'trial_type', 'onset']
fmri_img = data.epi_img

#########################################################################
# Project the fMRI image to the surface
# -------------------------------------

fsaverage = nilearn.datasets.fetch_surf_fsaverage5()
from nilearn import surface
texture = surface.vol_to_surf(fmri_img, fsaverage.pial_right)

#########################################################################
# Perform first level analysis
# ----------------------------
# Create design matrix
from nistats.design_matrix import make_design_matrix
frame_times = t_r * (np.arange(texture.shape[1]) + .5)
dmtx = make_design_matrix(
    frame_times, paradigm=paradigm, hrf_model='glover + derivative')

# Setup and fit GLM
from nistats.first_level_model import run_glm
labels, res = run_glm(texture.T, dmtx.values)

#########################################################################
# Estimate contrasts
# ------------------
# Specify the contrasts
contrast_matrix = np.eye(dmtx.shape[1])

# first create elementary contrasts
contrasts = dict([(column, contrast_matrix[i])
                  for i, column in enumerate(dmtx.columns)])

# create some intermediate contrasts 
contrasts["audio"] = contrasts["clicDaudio"] + contrasts["clicGaudio"] +\
    contrasts["calculaudio"] + contrasts["phraseaudio"]
contrasts["video"] = contrasts["clicDvideo"] + contrasts["clicGvideo"] + \
    contrasts["calculvideo"] + contrasts["phrasevideo"]
contrasts["computation"] = contrasts["calculaudio"] + contrasts["calculvideo"]
contrasts["sentences"] = contrasts["phraseaudio"] + contrasts["phrasevideo"]

#########################################################################
# Short list of more relevant contrasts
contrasts = {
    "left - right button press": (
        contrasts["clicGaudio"] + contrasts["clicGvideo"]
        - contrasts["clicDaudio"] - contrasts["clicDvideo"]),
    "horizontal - vertical checkerboard": (
        contrasts["damier_H"] - contrasts["damier_V"]),
    "audio - video": contrasts["audio"] - contrasts["video"],
    "computation - sentences": (contrasts["computation"] -
                                contrasts["sentences"])
    }

#########################################################################
# contrast estimation
from  nistats.contrasts import compute_contrast
from nilearn import plotting

for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    # compute contrasts
    contrast = compute_contrast(labels, res, contrast_val, contrast_type='t')
    z_score = contrast.z_score()

    plotting.plot_surf_stat_map(
        fsaverage.infl_right, z_score, hemi='right',
        title=contrast_id, colorbar=True,
        threshold=3., bg_map=fsaverage.sulc_right)

#########################################################################
# Analysing the left hemisphere
# Note that it requires little additional code
texture = surface.vol_to_surf(fmri_img, fsaverage.pial_left)
labels, res = run_glm(texture.T, dmtx.values)
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    # compute contrasts
    contrast = compute_contrast(labels, res, contrast_val, contrast_type='t')
    z_score = contrast.z_score()

    plotting.plot_surf_stat_map(
        fsaverage.infl_left, z_score, hemi='left',
        title=contrast_id, colorbar=True,
        threshold=3., bg_map=fsaverage.sulc_left)

plotting.show()
