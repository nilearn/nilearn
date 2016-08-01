"""
GLM fitting in second level fMRI
================================

Full step-by-step example of fitting a GLM to perform a second level analysis
in experimental data and visualizing the results.

More specifically:

1. A sequence of fMRI volumes are loaded
2. A design matrix describing all the effects related to the data is computed
3. a mask of the useful brain volume is computed
4. A GLM is applied to the dataset (effect/covariance,
   then contrast estimation)

Author : Martin Perez-Guevara: 2016
"""
print(__doc__)

import os
from os import mkdir, path

import numpy as np
import pandas as pd
from nilearn import plotting
import nibabel as nib

from nilearn.datasets import fetch_mixed_gambles
from nistats.second_level_model import SecondLevelModel

# write directory
write_dir = 'results'
first_level_paths = os.path.join('results', 'first_level_imgs')
if not path.exists(first_level_paths):
    os.makedirs(first_level_paths)

# get data ###################################################
data = fetch_mixed_gambles(16)
zmap_paths = []
for zidx, zmap in enumerate(data.zmaps):
    zmap_path = os.path.join(first_level_paths, 'map_%03d.nii' % zidx)
    nib.save(zmap, zmap_path)
    zmap_paths.append(zmap_path)
behavioral_target = np.ravel(['gain_%d' % (i + 1) for i in data.gain]).tolist()
subjects_id = np.ravel([['sub_%02d' % (i + 1)] * 48 for i in range(16)])
subjects_id = subjects_id.tolist()
mask_filename = data.mask_img


# Second level model #########################################
# create first_level_input
df_columns = [subjects_id, behavioral_target, zmap_paths]
df_columns = zip(*df_columns)
df_column_names = ['model_id', 'map_name', 'effects_map_path']
first_level_df = pd.DataFrame(df_columns, columns=df_column_names)

# estimate second level model
second_level_model = SecondLevelModel(mask=mask_filename, smoothing_fwhm=3.0)
second_level_model = second_level_model.fit(first_level_df)

# Estimate contrasts #########################################
# Specify the contrasts
design_matrix = second_level_model.design_matrix_
contrast_matrix = np.eye(design_matrix.shape[1])
contrasts = [(column, contrast_matrix[i])
             for i, column in enumerate(design_matrix.columns)]

# contrast estimation
for index, (contrast_id, contrast_val) in enumerate(contrasts[:4]):
    print('  Contrast % 2i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    z_map = second_level_model.compute_contrast(contrast_val,
                                                output_type='z_score')

    # Create snapshots of the contrasts
    display = plotting.plot_stat_map(z_map, display_mode='z',
                                     threshold=3.0, title=contrast_id)
    display.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))
    display.close()

plotting.show()
