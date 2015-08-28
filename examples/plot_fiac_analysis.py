"""
GLM fitting in fMRI
===================

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

print(__doc__)

from os import mkdir, path, getcwd

import numpy as np
import matplotlib.pyplot as plt

from nilearn.plotting import plot_stat_map
from nilearn.image import mean_img
import nibabel as nib

from nistats.glm import FirstLevelGLM
from nistats import datasets


# write directory
write_dir = path.join(getcwd(), 'results')
if not path.exists(write_dir):
    mkdir(write_dir)

# Data and analysis parameters
data = datasets.fetch_fiac_first_level()
fmri_files = [data['func1'], data['func2']]
design_files = [data['design_matrix1'], data['design_matrix2']]

# Load all the data into a common GLM
multi_session_model = FirstLevelGLM(data['mask'], standardize=False,
                                    noise_model='ar1')

# GLM fitting
multi_session_model.fit(fmri_files, design_files)

def make_fiac_contrasts(n_columns):
    """ Specify some contrasts for the FIAC experiment"""
    contrast = {}
    # the design matrices of both runs comprise 13 columns
    # the first 5 columns of the design matrices correspond to the following
    # conditions: ['SSt-SSp', 'SSt-DSp', 'DSt-SSp', 'DSt-DSp', 'FirstSt']

    def _pad_vector(contrast_, n_columns):
        return np.hstack((contrast_, np.zeros(n_columns - len(contrast_))))

    contrast['SStSSp_minus_DStDSp'] = _pad_vector([1, 0, 0, -1], n_columns)
    contrast['DStDSp_minus_SStSSp'] = - contrast['SStSSp_minus_DStDSp']
    contrast['DSt_minus_SSt'] = _pad_vector([- 1, - 1, 1, 1], n_columns)
    contrast['DSp_minus_SSp'] = _pad_vector([- 1, 1, - 1, 1], n_columns)
    contrast['DSt_minus_SSt_for_DSp'] = _pad_vector([0, - 1, 0, 1], n_columns)
    contrast['DSp_minus_SSp_for_DSt'] = _pad_vector([0, 0, - 1, 1], n_columns)
    contrast['Deactivation'] = _pad_vector([- 1, - 1, - 1, - 1, 4], n_columns)
    contrast['Effects_of_interest'] = np.eye(n_columns)[:5]
    return contrast

# compute fixed effects of the two runs and compute related images
n_columns = np.load(design_files[0])['X'].shape[1]
contrasts = make_fiac_contrasts(n_columns)

print('Computing contrasts...')
mean_ = mean_img(data['func1'])
for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' % (
        index + 1, len(contrasts), contrast_id))
    z_image_path = path.join(write_dir, '%s_z_map.nii' % contrast_id)
    z_map, = multi_session_model.transform(
        [contrast_val] * 2, contrast_name=contrast_id, output_z=True)
    nib.save(z_map, z_image_path)

    # make a snapshot of the contrast activation
    if contrast_id == 'Effects_of_interest':
        vmax = max(- z_map.get_data().min(), z_map.get_data().max())
        vmin = - vmax
        display = plot_stat_map(z_map, bg_img=mean_, threshold=2.5,
                                title=contrast_id)
        display.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))

print('All the  results were witten in %s' % write_dir)
plt.show()
