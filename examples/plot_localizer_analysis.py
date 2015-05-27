"""
GLM fitting in fMRI
===================

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
import matplotlib.pyplot as plt

from nibabel import save

from nistats.glm import FMRILinearModel
from nistats.design_matrix import make_dmtx
from nistats.experimental_paradigm import \
    load_paradigm_from_csv_file
from nistats import datasets

from nilearn import plotting


#######################################
# Data and analysis parameters
#######################################

# timing
n_scans = 128
tr = 2.4
# paradigm
frame_times = np.linspace(0.5 * tr, (n_scans - .5) * tr, n_scans)

# write directory
write_dir = 'results'
if not path.exists(write_dir):
    mkdir(write_dir)

data = datasets.fetch_localizer_first_level()
paradigm_file = data.paradigm
epi_img = data.epi_img

########################################
# Design matrix
########################################

paradigm = load_paradigm_from_csv_file(paradigm_file)['0']

design_matrix = make_dmtx(frame_times, paradigm,
                          hrf_model='canonical with derivative',
                          drift_model="cosine", hfcut=128)

# Plot the design matrix
ax = design_matrix.show()
ax.set_position([.05, .25, .9, .65])
ax.set_title('Design matrix')
plt.savefig(path.join(write_dir, 'design_matrix.png'))

########################################
# Perform a GLM analysis
########################################

fmri_glm = FMRILinearModel(epi_img,
                           design_matrix.matrix, mask='compute')
fmri_glm.fit(do_scaling=True, model='ar1')

#########################################
# Estimate contrasts
#########################################

# Specify the contrasts

# simplest ones
contrasts = {}
n_columns = len(design_matrix.names)
for i in range(paradigm.n_conditions):
    contrasts['%s' % design_matrix.names[2 * i]] = np.eye(n_columns)[2 * i]

# and more complex/ interesting ones
contrasts["audio"] = contrasts["clicDaudio"] + contrasts["clicGaudio"] +\
                     contrasts["calculaudio"] + contrasts["phraseaudio"]
contrasts["video"] = contrasts["clicDvideo"] + contrasts["clicGvideo"] + \
                     contrasts["calculvideo"] + contrasts["phrasevideo"]
contrasts["left-right"] = (contrasts["clicGaudio"] + contrasts["clicGvideo"]
                           - contrasts["clicDaudio"] - contrasts["clicDvideo"])
contrasts["computation"] = contrasts["calculaudio"] + contrasts["calculvideo"]
contrasts["sentences"] = contrasts["phraseaudio"] + contrasts["phrasevideo"]
contrasts["H-V"] = contrasts["damier_H"] - contrasts["damier_V"]
contrasts["V-H"] = contrasts["damier_V"] - contrasts["damier_H"]
contrasts["audio-video"] = contrasts["audio"] - contrasts["video"]
contrasts["video-audio"] = contrasts["video"] - contrasts["audio"]
contrasts["computation-sentences"] = contrasts["computation"] -  \
                                     contrasts["sentences"]
contrasts["reading-visual"] = contrasts["sentences"] * 2 - \
                              contrasts["damier_H"] - contrasts["damier_V"]


for index, (contrast_id, contrast_val) in enumerate(contrasts.items()):
    print('  Contrast % 2i out of %i: %s' %
          (index + 1, len(contrasts), contrast_id))
    # save the z_image
    image_path = path.join(write_dir, '%s_z_map.nii' % contrast_id)
    z_map, = fmri_glm.contrast(contrast_val, con_id=contrast_id, output_z=True)
    save(z_map, image_path)

    # Create snapshots of the contrasts
    vmax = max(-z_map.get_data().min(), z_map.get_data().max())
    plotting.plot_stat_map(z_map,
             display_mode='z', threshold=2.5, title=contrast_id)
    plt.savefig(path.join(write_dir, '%s_z_map.png' % contrast_id))


plt.show()
