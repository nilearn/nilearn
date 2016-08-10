"""
GLM fitting in second level fMRI
================================

Full step-by-step example of fitting a GLM to perform a second level analysis
in experimental data and visualizing the results.

More specifically:

1. A sequence of subject fMRI button press contrasts is downloaded.
2. a mask of the useful brain volume is computed
3. A GLM is applied to the dataset (as fixed effects, then contrast estimation)

Author : Martin Perez-Guevara: 2016
"""
print(__doc__)

import os

import numpy as np
import pandas as pd
from nilearn import plotting
from scipy.stats import norm
import matplotlib.pyplot as plt

from nilearn.datasets import fetch_localizer_contrasts
from nistats.second_level_model import SecondLevelModel

# Create writing directory
write_dir = 'results'
if not os.path.exists(write_dir):
    os.makedirs(write_dir)

# Get data ###################################################
n_subjects = 16
data = fetch_localizer_contrasts(["left vs right button press"], n_subjects,
                                 get_tmaps=True)

second_level_input = data['cmaps']
design_matrix = pd.DataFrame([1] * len(second_level_input),
                             columns=['left-right'])

# Display subject t_maps
subjects = [s[0] for s in data['ext_vars']]
fig, axes = plt.subplots(nrows=4, ncols=4)
for cidx, tmap in enumerate(data['tmaps']):
    plotting.plot_glass_brain(tmap, colorbar=False, threshold=2.0,
                              title=subjects[cidx],
                              axes=axes[cidx / 4, cidx % 4],
                              plot_abs=False, display_mode='z')
fig.suptitle('subjects t_map left-right button press')
fig.savefig(os.path.join(write_dir, 'left-right_all_subjects.png'))

# Estimate second level model
second_level_model = SecondLevelModel(smoothing_fwhm=8.0)
second_level_model = second_level_model.fit(second_level_input,
                                            design_matrix=design_matrix)

# Estimate contrast #########################################
z_map = second_level_model.compute_contrast('left-right',
                                            output_type='z_score')

# Display group level z_map thresholded for p < 0.001
p_val = 0.001
z_th = norm.isf(p_val)
display = plotting.plot_glass_brain(z_map, threshold=z_th, colorbar=True,
                                    plot_abs=False, display_mode='z')
display.savefig(os.path.join(write_dir, 'left-right_group.png'))

plotting.show()
