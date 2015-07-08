"""
Group analysis of resting-state fMRI with dictionary learning: DictLearning
=====================================================

An example applying dictionary learning to resting-state data. This example applies it
to 10 subjects of the ADHD200 datasets.

Dictionary learning is a sparsity based decomposition method for extracting spatial maps.

    * Gael Varoquaux et al.
    Multi-subject dictionary learning to segment an atlas of brain spontaneous activity
    Information Processing in Medical Imaging, 2011, pp. 562-573, Lecture Notes in Computer Science

Pre-prints for paper is available on hal
(http://hal.archives-ouvertes.fr)
"""

import numpy as np

### Load ADHD rest dataset ####################################################
from nilearn import datasets

# OUTPUT DIR
import os
import datetime

output_dir = os.path.expanduser('~/work/output/nilearn/plot_dict_learning_resting_state')
output_dir = os.path.join(output_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
try:
    os.makedirs(output_dir)
except OSError:
    pass

adhd_dataset = datasets.fetch_adhd(n_subjects=10)
func_filenames = adhd_dataset.func  # list of 4D nifti files for each subject

# print basic information on the dataset
print('First functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

### Apply DictLearning ########################################################
from nilearn.decomposition.dict_learning import DictLearning
n_components = 50

dict_learning = DictLearning(n_components=n_components, smoothing_fwhm=6.,
                             memory="nilearn_cache", memory_level=5,
                             threshold=1., verbose=2, random_state=0,
                             n_jobs=1, n_init=1, alpha=3, n_iter=1000)

dict_learning.fit(func_filenames)

print('')
print('[Example] Dumping results')

# Retrieve the independent components in brain space
components_img = dict_learning.masker_.inverse_transform(dict_learning.components_)
# components_img is a Nifti Image object, and can be saved to a file with
# the following line:
components_img.to_filename('dict_learning_resting_state.nii.gz')

### Visualize the results #####################################################
# Show some interesting components
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn.image import iter_img

for i, cur_img in enumerate(iter_img(components_img)):
    if i % 10 == 0:
        plot_stat_map(cur_img, title="Component %d" % i,
                      colorbar=False)

plt.show()