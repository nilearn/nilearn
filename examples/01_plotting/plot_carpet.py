"""
Visualizing global patterns with a carpet plot
==============================================

A common quality control step for functional MRI data is to visualize the data
over time in a carpet plot (also known as a Power plot or a grayplot).

The :func:`nilearn.plotting.plot_carpet()` function generates a carpet plot
from a 4D functional image.
"""

###############################################################################
# Fetching data from ADHD dataset
# -------------------------------
from nilearn import datasets

adhd_dataset = datasets.fetch_adhd(n_subjects=1)

# Print basic information on the dataset
print('First subject functional nifti image (4D) is at: %s' %
      adhd_dataset.func[0])  # 4D data

###############################################################################
# Deriving a mask
# ---------------
from nilearn import masking

# Build an EPI-based mask because we have no anatomical data
mask_img = masking.compute_epi_mask(adhd_dataset.func[0])

###############################################################################
# Visualizing global patterns over time
# -------------------------------------
import matplotlib.pyplot as plt

from nilearn.plotting import plot_carpet

display = plot_carpet(adhd_dataset.func[0], mask_img)

display.show()
