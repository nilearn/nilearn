import numpy as np

import matplotlib.pyplot as plt

import nibabel

from nilearn.plotting import slicers
from nilearn.plotting import img_plotting, cm
from nilearn import datasets

localizer = datasets.fetch_localizer_contrasts(["left vs right button press"],
                                               n_subjects=4,
                                               get_anats=True)
img = nibabel.load(localizer.cmaps[3])
data = img.get_data()
data[np.isnan(data)] = 0

bg_img, black_bg, vmin, vmax = img_plotting._load_anat()
slicer = img_plotting.plot_glass_brain(bg_img, vmin=3000, vmax=2*310983,
                                       threshold=0, title='anat')
slicer = img_plotting.plot_glass_brain(bg_img, vmin=3000, vmax=2*310983,
                                       threshold=0, black_bg=False, title='anat')

slicer = img_plotting.plot_glass_brain(img, colorbar=True, cmap=plt.cm.Reds,
                                       title='default threshold')
slicer = img_plotting.plot_glass_brain(img, colorbar=True, cmap=plt.cm.Reds, black_bg=False,
                                       title='default threshold')

slicer = img_plotting.plot_glass_brain(img, threshold=10, colorbar=True, cmap=plt.cm.Greens,
                                       title='threshold=10')
slicer = img_plotting.plot_glass_brain(img, threshold=10, colorbar=True, cmap=plt.cm.Greens, black_bg=False,
                                       title='threshold=10')

plt.show()
