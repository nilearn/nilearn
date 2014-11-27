import numpy as np

import matplotlib.pyplot as plt

import nibabel

from nilearn.plotting import img_plotting
from nilearn import datasets

localizer = datasets.fetch_localizer_contrasts(["left vs right button press"],
                                               n_subjects=4,
                                               get_anats=True,
                                               get_tmaps=True)
img = nibabel.load(localizer.tmaps[3])
data = img.get_data()
data[np.isnan(data)] = 0

bg_img, _, _, _ = img_plotting._load_anat()
display = img_plotting.plot_glass_brain(bg_img, threshold=0, black_bg=True,
                                        title='anat', alpha=0.7)
display = img_plotting.plot_glass_brain(bg_img, threshold=0,
                                        title='anat', alpha=0.7)

display = img_plotting.plot_glass_brain(img, black_bg=True,
                                        title='default threshold', alpha=0.7)
display = img_plotting.plot_glass_brain(img,
                                        title='default threshold', alpha=0.7)

display = img_plotting.plot_glass_brain(img, threshold=3, black_bg=True,
                                        title='threshold=3', alpha=0.7)
display = img_plotting.plot_glass_brain(img, threshold=3,
                                        title='threshold=3', alpha=0.7)

plt.show()
