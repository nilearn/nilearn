"""Test plotting asymmetric color bar."""

import matplotlib.pyplot as plt
import numpy as np

from nilearn import datasets, plotting, surface
from nilearn.image import threshold_img

stat_img = datasets.load_sample_motor_activation_image()
stat_img = threshold_img(
    stat_img,
    threshold=2,
    cluster_threshold=0,
    two_sided=False,
    copy=True,
)

fsaverage = datasets.fetch_surf_fsaverage()

curv_right = surface.load_surf_data(fsaverage.curv_right)
curv_right_sign = np.sign(curv_right)

texture = surface.vol_to_surf(stat_img, fsaverage.pial_right)

engine = "matplotlib"

plotting.plot_surf_stat_map(
    fsaverage.infl_right,
    texture,
    hemi="right",
    title="Surface right hemisphere",
    colorbar=True,
    threshold=1.0,
    bg_map=curv_right_sign,
    engine=engine,
    symmetric_cbar=False,
    cmap="black_red",
)
plt.show()

engine = "plotly"

print(f"Using plotting engine {engine}.")

fig = plotting.plot_surf_stat_map(
    fsaverage.infl_right,
    texture,
    hemi="right",
    title="Surface right hemisphere",
    colorbar=True,
    threshold=1.0,
    bg_map=curv_right_sign,
    bg_on_data=True,
    engine=engine,
    symmetric_cbar=False,
    cmap="black_red",
)
fig.show()
