"""
Viewing a statistical map from multiple angles on a surface.
================================

Visualize the DMN from the Smith2009 canonical resting-state networks
on a surface.

See :ref:`surface-plotting` for surface plotting details.
"""
from nilearn.datasets import fetch_atlas_smith_2009
from nilearn.image import index_img
from nilearn.plotting import plot_surf_montage, show

###########################################################################
# Get the RSN10 and load the DMN, the RSN4.
# ----------------
rsn10 = fetch_atlas_smith_2009()['rsn10']
dmn_index = 3
dmn = index_img(rsn10, dmn_index)


###########################################################################
# Visualize the DMN in both hemispheres from the lateral and medial sides.
# ----------------
plot_surf_montage(dmn, display_mode='lateral+medial',
                  hemisphere='left+right',
                  vmax=7, colorbar=True)
show()
