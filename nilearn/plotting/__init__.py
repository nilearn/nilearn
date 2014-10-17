"""
Plotting code for nilearn
"""
# Authors: Chris Filo Gorgolewski, Gael Varoquaux

###############################################################################
# Make sure that we don't get DISPLAY problems when running without X on
# unices

def _set_mpl_backend():
    # We are doing local imports here to avoid poluting our namespace
    import os
    if not os.name == 'posix':
        # We have the problem only on posix systems
        return
    if 'DISPLAY' in os.environ:
        return

    try:
        import matplotlib
        # Agg is a backend that will do PNGs and PDFs
        matplotlib.use('Agg')
    except ImportError:
        pass
        # No need to fail here, eg during tests


_set_mpl_backend()

###############################################################################

from . import cm
from .img_plotting import plot_img, plot_anat, plot_epi, \
    plot_roi, plot_stat_map, plot_glass_brain
from .find_cuts import find_xyz_cut_coords

__all__ = ['cm', 'plot_img', 'plot_anat', 'plot_epi',
           'plot_roi', 'plot_stat_map', 'plot_glass_brain',
           'find_xyz_cut_coords']
