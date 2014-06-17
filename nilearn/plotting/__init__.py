"""
Plotting code for nilearn
"""
# Authors: Chris Filo Gorgolewski, Gael Varoquaux

###############################################################################
# Make sure that we don't get DISPLAY problems when running without X on
# unices

def _set_mpl_backend():
    import os
    if not os.name == 'posix':
        # We have the problem only on posix systems
        return
    if 'DISPLAY' in os.environ:
        return
    import matplotlib
    # Agg is a backend that will do PNGs and PDFs
    matplotlib.use('Agg')

_set_mpl_backend()

###############################################################################

from . import cm
from .img_plotting import plot_img, plot_anat, demo_plot_roi, plot_epi, \
        plot_roi

__all__ = ['cm', 'plot_img', 'plot_anat', 'plot_roi', 'plot_epi',
           'demo_plot_roi']

