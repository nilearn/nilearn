"""
Plotting code for nilearn
"""
# Authors: Chris Filo Gorgolewski, Gael Varoquaux

from . import cm
from .img_plotting import plot_img, plot_anat, demo_plot_roi, plot_epi, \
        plot_roi

__all__ = ['cm', 'plot_img', 'plot_anat', 'plot_roi', 'plot_epi',
           'demo_plot_roi']

