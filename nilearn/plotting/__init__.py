"""
Plotting code for nilearn
"""
# Authors: Chris Filo Gorgolewski, Gael Varoquaux

from ..version import (_import_module_with_version_check,
                       OPTIONAL_MATPLOTLIB_MIN_VERSION)

###############################################################################
# Make sure that we don't get DISPLAY problems when running without X on
# unices


def _set_mpl_backend():
    try:
        # We are doing local imports here to avoid poluting our namespace
        import matplotlib
        import os
        # We have the problem only on posix systems
        if os.name == 'posix' and 'DISPLAY' not in os.environ:
            # Agg is a backend that will do PNGs and PDFs
            matplotlib.use('Agg')
    except ImportError:
        # No need to fail here, eg during tests
        pass
    else:
        # When matplotlib was successfully imported we need to check
        # that the version is greater that the minimum required one
        _import_module_with_version_check('matplotlib',
                                          OPTIONAL_MATPLOTLIB_MIN_VERSION)


_set_mpl_backend()

###############################################################################

from . import cm
from .img_plotting import plot_img, plot_anat, plot_epi, \
    plot_roi, plot_stat_map, plot_glass_brain, plot_connectome
from .find_cuts import find_xyz_cut_coords

__all__ = ['cm', 'plot_img', 'plot_anat', 'plot_epi',
           'plot_roi', 'plot_stat_map', 'plot_glass_brain',
           'plot_connectome',
           'find_xyz_cut_coords']
