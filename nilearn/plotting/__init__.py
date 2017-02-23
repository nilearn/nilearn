"""
Plotting code for nilearn
"""
# Authors: Chris Filo Gorgolewski, Gael Varoquaux

###############################################################################
# Make sure that we don't get DISPLAY problems when running without X on
# unices
def _set_mpl_backend():
    try:
        # We are doing local imports here to avoid poluting our namespace
        import matplotlib
        import os
        import sys
        # Set the backend to a non-interactive one for unices without X
        if (os.name == 'posix' and 'DISPLAY' not in os.environ
            and not (sys.platform == 'darwin'
                     and matplotlib.get_backend() == 'MacOSX'
                     )):
            matplotlib.use('Agg')
    except ImportError:
        from .._utils.testing import skip_if_running_nose
        # No need to fail when running tests
        skip_if_running_nose('matplotlib not installed')
        raise
    else:
        from ..version import (_import_module_with_version_check,
                               OPTIONAL_MATPLOTLIB_MIN_VERSION)
        # When matplotlib was successfully imported we need to check
        # that the version is greater that the minimum required one
        _import_module_with_version_check('matplotlib',
                                          OPTIONAL_MATPLOTLIB_MIN_VERSION)

_set_mpl_backend()

###############################################################################
import matplotlib
from distutils.version import LooseVersion

from . import cm
from .img_plotting import plot_img, plot_anat, plot_epi, \
    plot_roi, plot_stat_map, plot_glass_brain, plot_connectome, \
    plot_prob_atlas, show
from .surf_plotting import plot_surf, plot_surf_stat_map, plot_surf_roi
from .find_cuts import find_xyz_cut_coords, find_cut_slices

__all__ = ['cm', 'plot_img', 'plot_anat', 'plot_epi',
           'plot_roi', 'plot_stat_map', 'plot_glass_brain',
           'plot_connectome', 'plot_prob_atlas',
           'find_xyz_cut_coords', 'find_cut_slices',
           'show']

# matplotlib older versions 1.1.1 will not work to plot surface data using
# plotting functions from surf_plotting.py. Hence we check the version and
# import them only if we have recent versions.

if LooseVersion(matplotlib.__version__) > LooseVersion('1.3.1'):
    from .surf_plotting import plot_surf, plot_surf_stat_map, plot_surf_roi
    __all__.extend(['plot_surf', 'plot_surf_stat_map', 'plot_surf_roi'])
