"""
Plotting code for nilearn
"""
# Original Authors: Chris Filo Gorgolewski, Gael Varoquaux
import os
import sys
import importlib


###############################################################################
# Make sure that we don't get DISPLAY problems when running without X on
# unices
def _set_mpl_backend():
    # We are doing local imports here to avoid polluting our namespace
    try:
        import matplotlib
    except ImportError:
        if importlib.util.find_spec("pytest") is not None:
            from .._utils.testing import skip_if_running_tests
            # No need to fail when running tests
            skip_if_running_tests('matplotlib not installed')
        raise
    else:
        from ..version import (_import_module_with_version_check,
                               OPTIONAL_MATPLOTLIB_MIN_VERSION)
        # When matplotlib was successfully imported we need to check
        # that the version is greater that the minimum required one
        _import_module_with_version_check('matplotlib',
                                          OPTIONAL_MATPLOTLIB_MIN_VERSION)
        current_backend = matplotlib.get_backend().lower()

        if 'inline' in current_backend or 'nbagg' in current_backend:
            return
        # Set the backend to a non-interactive one for unices without X
        # (see gh-2560)
        if (sys.platform not in ('darwin', 'win32') and
                'DISPLAY' not in os.environ):
            matplotlib.use('Agg')


_set_mpl_backend()

###############################################################################
from . import cm
from .img_plotting import (
    plot_img, plot_anat, plot_epi, plot_roi, plot_stat_map,
    plot_glass_brain, plot_connectome, plot_connectome_strength,
    plot_markers, plot_prob_atlas, plot_carpet, plot_img_comparison, show)
from .find_cuts import find_xyz_cut_coords, find_cut_slices, \
    find_parcellation_cut_coords, find_probabilistic_atlas_cut_coords
from .matrix_plotting import (plot_matrix, plot_contrast_matrix,
                              plot_design_matrix, plot_event)
from .html_surface import view_surf, view_img_on_surf
from .html_stat_map import view_img
from .html_connectome import view_connectome, view_markers
from .surf_plotting import (plot_surf, plot_surf_stat_map, plot_surf_roi,
                            plot_img_on_surf, plot_surf_contours)

__all__ = ['cm', 'plot_img', 'plot_anat', 'plot_epi',
           'plot_roi', 'plot_stat_map', 'plot_glass_brain',
           'plot_markers', 'plot_connectome', 'plot_prob_atlas',
           'find_xyz_cut_coords', 'find_cut_slices',
           'plot_img_comparison',
           'show', 'plot_matrix',
           'plot_design_matrix', 'plot_contrast_matrix', 'plot_event',
           'view_surf', 'view_img_on_surf',
           'view_img', 'view_connectome', 'view_markers',
           'find_parcellation_cut_coords',
           'find_probabilistic_atlas_cut_coords',
           'plot_surf', 'plot_surf_stat_map', 'plot_surf_roi',
           'plot_img_on_surf', 'plot_connectome_strength', 'plot_carpet',
           'plot_surf_contours']
