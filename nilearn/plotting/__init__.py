"""Plotting code for nilearn."""

# Original Authors: Chris Filo Gorgolewski, Gael Varoquaux
from nilearn._utils.helpers import _set_mpl_backend

_set_mpl_backend()

###############################################################################
from . import cm
from .find_cuts import (
    find_cut_slices,
    find_parcellation_cut_coords,
    find_probabilistic_atlas_cut_coords,
    find_xyz_cut_coords,
)
from .html_connectome import view_connectome, view_markers
from .html_stat_map import view_img
from .html_surface import view_img_on_surf, view_surf
from .img_plotting import (
    plot_anat,
    plot_carpet,
    plot_connectome,
    plot_epi,
    plot_glass_brain,
    plot_img,
    plot_img_comparison,
    plot_markers,
    plot_prob_atlas,
    plot_roi,
    plot_stat_map,
    show,
)
from .matrix_plotting import (
    plot_contrast_matrix,
    plot_design_matrix,
    plot_design_matrix_correlation,
    plot_event,
    plot_matrix,
)
from .surf_plotting import (
    plot_img_on_surf,
    plot_surf,
    plot_surf_contours,
    plot_surf_roi,
    plot_surf_stat_map,
)

__all__ = [
    "cm",  # cm not in API doc
    "find_cut_slices",
    "find_parcellation_cut_coords",
    "find_probabilistic_atlas_cut_coords",
    "find_xyz_cut_coords",
    "plot_anat",
    "plot_carpet",
    "plot_connectome",
    "plot_contrast_matrix",
    "plot_design_matrix",
    "plot_design_matrix_correlation",
    "plot_epi",
    "plot_event",
    "plot_glass_brain",
    "plot_img",
    "plot_img_comparison",
    "plot_img_on_surf",
    "plot_markers",
    "plot_matrix",
    "plot_prob_atlas",
    "plot_roi",
    "plot_stat_map",
    "plot_surf",
    "plot_surf_contours",
    "plot_surf_roi",
    "plot_surf_stat_map",
    "show",
    "view_connectome",
    "view_img",
    "view_img_on_surf",
    "view_markers",
    "view_surf",
]
