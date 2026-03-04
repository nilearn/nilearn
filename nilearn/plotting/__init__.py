"""Plotting code for nilearn."""

from nilearn._utils.helpers import set_mpl_backend

set_mpl_backend()

###############################################################################
from nilearn.plotting import cm
from nilearn.plotting.find_cuts import (
    find_cut_slices,
    find_parcellation_cut_coords,
    find_probabilistic_atlas_cut_coords,
    find_xyz_cut_coords,
)
from nilearn.plotting.html_connectome import view_connectome, view_markers
from nilearn.plotting.html_stat_map import view_img
from nilearn.plotting.image import (
    plot_anat,
    plot_carpet,
    plot_connectome,
    plot_epi,
    plot_glass_brain,
    plot_img,
    plot_markers,
    plot_prob_atlas,
    plot_roi,
    plot_stat_map,
    show,
)
from nilearn.plotting.img_comparison import (
    plot_bland_altman,
    plot_img_comparison,
)
from nilearn.plotting.matrix import (
    plot_contrast_matrix,
    plot_design_matrix,
    plot_design_matrix_correlation,
    plot_event,
    plot_matrix,
)
from nilearn.plotting.surface import (
    plot_img_on_surf,
    plot_surf,
    plot_surf_contours,
    plot_surf_roi,
    plot_surf_stat_map,
    view_img_on_surf,
    view_surf,
)

__all__ = [
    "cm",  # cm not in API doc
    "find_cut_slices",
    "find_parcellation_cut_coords",
    "find_probabilistic_atlas_cut_coords",
    "find_xyz_cut_coords",
    "plot_anat",
    "plot_bland_altman",
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
