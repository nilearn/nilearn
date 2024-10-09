"""Plotting code for nilearn."""

# Original Authors: Chris Filo Gorgolewski, Gael Varoquaux
import importlib
import warnings

OPTIONAL_MATPLOTLIB_MIN_VERSION = "3.3.0"

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
            skip_if_running_tests("matplotlib not installed")
        raise
    else:
        from .._utils import compare_version

        # When matplotlib was successfully imported we need to check
        # that the version is greater that the minimum required one
        mpl_version = getattr(matplotlib, "__version__", "0.0.0")
        if not compare_version(
            mpl_version, ">=", OPTIONAL_MATPLOTLIB_MIN_VERSION
        ):
            raise ImportError(
                f"A matplotlib version of at least "
                f"{OPTIONAL_MATPLOTLIB_MIN_VERSION} "
                f"is required to use nilearn. {mpl_version} was found. "
                f"Please upgrade matplotlib"
            )
        current_backend = matplotlib.get_backend().lower()

        try:
            # Making sure the current backend is usable by matplotlib
            matplotlib.use(current_backend)
        except Exception:
            # If not, switching to default agg backend
            matplotlib.use("Agg")
        new_backend = matplotlib.get_backend().lower()

        if new_backend != current_backend:
            # Matplotlib backend has been changed, let's warn the user
            warnings.warn(f"Backend changed to {new_backend}...")


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
    "find_xyz_cut_coords",
    "find_parcellation_cut_coords",
    "find_probabilistic_atlas_cut_coords",
    "plot_anat",
    "plot_connectome",
    "plot_carpet",
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
