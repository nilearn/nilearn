from __future__ import annotations

from nilearn import plotting as old_plotting
from nilearn._utils.docs import fill_doc
from nilearn.plotting.surf_plotting import DEFAULT_HEMI, _check_inputs


def plot_surf(
    surf_map,
    surf_mesh=None,
    bg_map=None,
    hemi=DEFAULT_HEMI,
    **kwargs,
):
    """Plot surfaces with optional background and data."""
    return old_plotting.plot_surf(
        surf_mesh=surf_mesh,
        surf_map=surf_map,
        hemi=hemi,
        bg_map=bg_map,
        **kwargs,
    )


@fill_doc
def plot_surf_stat_map(
    stat_map,
    surf_mesh=None,
    bg_map=None,
    hemi=DEFAULT_HEMI,
    **kwargs,
):
    """Plot a stats map on a surface :term:`mesh` with optional background."""
    stat_map, surf_mesh, bg_map = _check_inputs(
        stat_map, surf_mesh, hemi, bg_map
    )

    return old_plotting.plot_surf_stat_map(
        surf_mesh=surf_mesh,
        stat_map=stat_map,
        hemi=hemi,
        bg_map=bg_map,
        **kwargs,
    )


@fill_doc
def plot_surf_contours(
    roi_map,
    hemi=DEFAULT_HEMI,
    surf_mesh=None,
    **kwargs,
):
    """Plot contours of ROIs on a surface, optionally on a statistical map."""
    roi_map, surf_mesh, _ = _check_inputs(roi_map, surf_mesh, hemi)

    return old_plotting.plot_surf_contours(
        surf_mesh=surf_mesh,
        roi_map=roi_map,
        hemi=hemi,
        **kwargs,
    )


@fill_doc
def plot_surf_roi(
    roi_map,
    surf_mesh=None,
    bg_map=None,
    hemi: str = DEFAULT_HEMI,
    **kwargs,
):
    """Plot ROI on a surface :term:`mesh` with optional background."""
    roi_map, surf_mesh, bg_map = _check_inputs(
        roi_map, surf_mesh, hemi, bg_map
    )

    return old_plotting.plot_surf_roi(
        surf_mesh=surf_mesh,
        roi_map=roi_map,
        hemi=hemi,
        bg_map=bg_map,
        **kwargs,
    )


@fill_doc
def view_surf(
    surf_mesh,
    surf_map=None,
    bg_map=None,
    hemi=DEFAULT_HEMI,
    **kwargs,
):
    """Insert a surface plot of a surface map into an HTML page."""
    surf_map, surf_mesh, bg_map = _check_inputs(
        surf_map, surf_mesh, hemi, bg_map
    )

    return old_plotting.view_surf(
        surf_mesh=surf_mesh,
        surf_map=surf_map,
        bg_map=bg_map,
        **kwargs,
    )
