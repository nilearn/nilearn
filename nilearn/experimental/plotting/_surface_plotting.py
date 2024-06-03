from __future__ import annotations

import numpy

from nilearn import plotting as old_plotting
from nilearn._utils.docs import fill_doc
from nilearn.experimental.surface import Mesh, PolyMesh, SurfaceImage

DEFAULT_HEMI = "left"


def plot_surf(
    surf_map,
    surf_mesh=None,
    bg_map: str | numpy.array | SurfaceImage | None = None,
    hemi: str = DEFAULT_HEMI,
    **kwargs,
):
    """Plot surfaces with optional background and data."""
    surf_mesh, bg_map = _check_inputs(surf_mesh, bg_map, hemi)

    if not isinstance(surf_map, SurfaceImage):
        return old_plotting.plot_surf(
            surf_mesh=surf_mesh,
            surf_map=surf_map,
            hemi=hemi,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = surf_map.mesh.parts

    assert surf_map.data.parts[hemi] is not None

    return old_plotting.plot_surf(
        surf_mesh=surf_mesh[hemi],
        surf_map=surf_map.data.parts[hemi],
        hemi=hemi,
        bg_map=bg_map,
        **kwargs,
    )


@fill_doc
def plot_surf_stat_map(
    stat_map: SurfaceImage | str | numpy.array | None,
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    bg_map: SurfaceImage | str | numpy.array | None = None,
    hemi: str = DEFAULT_HEMI,
    **kwargs,
):
    """Plot a stats map on a surface :term:`mesh` with optional background.

    Parameters
    ----------
    stat_map: SurfaceImage or :obj:`str` or numpy.ndarray, optional

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    bg_map : str or numpy.ndarray or SurfaceImage, optional

    %(hemi)s
    """
    surf_mesh, bg_map = _check_inputs(surf_mesh, bg_map, hemi)

    if not isinstance(stat_map, SurfaceImage):
        return old_plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh,
            stat_map=stat_map,
            hemi=hemi,
            bg_map=bg_map,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = stat_map.mesh.parts

    assert stat_map.data.parts[hemi] is not None

    fig = old_plotting.plot_surf_stat_map(
        surf_mesh=surf_mesh[hemi],
        stat_map=stat_map.data.parts[hemi],
        hemi=hemi,
        bg_map=bg_map,
        **kwargs,
    )
    return fig


@fill_doc
def plot_surf_contours(
    roi_map: SurfaceImage | str | numpy.array | list[numpy.ndarray],
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    hemi: str = DEFAULT_HEMI,
    **kwargs,
):
    """Plot contours of ROIs on a surface, optionally over a statistical map.

    Parameters
    ----------
    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    roi_map : SurfaceImage or :obj:`str` or numpy.ndarray \
              or :obj:`list` of numpy.ndarray

    %(hemi)s
    """
    if isinstance(surf_mesh, PolyMesh):
        _check_hemi_present(surf_mesh, hemi)
        surf_mesh = surf_mesh.parts

    if not isinstance(roi_map, SurfaceImage):
        return old_plotting.plot_surf_contours(
            surf_mesh=surf_mesh,
            roi_map=roi_map,
            hemi=hemi,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = roi_map.mesh.parts

    assert roi_map.data.parts[hemi] is not None

    fig = old_plotting.plot_surf_contours(
        surf_mesh=surf_mesh[hemi],
        roi_map=roi_map.data.parts[hemi],
        hemi=hemi,
        **kwargs,
    )
    return fig


@fill_doc
def plot_surf_roi(
    roi_map: SurfaceImage | str | numpy.array | list[numpy.ndarray],
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    bg_map: SurfaceImage | str | numpy.array | None = None,
    hemi: str = DEFAULT_HEMI,
    **kwargs,
):
    """Plot ROI on a surface :term:`mesh` with optional background.

    Parameters
    ----------
    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    roi_map : SurfaceImage or :obj:`str` or numpy.ndarray \
              or :obj:`list` of numpy.ndarray

    bg_map : str or numpy.ndarray or SurfaceImage, optional

    %(hemi)s
    """
    surf_mesh, bg_map = _check_inputs(surf_mesh, bg_map, hemi)

    if not isinstance(roi_map, SurfaceImage):
        return old_plotting.plot_surf_roi(
            surf_mesh=surf_mesh,
            roi_map=roi_map,
            hemi=hemi,
            bg_map=bg_map,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = roi_map.mesh.parts

    assert roi_map.data.parts[hemi] is not None

    fig = old_plotting.plot_surf_roi(
        surf_mesh=surf_mesh[hemi],
        roi_map=roi_map.data.parts[hemi],
        hemi=hemi,
        bg_map=bg_map,
        **kwargs,
    )
    return fig


@fill_doc
def view_surf(
    surf_mesh: str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None,
    surf_map: SurfaceImage | str | numpy.array | None = None,
    bg_map: str | numpy.array | SurfaceImage | None = None,
    hemi: str = DEFAULT_HEMI,
    **kwargs,
):
    """Insert a surface plot of a surface map into an HTML page.

    Parameters
    ----------
    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    surf_map : SurfaceImage or :obj:`str` or numpy.ndarray, optional

    bg_map : str or numpy.ndarray or SurfaceImage, optional

    %(hemi)s
    """
    surf_mesh, bg_map = _check_inputs(surf_mesh, bg_map, hemi)

    if not isinstance(surf_map, SurfaceImage):
        return old_plotting.view_surf(
            surf_mesh=surf_mesh,
            surf_map=surf_map,
            bg_map=bg_map,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = surf_map.mesh.parts

    assert surf_map.data.parts[hemi] is not None

    fig = old_plotting.view_surf(
        surf_mesh=surf_mesh[hemi],
        surf_map=surf_map.data.parts[hemi],
        bg_map=bg_map,
        **kwargs,
    )
    return fig


def _check_inputs(surf_mesh, bg_map, hemi):
    if isinstance(surf_mesh, PolyMesh):
        _check_hemi_present(surf_mesh, hemi)
        surf_mesh = surf_mesh.parts

    bg_map = _check_bg_map(bg_map, hemi)

    return surf_mesh, bg_map


def _check_bg_map(bg_map, hemi):
    """Return proper format of background map to be used."""
    if isinstance(bg_map, SurfaceImage):
        assert bg_map.data.parts[hemi] is not None
        bg_map = bg_map.data.parts[hemi]
    return bg_map


def _check_hemi_present(mesh: PolyMesh, hemi: str):
    """Check that a given hemisphere exists both in data and mesh."""
    if hemi not in mesh.parts:
        raise ValueError(f"{hemi} must be present in mesh")
