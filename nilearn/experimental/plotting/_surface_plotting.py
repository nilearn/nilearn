from __future__ import annotations

import numpy

from nilearn import plotting as old_plotting
from nilearn._utils.docs import fill_doc
from nilearn.experimental.surface import Mesh, PolyMesh, SurfaceImage


def plot_surf(
    img, part: str | None = None, mesh=None, view: str | None = None, **kwargs
):
    """Plot surfaces with optional background and data.

    Parameters
    ----------
    surf_map : SurfaceImage or :obj:`str` or numpy.ndarray, optional

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s

    bg_map : str or numpy.ndarray or SurfaceImage, optional
    """
    if not isinstance(img, SurfaceImage):
        return old_plotting.plot_surf(
            surf_mesh=mesh,
            surf_map=img,
            hemi=part,
            **kwargs,
        )

    if mesh is None:
        mesh = img.mesh
    if part is None:
        # only take the first hemisphere by default
        part = list(img.data.parts.keys())[0]
    if view is None:
        view = "lateral"

    return old_plotting.plot_surf(
        surf_mesh=mesh.parts[part],
        surf_map=img.data.parts[part],
        hemi=part,
        view=view,
        **kwargs,
    )


@fill_doc
def plot_surf_stat_map(
    stat_map: SurfaceImage | str | numpy.array | None = None,
    hemi: str = "left",
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    bg_map: str | numpy.array | SurfaceImage | None = None,
    **kwargs,
):
    """Plot a stats map on a surface :term:`mesh` with optional background.

    Parameters
    ----------
    stat_map: SurfaceImage or :obj:`str` or numpy.ndarray, optional

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s

    bg_map : str or numpy.ndarray or SurfaceImage, optional
    """
    bg_map = _check_bg_map(bg_map, hemi)

    if not isinstance(stat_map, SurfaceImage):
        return old_plotting.plot_surf_stat_map(
            surf_mesh=surf_mesh,
            stat_map=stat_map,
            hemi=hemi,
            bg_map=bg_map,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = stat_map.mesh

    if isinstance(surf_mesh, PolyMesh):
        _check_hemi_present(surf_mesh, hemi)
        surf_mesh = surf_mesh.parts

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
    hemi: str = "left",
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    **kwargs,
):
    """Plot contours of ROIs on a surface, \
    optionally over a statistical map.

    Parameters
    ----------
    roi_map : SurfaceImage or :obj:`str` or numpy.ndarray \
              or :obj:`list` of numpy.ndarray

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s
    """
    if not isinstance(roi_map, SurfaceImage):
        return old_plotting.plot_surf_contours(
            surf_mesh=surf_mesh,
            roi_map=roi_map,
            hemi=hemi,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = roi_map.mesh

    if isinstance(surf_mesh, PolyMesh):
        _check_hemi_present(surf_mesh, hemi)
        surf_mesh = surf_mesh.parts

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
    hemi: str = "left",
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    bg_map: str | numpy.array | SurfaceImage | None = None,
    **kwargs,
):
    """Plot ROI on a surface :term:`mesh` with optional background.

    Parameters
    ----------
    roi_map : SurfaceImage or :obj:`str` or numpy.ndarray \
              or :obj:`list` of numpy.ndarray

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s

    bg_map : str or numpy.ndarray or SurfaceImage, optional
    """
    bg_map = _check_bg_map(bg_map, hemi)

    if not isinstance(roi_map, SurfaceImage):
        return old_plotting.plot_surf_roi(
            surf_mesh=surf_mesh,
            roi_map=roi_map,
            hemi=hemi,
            bg_map=bg_map,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = roi_map.mesh

    if isinstance(surf_mesh, PolyMesh):
        _check_hemi_present(surf_mesh, hemi)
        surf_mesh = surf_mesh.parts

    # TODO refactor
    assert roi_map.data.parts[hemi] is not None

    fig = old_plotting.plot_surf_roi(
        surf_mesh=surf_mesh.parts[hemi],
        roi_map=roi_map.data.parts[hemi],
        hemi=hemi,
        bg_map=bg_map,
        **kwargs,
    )
    return fig


@fill_doc
def view_surf(
    surf_map: SurfaceImage | str | numpy.array | None = None,
    hemi: str = "left",
    surf_mesh: (
        str | list[numpy.array, numpy.array] | Mesh | PolyMesh | None
    ) = None,
    bg_map: str | numpy.array | SurfaceImage | None = None,
    **kwargs,
):
    """Insert a surface plot of a surface map into an HTML page.

    Parameters
    ----------
    surf_map : SurfaceImage or :obj:`str` or numpy.ndarray, optional

    surf_mesh : :obj:`str` or :obj:`list` of two numpy.ndarray \
                or Mesh or PolyMesh, optional

    %(hemi)s

    bg_map : str or numpy.ndarray or SurfaceImage, optional
    """
    bg_map = _check_bg_map(bg_map, hemi)

    if not isinstance(surf_map, SurfaceImage):
        return old_plotting.view_surf(
            surf_mesh=surf_mesh,
            surf_map=surf_map,
            bg_map=bg_map,
            **kwargs,
        )

    if surf_mesh is None:
        surf_mesh = surf_map.mesh

    if isinstance(surf_mesh, PolyMesh):
        _check_hemi_present(surf_mesh, hemi)
        surf_mesh = surf_mesh.parts

    assert surf_map.data.parts[hemi] is not None

    fig = old_plotting.view_surf(
        surf_mesh=surf_mesh[hemi],
        surf_map=surf_map.data.parts[hemi],
        bg_map=bg_map,
        **kwargs,
    )
    return fig


def _check_bg_map(bg_map, hemi):
    """Return proper format of background map to be used.

    TODO refactor when experimental gets integrated as part of stable code.
    """
    if isinstance(bg_map, SurfaceImage):
        assert bg_map.data.parts[hemi] is not None
        bg_map = bg_map.data.parts[hemi]
    return bg_map


def _check_hemi_present(mesh: PolyMesh, hemi: str):
    """Check that a given hemisphere exists both in data and mesh."""
    if hemi not in mesh.parts:
        raise ValueError(f"{hemi} must be present in mesh")
