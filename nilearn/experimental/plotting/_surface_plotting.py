from __future__ import annotations

from pathlib import Path

import numpy

from nilearn import plotting as old_plotting
from nilearn._utils.docs import fill_doc
from nilearn.experimental.surface import PolyMesh, SurfaceImage

DEFAULT_HEMI = "left"

# TODO double check types if we decide to keep them
# SURF_MESH_TYPE = (
#     str | Path | list[numpy.ndarray, numpy.ndarray] | Mesh | PolyMesh | None
# )
# MAP_TYPE = str | Path | numpy.ndarray | SurfaceImage | None


def _check_inputs(
    surf_map,
    surf_mesh,
    hemi: str,
    bg_map=None,
):
    """Check inputs for surface plotting.

    Where possible this will 'convert' the inputs to be able to pass them
    to the the 'old' surface plotting functions.
    """
    if isinstance(surf_mesh, PolyMesh):
        _check_hemi_present(surf_mesh, hemi)
        surf_mesh = surf_mesh.parts[hemi]

    if isinstance(surf_map, SurfaceImage):
        if surf_mesh is None:
            surf_mesh = surf_map.mesh.parts[hemi]
        surf_map = surf_map.data.parts[hemi]

    bg_map = _check_bg_map(bg_map, hemi)

    return surf_map, surf_mesh, bg_map


def _check_bg_map(bg_map, hemi: str) -> str | Path | numpy.ndarray | None:
    """Return proper format of background map to be used."""
    if isinstance(bg_map, SurfaceImage):
        assert bg_map.data.parts[hemi] is not None
        bg_map = bg_map.data.parts[hemi]
    return bg_map


def _check_hemi_present(mesh: PolyMesh, hemi: str) -> None:
    """Check that a given hemisphere exists both in data and mesh."""
    if hemi not in mesh.parts:
        raise ValueError(f"{hemi} must be present in mesh")


def plot_surf(
    surf_map,
    surf_mesh=None,
    bg_map=None,
    hemi: str = DEFAULT_HEMI,
    **kwargs,
):
    """Plot surfaces with optional background and data.

    Parameters
    ----------
    surf_mesh : PathLike or :obj:`list` of two numpy.ndarray \
                or Mesh Surface Mesh or PolyMesh
        Can be:

        - a file (valid formats are .gii or Freesurfer specific files
          such as .orig, .pial, .sphere, .white, .inflated)
        - a list of two Numpy arrays,
          the first containing the x-y-z coordinates
          of the :term:`mesh` :term:`vertices<vertex>`,
          the second containing the indices (into coords)
          of the :term:`mesh` :term:`faces`,
        - a Mesh object with "coordinates" and "faces" attributes.
        - a PolyMesh object
        - a SurfaceImage object


    surf_map : PathLike or numpy.ndarray or SurfaceImage, optional
        Data to be displayed on the surface :term:`mesh`.
        Can be:

        - a file (valid formats are .gii, .mgz, .nii, .nii.gz,
          or Freesurfer specific files such as
          .thickness, .area, .curv, .sulc, .annot, .label)
        - a Numpy array with a value for each :term:`vertex`
          of the `surf_mesh`
        - a SurfaceImage object.


    bg_map : PathLike or numpy.ndarray or SurfaceImage, optional
        Can be:

        - a file (valid formats are .gii, .mgz, .nii, .nii.gz,
          or Freesurfer specific files such as
          .thickness, .area, .curv, .sulc, .annot, .label)
        - a Numpy array with a value for each :term:`vertex`
          of the `surf_mesh`
        - a SurfaceImage object.


    """
    surf_map, surf_mesh, bg_map = _check_inputs(
        surf_map, surf_mesh, hemi, bg_map
    )

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
    hemi: str = DEFAULT_HEMI,
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
    hemi: str = DEFAULT_HEMI,
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
    hemi: str = DEFAULT_HEMI,
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
