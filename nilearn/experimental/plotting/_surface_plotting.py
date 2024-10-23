from __future__ import annotations

from nilearn import plotting as old_plotting
from nilearn._utils.docs import fill_doc
from nilearn.plotting.surf_plotting import _check_inputs

DEFAULT_HEMI = "left"


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
