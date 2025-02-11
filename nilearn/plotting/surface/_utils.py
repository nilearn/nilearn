from warnings import warn

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from nilearn.plotting import cm
from nilearn.plotting._utils import to_color_strings
from nilearn.surface import (
    PolyMesh,
    SurfaceImage,
    load_surf_data,
)
from nilearn.surface.surface import combine_hemispheres_meshes, get_data


def get_vertexcolor(
    surf_map,
    cmap,
    norm,
    absolute_threshold=None,
    bg_map=None,
    bg_on_data=None,
    darkness=None,
):
    """Get the color of the vertices."""
    if bg_map is None:
        bg_data = np.ones(len(surf_map)) * 0.5
        bg_vmin, bg_vmax = 0, 1
    else:
        bg_data = np.copy(load_surf_data(bg_map))

    # scale background map if need be
    bg_vmin, bg_vmax = np.min(bg_data), np.max(bg_data)
    if bg_vmin < 0 or bg_vmax > 1:
        bg_norm = mpl.colors.Normalize(vmin=bg_vmin, vmax=bg_vmax)
        bg_data = bg_norm(bg_data)

    if darkness is not None:
        bg_data *= darkness
        warn(
            (
                "The `darkness` parameter will be deprecated in release 0.13. "
                "We recommend setting `darkness` to None"
            ),
            DeprecationWarning,
        )

    bg_colors = plt.get_cmap("Greys")(bg_data)

    # select vertices which are filtered out by the threshold
    if absolute_threshold is None:
        under_threshold = np.zeros_like(surf_map, dtype=bool)
    else:
        under_threshold = np.abs(surf_map) < absolute_threshold

    surf_colors = cmap(norm(surf_map).data)
    # set transparency of voxels under threshold to 0
    surf_colors[under_threshold, 3] = 0
    if bg_on_data:
        # if need be, set transparency of voxels above threshold to 0.7
        # so that background map becomes visible
        surf_colors[~under_threshold, 3] = 0.7

    vertex_colors = cm.mix_colormaps(surf_colors, bg_colors)

    return to_color_strings(vertex_colors)


def sanitize_hemi_for_surface_image(hemi, map, mesh):
    if hemi is None and (
        isinstance(map, SurfaceImage) or isinstance(mesh, PolyMesh)
    ):
        return "left"

    if (
        hemi is not None
        and not isinstance(map, SurfaceImage)
        and not isinstance(mesh, PolyMesh)
    ):
        warn(
            category=UserWarning,
            message=(
                f"{hemi=} was passed "
                f"with {type(map)=} and {type(mesh)=}.\n"
                "This value will be ignored as it is only used when "
                "'roi_map' is a SurfaceImage instance "
                "and  / or 'surf_mesh' is a PolyMesh instance."
            ),
            stacklevel=3,
        )
    return hemi


def check_surface_plotting_inputs(
    surf_map,
    surf_mesh,
    hemi="left",
    bg_map=None,
    map_var_name="surf_map",
    mesh_var_name="surf_mesh",
):
    """Check inputs for surface plotting.

    Where possible this will 'convert' the inputs
    if SurfaceImage or PolyMesh objects are passed
    to be able to give them to the surface plotting functions.

    Returns
    -------
    surf_map : numpy.ndarray

    surf_mesh : numpy.ndarray

    bg_map : str | pathlib.Path | numpy.ndarray | None

    """
    if surf_mesh is None and surf_map is None:
        raise TypeError(
            f"{mesh_var_name} and {map_var_name} cannot both be None."
            f"If you want to pass {mesh_var_name}=None, "
            f"then {mesh_var_name} must be a SurfaceImage instance."
        )

    if surf_mesh is None and not isinstance(surf_map, SurfaceImage):
        raise TypeError(
            f"If you want to pass {mesh_var_name}=None, "
            f"then {mesh_var_name} must be a SurfaceImage instance."
        )

    if isinstance(surf_mesh, PolyMesh):
        surf_mesh = _get_hemi(surf_mesh, hemi)

    if isinstance(surf_mesh, SurfaceImage):
        raise TypeError(
            "'surf_mesh' cannot be a SurfaceImage instance. ",
            "Accepted types are: str, list of two numpy.ndarray, "
            "InMemoryMesh, PolyMesh, or None.",
        )

    if isinstance(surf_map, SurfaceImage):
        if surf_mesh is None:
            surf_mesh = _get_hemi(surf_map.mesh, hemi)
        if len(surf_map.shape) > 1 and surf_map.shape[1] > 1:
            raise TypeError(
                "Input data has incompatible dimensionality. "
                f"Expected dimension is ({surf_map.shape[0]},) "
                f"or ({surf_map.shape[0]}, 1) "
                f"and you provided a {surf_map.shape} surface image."
            )
        # concatenate the left and right data if hemi is "both"
        if hemi == "both":
            surf_map = get_data(surf_map).T
        else:
            surf_map = surf_map.data.parts[hemi].T

    bg_map = _check_bg_map(bg_map, hemi)

    return surf_map, surf_mesh, bg_map


def _get_hemi(mesh, hemi):
    """Check that a given hemisphere exists in a PolyMesh and return the
    corresponding mesh. If "both" is requested, combine the left and right
    hemispheres.
    """
    if hemi == "both":
        return combine_hemispheres_meshes(mesh)
    elif hemi in mesh.parts:
        return mesh.parts[hemi]
    else:
        raise ValueError("hemi must be one of left, right or both.")


def _check_bg_map(bg_map, hemi):
    """Get the requested hemisphere if bg_map is a SurfaceImage. If the
    hemisphere is not present, raise an error. If the hemisphere is "both",
    concatenate the left and right hemispheres.

    bg_map : Any

    hemi : str

    Returns
    -------
    bg_map : str | pathlib.Path | numpy.ndarray | None
    """
    if isinstance(bg_map, SurfaceImage):
        if len(bg_map.shape) > 1 and bg_map.shape[1] > 1:
            raise TypeError(
                "Input data has incompatible dimensionality. "
                f"Expected dimension is ({bg_map.shape[0]},) "
                f"or ({bg_map.shape[0]}, 1) "
                f"and you provided a {bg_map.shape} surface image."
            )
        if hemi == "both":
            bg_map = get_data(bg_map)
        else:
            assert bg_map.data.parts[hemi] is not None
            bg_map = bg_map.data.parts[hemi]
    return bg_map
