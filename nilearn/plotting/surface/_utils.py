"""Utility functions used in nilearn.plotting.surface module."""

from collections.abc import Sequence
from warnings import warn

import numpy as np

from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed
from nilearn._utils.logger import find_stack_level
from nilearn.plotting._utils import DEFAULT_ENGINE
from nilearn.surface import (
    PolyMesh,
    SurfaceImage,
    load_surf_data,
)
from nilearn.surface.surface import combine_hemispheres_meshes, get_data
from nilearn.surface.utils import check_polymesh_equal

DEFAULT_HEMI = "left"

VALID_VIEWS = (
    "anterior",
    "posterior",
    "medial",
    "lateral",
    "dorsal",
    "ventral",
    "left",
    "right",
)

VALID_HEMISPHERES = "left", "right", "both"


def get_surface_backend(engine=DEFAULT_ENGINE):
    """Instantiate and return the required backend engine.

    Parameters
    ----------
    engine: :obj:`str`, default='matplotlib'
        Name of the required backend engine. Can be ``matplotlib`` or
    ``plotly``.

    Returns
    -------
    backend : :class:`~nilearn.plotting.surface._matplotlib_backend` or
    :class:`~nilearn.plotting.surface._plotly_backend`.
        The backend module for the specified engine.
    """
    if engine == "matplotlib":
        if is_matplotlib_installed():
            import nilearn.plotting.surface._matplotlib_backend as backend
        else:
            raise ImportError(
                "Using engine='matplotlib' requires that ``matplotlib`` is "
                "installed."
            )
    elif engine == "plotly":
        if is_plotly_installed():
            import nilearn.plotting.surface._plotly_backend as backend
        else:
            raise ImportError(
                "Using engine='plotly' requires that ``plotly`` is installed."
            )
    else:
        raise ValueError(
            f"Unknown plotting engine {engine}. "
            "Please use either 'matplotlib' or "
            "'plotly'."
        )
    return backend


def check_engine_params(params, engine):
    """Check default values of the parameters that are not implemented for
    current engine and warn the user if the parameter has other value then
    None.

    Parameters
    ----------
    params: :obj:`dict`
        A dictionary where keys are the unimplemented parameter names for a
    specific engine and values are the assigned value for corresponding
    parameter.
    """
    for parameter, value in params.items():
        if value is not None:
            warn(
                f"'{parameter}' is not implemented "
                f"for the {engine} engine.\n"
                f"Got '{parameter} = {value}'.\n"
                f"Use '{parameter} = None' to silence this warning.",
                stacklevel=find_stack_level(),
            )


def _check_hemisphere_is_valid(hemi):
    return hemi in VALID_HEMISPHERES


def check_hemispheres(hemispheres):
    """Check whether the hemispheres passed to in plot_img_on_surf are \
    correct.

    hemispheres : :obj:`list`
        Any combination of 'left' and 'right'.

    """
    invalid_hemis = [
        not _check_hemisphere_is_valid(hemi) for hemi in hemispheres
    ]
    if any(invalid_hemis):
        raise ValueError(
            "Invalid hemispheres definition!\n"
            f"Got: {np.array(hemispheres)[invalid_hemis]!s}\n"
            f"Supported values are: {VALID_HEMISPHERES!s}"
        )
    return hemispheres


def check_surf_map(surf_map, n_vertices):
    """Help for plot_surf.

    This function checks the dimensions of provided surf_map.
    """
    surf_map_data = load_surf_data(surf_map)
    if surf_map_data.ndim != 1:
        raise ValueError(
            "'surf_map' can only have one dimension "
            f"but has '{surf_map_data.ndim}' dimensions"
        )
    if surf_map_data.shape[0] != n_vertices:
        raise ValueError(
            "The surf_map does not have the same number "
            "of vertices as the mesh."
        )
    return surf_map_data


def _check_view_is_valid(view) -> bool:
    """Check whether a single view is one of two valid input types.

    Parameters
    ----------
    view : :obj:`str` in {"anterior", "posterior", "medial", "lateral",
        "dorsal", "ventral" or pair of floats (elev, azim).

    Returns
    -------
    valid : True if view is valid, False otherwise.
    """
    if isinstance(view, str) and (view in VALID_VIEWS):
        return True
    return (
        isinstance(view, Sequence)
        and len(view) == 2
        and all(isinstance(x, (int, float)) for x in view)
    )


def check_views(views) -> list:
    """Check whether the views passed to in plot_img_on_surf are correct.

    Parameters
    ----------
    views : :obj:`list`
        Any combination of strings in {"anterior", "posterior", "medial",
        "lateral", "dorsal", "ventral"} and / or pair of floats (elev, azim).

    Returns
    -------
    views : :obj:`list`
        Views given as inputs.
    """
    invalid_views = [not _check_view_is_valid(view) for view in views]

    if any(invalid_views):
        raise ValueError(
            "Invalid view definition!\n"
            f"Got: {np.array(views)[invalid_views]!s}\n"
            f"Supported values are: {VALID_VIEWS!s}"
            " or a sequence of length 2"
            " setting the elevation and azimut of the camera."
        )

    return views


def _check_bg_map(bg_map, hemi):
    """Get the requested hemisphere if ``bg_map`` is a
    :obj:`~nilearn.surface.SurfaceImage`. If the hemisphere is not present,
    raise an error. If the hemisphere is `"both"`, concatenate the left and
    right hemispheres.

    Parameters
    ----------
    bg_map : Any

    hemi : :obj:`str`

    Returns
    -------
    bg_map : :obj:`str` | :obj:`pathlib.Path` | :obj:`numpy.ndarray` | None
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


def _get_hemi(surf_mesh, hemi):
    """Check that a given hemisphere exists in a
    :obj:`~nilearn.surface.PolyMesh` and return the corresponding
    ``surf_mesh``. If "both" is requested, combine the left and right
    hemispheres.

    Parameters
    ----------
    surf_mesh: :obj:`~nilearn.surface.PolyMesh`
        The surface mesh object containing the left and/or right hemisphere
        meshes.
    hemi: {'left', 'right', 'both'}

    Returns
    -------
    surf_mesh : :obj:`numpy.ndarray`,  :obj:`~nilearn.surface.InMemoryMesh`
        Surface mesh corresponding to the specified ``hemi``.

        - If ``hemi='left'`` or ``hemi='right'``, returns
          :obj:`numpy.ndarray`.
        - If ``hemi='both'``, returns :obj:`~nilearn.surface.InMemoryMesh`
    """
    if not isinstance(surf_mesh, PolyMesh):
        raise ValueError("mesh should be of type PolyMesh.")

    if hemi == "both":
        return combine_hemispheres_meshes(surf_mesh)
    elif hemi in ["left", "right"]:
        if hemi in surf_mesh.parts:
            return surf_mesh.parts[hemi]
        else:
            raise ValueError(
                f"{hemi=} does not exist in mesh. Available hemispheres are:"
                f"{surf_mesh.parts.keys()}."
            )
    else:
        raise ValueError("hemi must be one of 'left', 'right' or 'both'.")


@fill_doc
def check_surface_plotting_inputs(
    surf_map,
    surf_mesh,
    hemi=DEFAULT_HEMI,
    bg_map=None,
    map_var_name="surf_map",
    mesh_var_name="surf_mesh",
):
    """Check inputs for surface plotting.

    Where possible this will 'convert' the inputs if
    :obj:`~nilearn.surface.SurfaceImage` or :obj:`~nilearn.surface.PolyMesh`
    objects are passed to be able to give them to the surface plotting
    functions.

    - ``surf_mesh`` and ``surf_map`` cannot be `None` at the same time.
    - If ``surf_mesh=None``, then ``surf_map`` should be of type
    :obj:`~nilearn.surface.SurfaceImage`.
    - ``surf_mesh`` cannot be of type :obj:`~nilearn.surface.SurfaceImage`.
    - If ``surf_map`` and ``bg_map`` are of type
    :obj:`~nilearn.surface.SurfaceImage`, ``bg_map.mesh`` should be equal to
    ``surf_map.mesh``.

    Parameters
    ----------
    surf_map: :obj:`~nilearn.surface.SurfaceImage` | :obj:`numpy.ndarray`
              | None

    %(surf_mesh)s
        If `None` is passed, then ``surf_map`` must be a
        :obj:`~nilearn.surface.SurfaceImage` instance and the mesh from that
        :obj:`~nilearn.surface.SurfaceImage` instance will be used.

    %(hemi)s

    %(bg_map)s

    Returns
    -------
    surf_map : :obj:`numpy.ndarray`

    surf_mesh : :obj:`numpy.ndarray`,  :obj:`~nilearn.surface.InMemoryMesh`
        Surface mesh corresponding to the specified ``hemi``.

        - If ``hemi='left'`` or ``hemi='right'``, returns
          :obj:`numpy.ndarray`.
        - If ``hemi='both'``, returns :obj:`~nilearn.surface.InMemoryMesh`
    bg_map : :obj:`str` | :obj:`pathlib.Path` | :obj:`numpy.ndarray` | None

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
            f"then {map_var_name} must be a SurfaceImage instance."
        )

    if isinstance(surf_mesh, SurfaceImage):
        raise TypeError(
            "'surf_mesh' cannot be a SurfaceImage instance. ",
            "Accepted types are: str, list of two numpy.ndarray, "
            "InMemoryMesh, PolyMesh, or None.",
        )

    if isinstance(surf_mesh, PolyMesh):
        surf_mesh = _get_hemi(surf_mesh, hemi)

    if isinstance(surf_map, SurfaceImage):
        if len(surf_map.shape) > 1 and surf_map.shape[1] > 1:
            raise TypeError(
                "Input data has incompatible dimensionality. "
                f"Expected dimension is ({surf_map.shape[0]},) "
                f"or ({surf_map.shape[0]}, 1) "
                f"and you provided a {surf_map.shape} surface image."
            )

        if isinstance(bg_map, SurfaceImage):
            check_polymesh_equal(bg_map.mesh, surf_map.mesh)

        if surf_mesh is None:
            surf_mesh = _get_hemi(surf_map.mesh, hemi)

        # concatenate the left and right data if hemi is "both"
        if hemi == "both":
            surf_map = get_data(surf_map).T
        else:
            surf_map = surf_map.data.parts[hemi].T

    bg_map = _check_bg_map(bg_map, hemi)

    return surf_map, surf_mesh, bg_map


def get_bg_data(bg_map, n_vertices):
    """Get bg_data for bg_map and check if its number of vertices comply with
    n_vertices.
       If bg_map is None,  return an array of n_vertices elements with value
    0.5.
       If bg_map is not None, but number of vertices is not equal to
    n_vertices, raise ValueError.
    """
    if bg_map is None:
        bg_data = np.ones(n_vertices) * 0.5
    else:
        bg_data = np.copy(load_surf_data(bg_map))
        if bg_data.shape[0] != n_vertices:
            raise ValueError(
                "The bg_map does not have the same number "
                "of vertices as the mesh."
            )
    return bg_data


def get_faces_on_edge(faces, parc_idx):
    """Identify which faces lie on the outeredge of the parcellation defined by
    the indices in parc_idx.

    Parameters
    ----------
    faces : :obj:`numpy.ndarray` of shape (n, 3), indices of the mesh faces

    parc_idx : :obj:`numpy.ndarray`, indices of the vertices of the region to
    be plotted

    """
    # count how many vertices belong to the given parcellation in each face
    verts_per_face = np.isin(faces, parc_idx).sum(axis=1)

    # test if parcellation forms regions
    if np.all(verts_per_face < 2):
        raise ValueError("Vertices in parcellation do not form region.")

    vertices_on_edge = np.intersect1d(
        np.unique(faces[verts_per_face == 2]), parc_idx
    )
    faces_outside_edge = np.isin(faces, vertices_on_edge).sum(axis=1)

    return np.logical_and(faces_outside_edge > 0, verts_per_face < 3)


def sanitize_hemi_view(hemi, view):
    """Check ``hemi`` and ``view``, if ``view`` is `None`, set value for
    ``view`` depending on the ``hemi`` value and return ``view``.
    """
    check_hemispheres([hemi])
    if view is None:
        view = "dorsal" if hemi == "both" else "lateral"
    check_views([view])
    return view
