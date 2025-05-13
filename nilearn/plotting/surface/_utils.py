"""Utility functions used in nilearn.plotting.surface module."""

from warnings import warn

from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed
from nilearn.surface import (
    PolyMesh,
    SurfaceImage,
)
from nilearn.surface.surface import combine_hemispheres_meshes, get_data
from nilearn.surface.utils import check_polymesh_equal


def get_surface_backend(engine="matplotlib"):
    """Instantiate and return the required backend engine.

    Parameters
    ----------
    engine: :obj:`str`, default='matplotlib'
        Name of the required backend engine. Can be 'matplotlib' or 'plotly'.

    Returns
    -------
    backend : :class:`~nilearn.plotting.surface._backend.BaseSurfaceBackend`
        The backend for the specified engine.
    """
    if engine == "matplotlib":
        if not is_matplotlib_installed():
            raise ImportError(
                "Using engine='matplotlib' requires that ``matplotlib`` is "
                "installed."
            )
        from nilearn.plotting.surface._matplotlib_backend import (
            MatplotlibSurfaceBackend,
        )

        return MatplotlibSurfaceBackend()
    elif engine == "plotly":
        if is_plotly_installed():
            from nilearn.plotting.surface._plotly_backend import (
                PlotlySurfaceBackend,
            )

            return PlotlySurfaceBackend()
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


def _check_bg_map(bg_map, hemi):
    """Get the requested hemisphere if bg_map is a SurfaceImage. If the
    hemisphere is not present, raise an error. If the hemisphere is "both",
    concatenate the left and right hemispheres.

    bg_map : Any

    hemi : str

    surf_map: SurfaceImage

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

        if isinstance(bg_map, SurfaceImage):
            check_polymesh_equal(bg_map.mesh, surf_map.mesh)

        # concatenate the left and right data if hemi is "both"
        if hemi == "both":
            surf_map = get_data(surf_map).T
        else:
            surf_map = surf_map.data.parts[hemi].T

    bg_map = _check_bg_map(bg_map, hemi)

    return surf_map, surf_mesh, bg_map


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
