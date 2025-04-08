"""Functions that are common to all possible backend implementations for
surface visualization functions in
:obj:`~nilearn.plotting.surface.surf_plotting`.

"Matplotlib" is default engine for surface visualization functions. For some
functions, there is also a "plotly" implementation.

All dependencies and functions related to "matplotlib" implementation is in
:obj:`~nilearn.plotting.surface._matplotlib_backend` module.

All dependencies and functions related to "plotly" implementation is in
:obj:`~nilearn.plotting.surface._plotly_backend` module.

Each backend engine implementation should be a self contained module. Any
imports on the engine package, or engine specific utility functions should not
appear elsewhere.
"""

from collections.abc import Sequence
from warnings import warn

import numpy as np

from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed
from nilearn._utils.param_validation import check_params
from nilearn.plotting.surface._utils import check_surface_plotting_inputs
from nilearn.surface import load_surf_data, load_surf_mesh
from nilearn.surface.surface import (
    FREESURFER_DATA_EXTENSIONS,
    check_extensions,
)

# subset of data format extensions supported
DATA_EXTENSIONS = (
    "gii",
    "gii.gz",
    "mgz",
)

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
        if is_matplotlib_installed():
            from nilearn.plotting.surface._matplotlib_backend import (
                MatplotlibBackend,
            )

            return MatplotlibBackend()
        else:
            raise ImportError(
                "Using engine='matplotlib' requires that ``matplotlib`` is "
                "installed."
            )
    elif engine == "plotly":
        if is_plotly_installed():
            from nilearn.plotting.surface._plotly_backend import PlotlyBackend

            return PlotlyBackend()
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


class BaseSurfaceBackend:
    """A base class that behaves as an interface for Surface plotting
    backend.

    The methods of class should be implemented by each engine used as backend.
    """

    def _check_engine_params(self, params):
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
                    f"for the {self.name} engine.\n"
                    f"Got '{parameter} = {value}'.\n"
                    f"Use '{parameter} = None' to silence this warning."
                )

    def plot_surf(
        self,
        surf_mesh=None,
        surf_map=None,
        bg_map=None,
        hemi="left",
        view=None,
        cmap=None,
        symmetric_cmap=None,
        colorbar=True,
        avg_method=None,
        threshold=None,
        alpha=None,
        bg_on_data=False,
        darkness=0.7,
        vmin=None,
        vmax=None,
        cbar_vmin=None,
        cbar_vmax=None,
        cbar_tick_format="auto",
        title=None,
        title_font_size=None,
        output_file=None,
        axes=None,
        figure=None,
    ):
        check_params(locals())
        if view is None:
            view = "dorsal" if hemi == "both" else "lateral"

        surf_map, surf_mesh, bg_map = check_surface_plotting_inputs(
            surf_map, surf_mesh, hemi, bg_map
        )

        check_extensions(surf_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

        coords, faces = load_surf_mesh(surf_mesh)

        return self._plot_surf(
            coords,
            faces,
            surf_map=surf_map,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            colorbar=colorbar,
            avg_method=avg_method,
            threshold=threshold,
            alpha=alpha,
            bg_on_data=bg_on_data,
            darkness=darkness,
            vmin=vmin,
            vmax=vmax,
            cbar_vmin=cbar_vmin,
            cbar_vmax=cbar_vmax,
            cbar_tick_format=cbar_tick_format,
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
            axes=axes,
            figure=figure,
        )


def _check_hemisphere_is_valid(hemi):
    return hemi in VALID_HEMISPHERES


def _check_hemispheres(hemispheres):
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


def _check_views(views) -> list:
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


def _check_surf_map(surf_map, n_vertices):
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
