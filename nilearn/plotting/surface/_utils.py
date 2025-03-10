from collections.abc import Sequence
from warnings import warn

import numpy as np

from nilearn.surface import load_surf_data

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


class SurfaceBackend():

    def _check_params(self, params_not_implemented):
        for parameter, value in params_not_implemented.items():
            if value is not None:
                warn(
                    f"'{parameter}' is not implemented "
                    f"for the {self.name} engine.\n"
                    f"Got '{parameter} = {value}'.\n"
                    f"Use '{parameter} = None' to silence this warning."
                )

    def _check_surf_map(self, surf_map, n_vertices):
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


class SurfaceFigure:
    """Abstract class for surface figures.

    Parameters
    ----------
    figure : Figure instance or ``None``, optional
        Figure to be wrapped.

    output_file : :obj:`str` or ``None``, optional
        Path to output file.
    """

    def __init__(self, figure=None, output_file=None, hemi="left"):
        self.figure = figure
        self.output_file = output_file
        self.hemi = hemi

    def show(self):
        """Show the figure."""
        raise NotImplementedError

    def _check_output_file(self, output_file=None):
        """If an output file is provided, \
        set it as the new default output file.

        Parameters
        ----------
        output_file : :obj:`str` or ``None``, optional
            Path to output file.
        """
        if output_file is None:
            if self.output_file is None:
                raise ValueError(
                    "You must provide an output file name to save the figure."
                )
        else:
            self.output_file = output_file

    def add_contours(self):
        """Draw boundaries around roi."""
        raise NotImplementedError
