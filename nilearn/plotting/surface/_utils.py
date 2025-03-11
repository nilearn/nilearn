from collections.abc import Sequence
from warnings import warn

import numpy as np

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn._utils.param_validation import check_params
from nilearn.plotting._utils import (
    check_surface_plotting_inputs,
    get_colorbar_and_data_ranges,
)
from nilearn.surface import load_surf_data, load_surf_mesh
from nilearn.surface.surface import (
    FREESURFER_DATA_EXTENSIONS,
    check_extensions,
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


# subset of data format extensions supported
DATA_EXTENSIONS = (
    "gii",
    "gii.gz",
    "mgz",
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


def _get_faces_on_edge(faces, parc_idx):
    """Identify which faces lie on the outeredge of the parcellation \
    defined by the indices in parc_idx.

    Parameters
    ----------
    faces : :class:`numpy.ndarray` of shape (n, 3), indices of the mesh faces

    parc_idx : :class:`numpy.ndarray`, indices of the vertices
        of the region to be plotted

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


class SurfaceBackend:
    def plot_surf(
        self,
        surf_mesh=None,
        surf_map=None,
        bg_map=None,
        hemi=None,
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

        fig = self._plot_surf(
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

        return fig

    def plot_surf_stat_map(
        self,
        surf_mesh=None,
        stat_map=None,
        bg_map=None,
        hemi="left",
        view=None,
        threshold=None,
        alpha=None,
        vmin=None,
        vmax=None,
        cmap=DEFAULT_DIVERGING_CMAP,
        colorbar=True,
        symmetric_cbar="auto",
        cbar_tick_format="auto",
        bg_on_data=False,
        darkness=0.7,
        title=None,
        title_font_size=18,
        output_file=None,
        axes=None,
        figure=None,
        avg_method=None,
        **kwargs,
    ):
        check_params(locals())

        stat_map, surf_mesh, bg_map = check_surface_plotting_inputs(
            stat_map, surf_mesh, hemi, bg_map, map_var_name="stat_map"
        )

        check_extensions(stat_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)
        loaded_stat_map = load_surf_data(stat_map)

        # Call get_colorbar_and_data_ranges to derive symmetric vmin, vmax
        # And colorbar limits depending on symmetric_cbar settings
        cbar_vmin, cbar_vmax, vmin, vmax = get_colorbar_and_data_ranges(
            loaded_stat_map,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=symmetric_cbar,
        )

        fig = self._plot_surf_stat_map(
            surf_mesh,
            surf_map=loaded_stat_map,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            avg_method=avg_method,
            threshold=threshold,
            cmap=cmap,
            symmetric_cmap=True,
            colorbar=colorbar,
            cbar_tick_format=cbar_tick_format,
            alpha=alpha,
            bg_on_data=bg_on_data,
            darkness=darkness,
            vmin=vmin,
            vmax=vmax,
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
            axes=axes,
            figure=figure,
            cbar_vmin=cbar_vmin,
            cbar_vmax=cbar_vmax,
            **kwargs,
        )
        return fig

    def _check_backend_params(self, params_not_implemented):
        for parameter, value in params_not_implemented.items():
            if value is not None:
                warn(
                    f"'{parameter}' is not implemented "
                    f"for the {self.name} engine.\n"
                    f"Got '{parameter} = {value}'.\n"
                    f"Use '{parameter} = None' to silence this warning."
                )


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
