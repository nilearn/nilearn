from collections.abc import Sequence
from warnings import warn

import numpy as np
import pandas as pd

from nilearn import DEFAULT_DIVERGING_CMAP, image
from nilearn._utils import check_niimg_3d
from nilearn._utils.param_validation import check_params
from nilearn.plotting._utils import (
    _check_bg_map,
    _get_hemi,
    create_colormap_from_lut,
    get_colorbar_and_data_ranges,
)
from nilearn.surface import (
    PolyMesh,
    SurfaceImage,
    load_surf_data,
    load_surf_mesh,
    vol_to_surf,
)
from nilearn.surface.surface import (
    FREESURFER_DATA_EXTENSIONS,
    check_extensions,
    check_mesh_is_fsaverage,
    get_data,
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

    def plot_surf_contours(
        self,
        surf_mesh=None,
        roi_map=None,
        hemi=None,
        levels=None,
        labels=None,
        colors=None,
        legend=False,
        cmap="tab20",
        title=None,
        output_file=None,
        axes=None,
        figure=None,
        **kwargs,
    ):
        # TODO hemi returns None from here, if I pass to plot_surf,
        # returns error
        hemi = sanitize_hemi_for_surface_image(hemi, roi_map, surf_mesh)
        roi_map, surf_mesh, _ = check_surface_plotting_inputs(
            roi_map, surf_mesh, hemi, map_var_name="roi_map"
        )
        check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

        fig = self._plot_surf_contours(
            surf_mesh=surf_mesh,
            roi_map=roi_map,
            levels=levels,
            labels=labels,
            colors=colors,
            legend=legend,
            cmap=cmap,
            title=title,
            output_file=output_file,
            axes=axes,
            figure=figure,
            **kwargs,
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
        title_font_size=None,
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
            threshold=threshold,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            colorbar=colorbar,
            cbar_tick_format=cbar_tick_format,
            bg_on_data=bg_on_data,
            darkness=darkness,
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
            axes=axes,
            figure=figure,
            avg_method=avg_method,
            cbar_vmin=cbar_vmin,
            cbar_vmax=cbar_vmax,
            **kwargs,
        )
        return fig

    def plot_img_on_surf(
        self,
        stat_map,
        surf_mesh="fsaverage5",
        mask_img=None,
        hemispheres=None,
        bg_on_data=False,
        inflate=False,
        views=None,
        output_file=None,
        title=None,
        colorbar=True,
        vmin=None,
        vmax=None,
        threshold=None,
        symmetric_cbar="auto",
        cmap=DEFAULT_DIVERGING_CMAP,
        cbar_tick_format="%i",
        **kwargs,
    ):
        check_params(locals())
        if hemispheres in (None, "both"):
            hemispheres = ["left", "right"]
        if views is None:
            views = ["lateral", "medial"]

        stat_map = check_niimg_3d(stat_map, dtype="auto")
        modes = _check_views(views)
        hemis = _check_hemispheres(hemispheres)
        surf_mesh = check_mesh_is_fsaverage(surf_mesh)

        mesh_prefix = "infl" if inflate else "pial"
        surf = {
            "left": surf_mesh[f"{mesh_prefix}_left"],
            "right": surf_mesh[f"{mesh_prefix}_right"],
        }

        texture = {
            "left": vol_to_surf(
                stat_map, surf_mesh["pial_left"], mask_img=mask_img
            ),
            "right": vol_to_surf(
                stat_map, surf_mesh["pial_right"], mask_img=mask_img
            ),
        }

        # get vmin and vmax for entire data (all hemis)
        _, _, vmin, vmax = get_colorbar_and_data_ranges(
            image.get_data(stat_map),
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=symmetric_cbar,
        )

        fig = self._plot_img_on_surf(
            stat_map=stat_map,
            surf_mesh=surf_mesh,
            hemispheres=hemispheres,
            modes=modes,
            hemis=hemis,
            surf=surf,
            texture=texture,
            bg_on_data=bg_on_data,
            inflate=inflate,
            threshold=threshold,
            colorbar=colorbar,
            cbar_tick_format=cbar_tick_format,
            symmetric_cbar=symmetric_cbar,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            title=title,
            output_file=output_file,
            **kwargs,
        )

        return fig

    def plot_surf_roi(
        self,
        surf_mesh=None,
        roi_map=None,
        bg_map=None,
        hemi="left",
        view=None,
        avg_method=None,
        threshold=1e-14,
        alpha=None,
        vmin=None,
        vmax=None,
        cmap="gist_ncar",
        cbar_tick_format="auto",
        bg_on_data=False,
        darkness=0.7,
        title=None,
        title_font_size=None,
        output_file=None,
        axes=None,
        figure=None,
        colorbar=True,
        **kwargs,
    ):
        # set default view to dorsal if hemi is both and view is not set
        check_params(locals())
        if view is None:
            view = "dorsal" if hemi == "both" else "lateral"

        roi_map, surf_mesh, bg_map = check_surface_plotting_inputs(
            roi_map, surf_mesh, hemi, bg_map
        )
        # preload roi and mesh to determine vmin, vmax and give more useful
        # error messages in case of wrong inputs
        check_extensions(roi_map, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS)

        roi = load_surf_data(roi_map)

        idx_not_na = ~np.isnan(roi)
        if vmin is None:
            vmin = float(np.nanmin(roi))
        if vmax is None:
            vmax = float(1 + np.nanmax(roi))

        mesh = load_surf_mesh(surf_mesh)

        if roi.ndim != 1:
            raise ValueError(
                "roi_map can only have one dimension but has "
                f"{roi.ndim} dimensions"
            )
        if roi.shape[0] != mesh.n_vertices:
            raise ValueError(
                "roi_map does not have the same number of vertices "
                "as the mesh. If you have a list of indices for the "
                "ROI you can convert them into a ROI map like this:\n"
                "roi_map = np.zeros(n_vertices)\n"
                "roi_map[roi_idx] = 1"
            )
        if (roi < 0).any():
            # TODO raise ValueError in release 0.13
            warn(
                (
                    "Negative values in roi_map will no longer be allowed in"
                    " Nilearn version 0.13"
                ),
                DeprecationWarning,
            )
        if not np.array_equal(roi[idx_not_na], roi[idx_not_na].astype(int)):
            # TODO raise ValueError in release 0.13
            warn(
                (
                    "Non-integer values in roi_map will no longer be allowed "
                    "in Nilearn version 0.13"
                ),
                DeprecationWarning,
            )

        if isinstance(cmap, pd.DataFrame):
            cmap = create_colormap_from_lut(cmap)

        fig = self._plot_surf_roi(
            mesh,
            roi_map=roi,
            bg_map=bg_map,
            hemi=hemi,
            view=view,
            avg_method=avg_method,
            threshold=threshold,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            cbar_tick_format=cbar_tick_format,
            bg_on_data=bg_on_data,
            darkness=darkness,
            title=title,
            title_font_size=title_font_size,
            output_file=output_file,
            axes=axes,
            figure=figure,
            colorbar=colorbar,
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
