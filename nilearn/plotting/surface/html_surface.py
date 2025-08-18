"""Handle plotting of surfaces for html rendering."""

import json
from warnings import warn

import numpy as np

from nilearn import DEFAULT_DIVERGING_CMAP
from nilearn._utils.docs import fill_doc
from nilearn._utils.html_document import HTMLDocument
from nilearn._utils.logger import find_stack_level
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn._utils.param_validation import check_params
from nilearn.plotting import cm
from nilearn.plotting._engine_utils import colorscale
from nilearn.plotting.js_plotting_utils import (
    add_js_lib,
    get_html_template,
    mesh_to_plotly,
)
from nilearn.plotting.surface._utils import (
    DEFAULT_ENGINE,
    DEFAULT_HEMI,
    check_surface_plotting_inputs,
    get_surface_backend,
)
from nilearn.surface import (
    PolyMesh,
    SurfaceImage,
    load_surf_data,
    load_surf_mesh,
)
from nilearn.surface.surface import (
    check_mesh_and_data,
    check_mesh_is_fsaverage,
    combine_hemispheres_meshes,
    get_data,
)


class SurfaceView(HTMLDocument):  # noqa: D101
    pass


def _one_mesh_info(
    surf_map,
    surf_mesh,
    threshold=None,
    cmap=cm.cold_hot,
    black_bg=False,
    bg_map=None,
    symmetric_cmap=True,
    bg_on_data=False,
    darkness=0.7,
    vmax=None,
    vmin=None,
):
    """Prepare info for plotting one surface map on a single mesh.

    This computes the dictionary that gets inserted in the web page,
    which contains the encoded mesh, colors, min and max values, and
    background color.

    """
    colors = colorscale(
        cmap,
        surf_map,
        threshold,
        symmetric_cmap=symmetric_cmap,
        vmax=vmax,
        vmin=vmin,
    )
    info = {"inflated_both": mesh_to_plotly(surf_mesh)}
    backend = get_surface_backend(DEFAULT_ENGINE)
    info["vertexcolor_both"] = backend._get_vertexcolor(
        surf_map,
        colors["cmap"],
        colors["norm"],
        absolute_threshold=colors["abs_threshold"],
        bg_map=bg_map,
        bg_on_data=bg_on_data,
        darkness=darkness,
    )
    info["cmin"], info["cmax"] = float(colors["vmin"]), float(colors["vmax"])
    info["black_bg"] = black_bg
    info["full_brain_mesh"] = False
    info["colorscale"] = colors["colors"]
    return info


def one_mesh_info(
    surf_map,
    surf_mesh,
    threshold=None,
    cmap=DEFAULT_DIVERGING_CMAP,
    black_bg=False,
    bg_map=None,
    symmetric_cmap=True,
    bg_on_data=False,
    darkness=0.7,
    vmax=None,
    vmin=None,
):
    """Deprecate public function. See _one_mesh_info."""
    warn(
        category=DeprecationWarning,
        message="one_mesh_info is a private function and is renamed "
        "to _one_mesh_info. Using the deprecated name will "
        "raise an error in release 0.13",
        stacklevel=find_stack_level(),
    )

    return _one_mesh_info(
        surf_map,
        surf_mesh,
        threshold=threshold,
        cmap=cmap,
        black_bg=black_bg,
        bg_map=bg_map,
        symmetric_cmap=symmetric_cmap,
        bg_on_data=bg_on_data,
        darkness=darkness,
        vmax=vmax,
        vmin=vmin,
    )


def _get_combined_curvature_map(mesh_left, mesh_right):
    """Get combined curvature map from left and right hemisphere maps.
    Only used in _full_brain_info.
    """
    curv_left = load_surf_data(mesh_left)
    curv_right = load_surf_data(mesh_right)
    curv_left_sign = np.sign(curv_left)
    curv_right_sign = np.sign(curv_right)
    curv_left_sign[np.isnan(curv_left)] = 0
    curv_right_sign[np.isnan(curv_right)] = 0
    curv_combined = np.concatenate([curv_left_sign, curv_right_sign])
    return curv_combined


def _full_brain_info(
    volume_img,
    mesh="fsaverage5",
    threshold=None,
    cmap=DEFAULT_DIVERGING_CMAP,
    black_bg=False,
    symmetric_cmap=True,
    bg_on_data=False,
    darkness=0.7,
    vmax=None,
    vmin=None,
    vol_to_surf_kwargs=None,
):
    """Project 3D map on cortex; prepare info to plot both hemispheres.

    This computes the dictionary that gets inserted in the web page,
    which contains encoded meshes, colors, min and max values, and
    background color.

    """
    if vol_to_surf_kwargs is None:
        vol_to_surf_kwargs = {}
    info = {}
    mesh = check_mesh_is_fsaverage(mesh)
    surface_maps = SurfaceImage.from_volume(
        mesh=PolyMesh(
            left=mesh["pial_left"],
            right=mesh["pial_right"],
        ),
        volume_img=volume_img,
        inner_mesh=PolyMesh(
            left=mesh.get("white_left", None),
            right=mesh.get("white_right", None),
        ),
        **vol_to_surf_kwargs,
    )
    colors = colorscale(
        cmap,
        get_data(surface_maps).ravel(),
        threshold,
        symmetric_cmap=symmetric_cmap,
        vmax=vmax,
        vmin=vmin,
    )

    for hemi, surf_map in surface_maps.data.parts.items():
        curv_map = load_surf_data(mesh[f"curv_{hemi}"])
        bg_map = np.sign(curv_map)

        info[f"pial_{hemi}"] = mesh_to_plotly(mesh[f"pial_{hemi}"])
        info[f"inflated_{hemi}"] = mesh_to_plotly(mesh[f"infl_{hemi}"])

        backend = get_surface_backend(DEFAULT_ENGINE)
        info[f"vertexcolor_{hemi}"] = backend._get_vertexcolor(
            surf_map,
            colors["cmap"],
            colors["norm"],
            absolute_threshold=colors["abs_threshold"],
            bg_map=bg_map,
            bg_on_data=bg_on_data,
            darkness=darkness,
        )

    # also add info for both hemispheres
    for mesh_type in ["infl", "pial"]:
        if mesh_type == "infl":
            info["inflated_both"] = mesh_to_plotly(
                combine_hemispheres_meshes(
                    PolyMesh(
                        left=mesh[f"{mesh_type}_left"],
                        right=mesh[f"{mesh_type}_right"],
                    )
                )
            )
        else:
            info[f"{mesh_type}_both"] = mesh_to_plotly(
                combine_hemispheres_meshes(
                    PolyMesh(
                        left=mesh[f"{mesh_type}_left"],
                        right=mesh[f"{mesh_type}_right"],
                    )
                )
            )
    backend = get_surface_backend(DEFAULT_ENGINE)
    info["vertexcolor_both"] = backend._get_vertexcolor(
        get_data(surface_maps),
        colors["cmap"],
        colors["norm"],
        absolute_threshold=colors["abs_threshold"],
        bg_map=_get_combined_curvature_map(
            mesh["curv_left"], mesh["curv_right"]
        ),
        bg_on_data=bg_on_data,
        darkness=darkness,
    )
    info["cmin"], info["cmax"] = float(colors["vmin"]), float(colors["vmax"])
    info["black_bg"] = black_bg
    info["full_brain_mesh"] = True
    info["colorscale"] = colors["colors"]
    return info


def full_brain_info(
    volume_img,
    mesh="fsaverage5",
    threshold=None,
    cmap=DEFAULT_DIVERGING_CMAP,
    black_bg=False,
    symmetric_cmap=True,
    bg_on_data=False,
    darkness=0.7,
    vmax=None,
    vmin=None,
    vol_to_surf_kwargs=None,
):
    """Deprecate public function. See _full_brain_info."""
    warn(
        category=DeprecationWarning,
        message="full_brain_info is a private function and is renamed to "
        "_full_brain_info. Using the deprecated name will raise an error "
        "in release 0.13",
        stacklevel=find_stack_level(),
    )

    return _full_brain_info(
        volume_img,
        mesh=mesh,
        threshold=threshold,
        cmap=cmap,
        black_bg=black_bg,
        symmetric_cmap=symmetric_cmap,
        bg_on_data=bg_on_data,
        darkness=darkness,
        vmax=vmax,
        vmin=vmin,
        vol_to_surf_kwargs=vol_to_surf_kwargs,
    )


def _fill_html_template(info, embed_js=True):
    as_json = json.dumps(info)
    as_html = get_html_template("surface_plot_template.html").safe_substitute(
        {
            "INSERT_STAT_MAP_JSON_HERE": as_json,
            "INSERT_PAGE_TITLE_HERE": info["title"] or "Surface plot",
        }
    )
    as_html = add_js_lib(as_html, embed_js=embed_js)
    return SurfaceView(as_html)


@fill_doc
def view_img_on_surf(
    stat_map_img,
    surf_mesh="fsaverage5",
    threshold=None,
    cmap=DEFAULT_DIVERGING_CMAP,
    black_bg=False,
    vmax=None,
    vmin=None,
    symmetric_cmap=True,
    bg_on_data=False,
    darkness=0.7,
    colorbar=True,
    colorbar_height=0.5,
    colorbar_fontsize=25,
    title=None,
    title_fontsize=25,
    vol_to_surf_kwargs=None,
):
    """Insert a surface plot of a statistical map into an HTML page.

    Parameters
    ----------
    stat_map_img : Niimg-like object, 3D
        See :ref:`extracting_data`.

    surf_mesh : :obj:`str` or :obj:`dict`, default='fsaverage5'
        If a string, it should be one of the following values:
        %(fsaverage_options)s
        If a dictionary, it should have the same structure as those returned by
        nilearn.datasets.fetch_surf_fsaverage, i.e. keys should be 'infl_left',
        'pial_left', 'sulc_left', 'infl_right', 'pial_right', and 'sulc_right',
        containing inflated and pial meshes, and sulcal depth values for left
        and right hemispheres.

    threshold : :obj:`str`, number or None, default=None
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%%", and only values of amplitude above the
        given percentile will be shown.

    %(cmap)s
        default="RdBu_r"

    black_bg : :obj:`bool`, default=False
        If True, image is plotted on a black background. Otherwise on a
        white background.

    %(bg_on_data)s

    %(darkness)s
        Default=1.

    vmax : :obj:`float` or None, default=None
        upper bound for the colorbar. if None, use the absolute max of the
        brain map.

    vmin : :obj:`float` or None, default=None
        min value for mapping colors.
        If `symmetric_cmap` is `True`, `vmin` is always equal to `-vmax` and
        cannot be chosen.
        If `symmetric_cmap` is `False`, `vmin` is equal to the min of the
        image, or 0 when a threshold is used.

    symmetric_cmap : :obj:`bool`, default=True
        Make colormap symmetric (ranging from -vmax to vmax).
        You can set it to False if you are plotting only positive values.

    colorbar : :obj:`bool`, default=True
        Add a colorbar or not.

    colorbar_height : :obj:`float`, default=0.5
        Height of the colorbar, relative to the figure height

    colorbar_fontsize : :obj:`int`, default=25
        Fontsize of the colorbar tick labels.

    title : :obj:`str`, default=None
        Title for the plot.

    title_fontsize : :obj:`int`, default=25
        Fontsize of the title.

    vol_to_surf_kwargs : :obj:`dict`, default=None
        Dictionary of keyword arguments that are passed on to
        :func:`nilearn.surface.vol_to_surf` when extracting a surface from
        the input image. See the function documentation for details.This
        parameter is especially useful when plotting an atlas. See
        https://nilearn.github.io/stable/auto_examples/01_plotting/plot_3d_map_to_surface_projection.html
        Will default to ``{}`` if ``None`` is passed.

    Returns
    -------
    SurfaceView : plot of the stat map.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :

        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.

    See Also
    --------
    nilearn.plotting.view_surf: plot from a surface map on a cortical mesh.

    """
    if vol_to_surf_kwargs is None:
        vol_to_surf_kwargs = {}
    stat_map_img = check_niimg_3d(stat_map_img)
    info = _full_brain_info(
        volume_img=stat_map_img,
        mesh=surf_mesh,
        threshold=threshold,
        cmap=cmap,
        black_bg=black_bg,
        vmax=vmax,
        vmin=vmin,
        bg_on_data=bg_on_data,
        darkness=darkness,
        symmetric_cmap=symmetric_cmap,
        vol_to_surf_kwargs=vol_to_surf_kwargs,
    )
    info["colorbar"] = colorbar
    info["cbar_height"] = colorbar_height
    info["cbar_fontsize"] = colorbar_fontsize
    info["title"] = title
    info["title_fontsize"] = title_fontsize
    return _fill_html_template(info, embed_js=True)


@fill_doc
def view_surf(
    surf_mesh=None,
    surf_map=None,
    bg_map=None,
    hemi=DEFAULT_HEMI,
    threshold=None,
    cmap=DEFAULT_DIVERGING_CMAP,
    black_bg=False,
    vmax=None,
    vmin=None,
    bg_on_data=False,
    darkness=0.7,
    symmetric_cmap=True,
    colorbar=True,
    colorbar_height=0.5,
    colorbar_fontsize=25,
    title=None,
    title_fontsize=25,
):
    """Insert a surface plot of a surface map into an HTML page.

    Parameters
    ----------
    %(surf_mesh)s
        If None is passed, then ``surf_map`` must be a
        :obj:`~nilearn.surface.SurfaceImage` instance and the mesh from that
        :obj:`~nilearn.surface.SurfaceImage` instance will be used.

    surf_map : :obj:`str` or :class:`numpy.ndarray`, \
               or :obj:`~nilearn.surface.SurfaceImage` or None, \
               default=None
        Data to be displayed on the surface :term:`mesh`.
        Can be a file (valid formats are .gii, .mgz, .nii, .nii.gz,
        or Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or
        a Numpy array.
        If None is passed for ``surf_mesh``
        then ``surf_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and its the mesh will be used for plotting.

    %(bg_map)s

    %(hemi)s
        It is only used if ``surf_map`` is :obj:`~nilearn.surface.SurfaceImage`
        and / or ``surf_mesh`` is :obj:`~nilearn.surface.PolyMesh`.
        Otherwise a warning will be displayed.

        .. versionadded:: 0.11.0

    %(bg_on_data)s

    %(darkness)s
        Default=1.

    threshold : :obj:`str`, number or None, default=None
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%%", and only values of amplitude above the
        given percentile will be shown.

    %(cmap)s
        default="RdBu_r"

    black_bg : :obj:`bool`, default=False
        If True, image is plotted on a black background. Otherwise on a
        white background.

    symmetric_cmap : :obj:`bool`, default=True
        Make colormap symmetric (ranging from -vmax to vmax).
        Set it to False if you are plotting a surface atlas.

    vmax : :obj:`float` or None, default=None
        upper bound for the colorbar. if None, use the absolute max of the
        brain map.

    vmin : :obj:`float` or None, default=None
        min value for mapping colors.
        If `symmetric_cmap` is `True`, `vmin` is always equal to `-vmax` and
        cannot be chosen.
        If `symmetric_cmap` is `False`, `vmin` defaults to the min of the
        image, or 0 when a threshold is used.

    colorbar : :obj:`bool`, default=True
        Add a colorbar or not.

    colorbar_height : :obj:`float`, default=0.5
        Height of the colorbar, relative to the figure height.

    colorbar_fontsize : :obj:`int`, default=25
        Fontsize of the colorbar tick labels.

    title : :obj:`str`, default=None
        Title for the plot.

    title_fontsize : :obj:`int`, default=25
        Fontsize of the title.

    Returns
    -------
    SurfaceView : plot of the stat map.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook. Useful methods are :

        - 'resize' to resize the plot displayed in a Jupyter notebook
        - 'save_as_html' to save the plot to a file
        - 'open_in_browser' to save the plot and open it in a web browser.

    See Also
    --------
    nilearn.plotting.view_img_on_surf: Surface plot from a 3D statistical map.
    """
    check_params(locals())
    surf_map, surf_mesh, bg_map = check_surface_plotting_inputs(
        surf_map, surf_mesh, hemi, bg_map, map_var_name="surf_map"
    )

    surf_mesh = load_surf_mesh(surf_mesh)
    if surf_map is None:
        surf_map = np.ones(len(surf_mesh[0]))
    else:
        surf_mesh, surf_map = check_mesh_and_data(surf_mesh, surf_map)
    if bg_map is not None:
        _, bg_map = check_mesh_and_data(surf_mesh, bg_map)
    info = _one_mesh_info(
        surf_map=surf_map,
        surf_mesh=surf_mesh,
        threshold=threshold,
        cmap=cmap,
        black_bg=black_bg,
        bg_map=bg_map,
        bg_on_data=bg_on_data,
        darkness=darkness,
        symmetric_cmap=symmetric_cmap,
        vmax=vmax,
        vmin=vmin,
    )
    info["colorbar"] = colorbar
    info["cbar_height"] = colorbar_height
    info["cbar_fontsize"] = colorbar_fontsize
    info["title"] = title
    info["title_fontsize"] = title_fontsize
    return _fill_html_template(info, embed_js=True)
