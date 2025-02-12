"""Handle plotting of surfaces for html rendering."""

import base64
import json
from warnings import warn

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps

from nilearn._utils import fill_doc
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn.plotting import cm
from nilearn.plotting._utils import (
    check_surface_plotting_inputs,
    sanitize_hemi_for_surface_image,
)
from nilearn.plotting.html_document import HTMLDocument
from nilearn.plotting.js_plotting_utils import (
    add_js_lib,
    colorscale,
    colorscale_niivue,
    get_html_template,
    mesh_to_plotly,
    to_color_strings,
)
from nilearn.surface.surface import (
    PolyMesh,
    SurfaceImage,
    _data_to_gifti,
    _mesh_to_gifti,
    check_mesh_and_data,
    check_mesh_is_fsaverage,
    combine_hemispheres_meshes,
    get_data,
    load_surf_data,
    load_surf_mesh,
)


class SurfaceView(HTMLDocument):  # noqa: D101
    pass


def _check_engine(engine):
    """Check engine is valid."""
    valid_engines = ["plotly", "niivue"]
    if engine not in valid_engines:
        raise ValueError(
            f"Invalid engine {engine}. Valid engines are {valid_engines}."
        )
    return engine


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
    info["vertexcolor_both"] = get_vertexcolor(
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


def _matplotlib_cm_to_niivue_cm(cmap):
    """Convert matplotlib colormap to niivue colormap.

    Parameters
    ----------
    cmap_name : str
        Name of the colormap to convert.

    Returns
    -------
    cmap : dict of list
        Converted colormap, with keys "R", "G", "B", "A", "I".
    """
    if not isinstance(cmap, (mpl.colors.Colormap, str)):
        warn(
            f"'cmap' must be a str or a Colormap. Got {type(cmap)}",
            stacklevel=4,
        )
        return None

    name = None
    spec = None
    reverse = False

    if isinstance(cmap, str):
        if cmap not in list(colormaps) and cmap not in cm._cmap_d:
            warn(f"Colormap not available: {cmap}", stacklevel=4)
            return None

        name = cmap

        spec = plt.get_cmap(name)

    else:
        spec = cmap
        name = cmap.name

    if name.endswith("_r"):
        name = name[:-2]
        reverse = True

    n_nodes = 255
    colors = spec(np.linspace(0, 1, 255))

    js = {"R": [], "G": [], "B": [], "A": [], "I": []}
    js["R"] = (255 * colors[..., 0]).astype(int).tolist()
    js["G"] = (255 * colors[..., 1]).astype(int).tolist()
    js["B"] = (255 * colors[..., 2]).astype(int).tolist()
    js["A"] = [64 for _ in range(n_nodes)]
    js["I"] = list(range(n_nodes))

    if reverse:
        for k in js:
            js[k] = js[k][::-1]

    return js


def _one_mesh_info_niivue(
    surf_map,
    surf_mesh,
    threshold=None,
    bg_map=None,
    cmap=None,
    colorbar=None,
):
    """Build dict for plotting one surface map on a single mesh."""
    info = {}

    # Handle mesh
    surf_mesh_gifti = _mesh_to_gifti(surf_mesh.coordinates, surf_mesh.faces)
    info["surf_mesh"] = base64.b64encode(surf_mesh_gifti.to_bytes()).decode(
        "UTF-8"
    )

    # Handle surface data
    gii = _data_to_gifti(surf_map)
    info["surf_map"] = base64.b64encode(gii.to_bytes()).decode("UTF-8")

    info["cmap"] = _matplotlib_cm_to_niivue_cm(cmap)

    if isinstance(colorbar, bool):
        info["colorbar"] = str(colorbar).lower()

    threshold = colorscale_niivue(surf_map, threshold=threshold)
    info["threshold"] = threshold

    # Handle background map
    if bg_map is not None:
        gii = _data_to_gifti(bg_map)
        info["bg_map"] = base64.b64encode(gii.to_bytes()).decode("UTF-8")

    return info


def one_mesh_info(
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
    """Deprecate public function. See _one_mesh_info."""
    warn(
        category=DeprecationWarning,
        message="one_mesh_info is a private function and is renamed "
        "to _one_mesh_info. Using the deprecated name will "
        "raise an error in release 0.13",
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
    cmap=cm.cold_hot,
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

        info[f"vertexcolor_{hemi}"] = get_vertexcolor(
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
    info["vertexcolor_both"] = get_vertexcolor(
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
    cmap=cm.cold_hot,
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
    as_html = add_js_lib(
        as_html, libraries=["plotly", "jquery"], embed_js=embed_js
    )
    return SurfaceView(as_html)


def _fill_html_template_niivue(info, embed_js=True):
    as_html = get_html_template(
        "surface_plot_template_niivue.html"
    ).safe_substitute(
        {
            "INSERT_SURF_MAP_BASE64_HERE": info["surf_map"],
            "INSERT_SURF_COLORMAP_HERE": info["cmap"],
            "INSERT_MESH_BASE64_HERE": info["surf_mesh"],
            "INSERT_BG_MAP_BASE64_HERE": info["bg_map"],
            "INSERT_COLORBAR_HERE": info["colorbar"],
            "INSERT_THRESHOLD_HERE": json.dumps(info["threshold"]),
            "INSERT_PAGE_TITLE_HERE": info["title"] or "Surface plot",
        }
    )
    as_html = add_js_lib(as_html, libraries=["niivue"], embed_js=embed_js)
    return SurfaceView(as_html)


@fill_doc
def view_img_on_surf(
    stat_map_img,
    surf_mesh="fsaverage5",
    threshold=None,
    cmap="RdBu_r",
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
    hemi=None,
    threshold=None,
    cmap="RdBu_r",
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
    engine="plotly",
):
    """Insert a surface plot of a surface map into an HTML page.

    Parameters
    ----------
    surf_mesh : :obj:`str` or :obj:`list` of two :class:`numpy.ndarray`, \
                or a :obj:`~nilearn.surface.InMemoryMesh`, \
                or a :obj:`~nilearn.surface.PolyMesh`, or None, default=None
        Surface :term:`mesh` geometry, can be a file
        (valid formats are .gii or Freesurfer specific files
        such as .orig, .pial, .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the :term:`mesh` vertices, the second containing the indices
        (into coords) of the :term:`mesh` :term:`faces`.
        or a :obj:`~nilearn.surface.InMemoryMesh` object with
        "coordinates" and "faces" attributes,
        or a :obj:`~nilearn.surface.PolyMesh` object,
        or None.
        If None is passed, then ``surf_map``
        must be a :obj:`~nilearn.surface.SurfaceImage` instance
        and the mesh from that :obj:`~nilearn.surface.SurfaceImage` instance
        will be used.

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

    bg_map : :obj:`str` or :class:`numpy.ndarray`, default=None
        Background image to be plotted on the :term:`mesh` underneath
        the surf_data in grayscale, most likely a sulcal depth map for
        realistic shading.
        If the map contains values outside [0, 1],
        it will be rescaled such that all values are in [0, 1].
        Otherwise, it will not be modified.

    hemi : {"left", "right", "both", None}, default=None
        Hemisphere to display in case a :obj:`~nilearn.surface.SurfaceImage`
        is passed as ``surf_map``
        and / or if :obj:`~nilearn.surface.PolyMesh`
        is passed as ``surf_mesh``.
        In these cases, if ``hemi`` is set to None, it will default to "left".

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

    engine : {'plotly', 'niivue'}, optional
        Engine to use for plotting. Default='plotly'.

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
    hemi = sanitize_hemi_for_surface_image(hemi, surf_map, surf_mesh)
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

    _check_engine(engine)

    if engine == "plotly":
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

    elif engine == "niivue":
        info = _one_mesh_info_niivue(
            surf_map=surf_map,
            surf_mesh=surf_mesh,
            bg_map=bg_map,
            cmap=cmap,
            colorbar=colorbar,
            threshold=threshold,
        )
        info["title"] = title
        return _fill_html_template_niivue(info, embed_js=True)
