import collections.abc
import json
import warnings

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn._utils import fill_doc
from nilearn import surface
from nilearn import datasets
from nilearn.plotting.html_document import HTMLDocument
from nilearn.plotting import cm
from nilearn.plotting.js_plotting_utils import (
    colorscale, mesh_to_plotly, get_html_template, add_js_lib,
    to_color_strings)


class SurfaceView(HTMLDocument):
    pass


def _mix_colormaps(fg, bg):
    """Mixes foreground and background arrays of RGBA colors.

    Parameters
    ----------
    fg : numpy.ndarray
        Array of shape (n, 4), foreground RGBA colors
        represented as floats in [0, 1]
    bg : numpy.ndarray
        Array of shape (n, 4), background RGBA colors
        represented as floats in [0, 1]

    Returns
    -------
    mix : numpy.ndarray
        Array of shape (n, 4), mixed colors
        represented as floats in [0, 1]
    """
    # Adapted from https://stackoverflow.com/questions/726549/algorithm-for-additive-color-mixing-for-rgb-values/727339#727339 # noqa: E501
    if fg.shape != bg.shape:
        raise ValueError(
            "Trying to mix colormaps with different shapes: "
            f"{fg.shape}, {bg.shape}"
        )

    mix = np.empty_like(fg)

    mix[:, 3] = 1 - (1 - fg[:, 3]) * (1 - bg[:, 3])

    for color_index in range(0, 3):
        mix[:, color_index] = (
            fg[:, color_index] * fg[:, 3]
            + bg[:, color_index] * bg[:, 3] * (1 - fg[:, 3])
        ) / mix[:, 3]

    return mix


def _get_vertexcolor(surf_map, cmap, norm,
                     absolute_threshold=None, bg_map=None,
                     bg_on_data=None, darkness=None):
    if bg_map is None:
        bg_data = np.ones(len(surf_map)) * .5
        bg_vmin, bg_vmax = 0, 1
    else:
        bg_data = np.copy(surface.load_surf_data(bg_map))

    # scale background map if need be
    bg_vmin, bg_vmax = np.min(bg_data), np.max(bg_data)
    if (bg_vmin < 0 or bg_vmax > 1):
        bg_norm = mpl.colors.Normalize(vmin=bg_vmin, vmax=bg_vmax)
        bg_data = bg_norm(bg_data)

    if darkness is not None:
        bg_data *= darkness

    bg_colors = plt.get_cmap('Greys')(bg_data)

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

    vertex_colors = _mix_colormaps(surf_colors, bg_colors)

    return to_color_strings(vertex_colors)


def one_mesh_info(surf_map, surf_mesh, threshold=None, cmap=cm.cold_hot,
                  black_bg=False, bg_map=None, symmetric_cmap=True,
                  bg_on_data=False, darkness=.7,
                  vmax=None, vmin=None):
    """Prepare info for plotting one surface map on a single mesh.

    This computes the dictionary that gets inserted in the web page,
    which contains the encoded mesh, colors, min and max values, and
    background color.

    """
    info = {}
    colors = colorscale(
        cmap, surf_map, threshold, symmetric_cmap=symmetric_cmap,
        vmax=vmax, vmin=vmin)
    info['inflated_left'] = mesh_to_plotly(surf_mesh)
    info['vertexcolor_left'] = _get_vertexcolor(
        surf_map, colors['cmap'], colors['norm'],
        absolute_threshold=colors['abs_threshold'], bg_map=bg_map,
        bg_on_data=bg_on_data, darkness=darkness,
    )
    info["cmin"], info["cmax"] = float(colors['vmin']), float(colors['vmax'])
    info['black_bg'] = black_bg
    info['full_brain_mesh'] = False
    info['colorscale'] = colors['colors']
    return info


def _check_mesh(mesh):
    if isinstance(mesh, str):
        return datasets.fetch_surf_fsaverage(mesh)
    if not isinstance(mesh, collections.abc.Mapping):
        raise TypeError("The mesh should be a str or a dictionary, "
                        "you provided: {}.".format(type(mesh).__name__))
    missing = {'pial_left', 'pial_right', 'sulc_left', 'sulc_right',
               'infl_left', 'infl_right'}.difference(mesh.keys())
    if missing:
        raise ValueError(
            "{} {} missing from the provided mesh dictionary".format(
                missing, ('are' if len(missing) > 1 else 'is')))
    return mesh


def full_brain_info(volume_img, mesh='fsaverage5', threshold=None,
                    cmap=cm.cold_hot, black_bg=False, symmetric_cmap=True,
                    bg_on_data=False, darkness=.7,
                    vmax=None, vmin=None, vol_to_surf_kwargs={}):
    """Project 3D map on cortex; prepare info to plot both hemispheres.

    This computes the dictionary that gets inserted in the web page,
    which contains encoded meshes, colors, min and max values, and
    background color.

    """
    info = {}
    mesh = surface.surface._check_mesh(mesh)
    surface_maps = {
        h: surface.vol_to_surf(volume_img, mesh['pial_{}'.format(h)],
                               inner_mesh=mesh.get('white_{}'.format(h), None),
                               **vol_to_surf_kwargs)
        for h in ['left', 'right']
    }
    colors = colorscale(
        cmap, np.asarray(list(surface_maps.values())).ravel(), threshold,
        symmetric_cmap=symmetric_cmap, vmax=vmax, vmin=vmin)

    for hemi, surf_map in surface_maps.items():
        curv_map = surface.load_surf_data(mesh["curv_{}".format(hemi)])
        bg_map = np.sign(curv_map)

        info['pial_{}'.format(hemi)] = mesh_to_plotly(
            mesh['pial_{}'.format(hemi)])
        info['inflated_{}'.format(hemi)] = mesh_to_plotly(
            mesh['infl_{}'.format(hemi)])

        info['vertexcolor_{}'.format(hemi)] = _get_vertexcolor(
            surf_map, colors['cmap'], colors['norm'],
            absolute_threshold=colors['abs_threshold'], bg_map=bg_map,
            bg_on_data=bg_on_data, darkness=darkness,
        )
    info["cmin"], info["cmax"] = float(colors['vmin']), float(colors['vmax'])
    info['black_bg'] = black_bg
    info['full_brain_mesh'] = True
    info['colorscale'] = colors['colors']
    return info


def _fill_html_template(info, embed_js=True):
    as_json = json.dumps(info)
    as_html = get_html_template('surface_plot_template.html').safe_substitute(
        {'INSERT_STAT_MAP_JSON_HERE': as_json,
         'INSERT_PAGE_TITLE_HERE': info["title"] or "Surface plot"})
    as_html = add_js_lib(as_html, embed_js=embed_js)
    return SurfaceView(as_html)


@fill_doc
def view_img_on_surf(stat_map_img, surf_mesh='fsaverage5',
                     threshold=None, cmap=cm.cold_hot,
                     black_bg=False, vmax=None, vmin=None, symmetric_cmap=True,
                     bg_on_data=False, darkness=.7,
                     colorbar=True, colorbar_height=.5, colorbar_fontsize=25,
                     title=None, title_fontsize=25, vol_to_surf_kwargs={}):
    """Insert a surface plot of a statistical map into an HTML page.

    Parameters
    ----------
    stat_map_img : Niimg-like object, 3D
        See :ref:`extracting_data`.

    surf_mesh : str or dict, optional.
        If a string, it should be one of the following values:
        %(fsaverage_options)s
        If a dictionary, it should have the same structure as those returned by
        nilearn.datasets.fetch_surf_fsaverage, i.e. keys should be 'infl_left',
        'pial_left', 'sulc_left', 'infl_right', 'pial_right', and 'sulc_right',
        containing inflated and pial meshes, and sulcal depth values for left
        and right hemispheres.
        Default='fsaverage5'.

    threshold : str, number or None, optional
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%%", and only values of amplitude above the
        given percentile will be shown.

    cmap : str or matplotlib colormap, optional
        Colormap to use. Default=cm.cold_hot.

    black_bg : bool, optional
        If True, image is plotted on a black background. Otherwise on a
        white background. Default=False.

    %(bg_on_data)s
        Default=False.

    %(darkness)s
        Default=1.

    vmax : float or None, optional
        upper bound for the colorbar. if None, use the absolute max of the
        brain map.

    vmin : float or None, optional
        min value for mapping colors.
        If `symmetric_cmap` is `True`, `vmin` is always equal to `-vmax` and
        cannot be chosen.
        If `symmetric_cmap` is `False`, `vmin` defaults to the min of the
        image, or 0 when a threshold is used.

    symmetric_cmap : bool, optional
        Make colormap symmetric (ranging from -vmax to vmax).
        You can set it to False if you are plotting only positive values.
        Default=True.

    colorbar : bool, optional
        Add a colorbar or not. Default=True.

    colorbar_height : float, optional
        Height of the colorbar, relative to the figure height. Default=0.5.

    colorbar_fontsize : int, optional
        Fontsize of the colorbar tick labels. Default=25.

    title : str, optional
        Title for the plot.

    title_fontsize : int, optional
        Fontsize of the title. Default=25.

    vol_to_surf_kwargs : dict, optional
        Dictionary of keyword arguments that are passed on to
        :func:`nilearn.surface.vol_to_surf` when extracting a surface from
        the input image. See the function documentation for details.This
        parameter is especially useful when plotting an atlas. See
        https://nilearn.github.io/stable/auto_examples/01_plotting/plot_3d_map_to_surface_projection.html

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
    stat_map_img = check_niimg_3d(stat_map_img)
    info = full_brain_info(
        volume_img=stat_map_img, mesh=surf_mesh, threshold=threshold,
        cmap=cmap, black_bg=black_bg, vmax=vmax, vmin=vmin,
        bg_on_data=bg_on_data, darkness=darkness,
        symmetric_cmap=symmetric_cmap, vol_to_surf_kwargs=vol_to_surf_kwargs
    )
    info['colorbar'] = colorbar
    info['cbar_height'] = colorbar_height
    info['cbar_fontsize'] = colorbar_fontsize
    info['title'] = title
    info['title_fontsize'] = title_fontsize
    return _fill_html_template(info, embed_js=True)


def view_surf(surf_mesh, surf_map=None, bg_map=None, threshold=None,
              cmap=cm.cold_hot, black_bg=False, vmax=None, vmin=None,
              bg_on_data=False, darkness=.7, symmetric_cmap=True,
              colorbar=True, colorbar_height=.5, colorbar_fontsize=25,
              title=None, title_fontsize=25):
    """Insert a surface plot of a surface map into an HTML page.

    Parameters
    ----------
    surf_mesh : str or list of two numpy.ndarray
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces.

    surf_map : str or numpy.ndarray, optional
        Data to be displayed on the surface mesh. Can be a file (valid formats
        are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label) or
        a Numpy array

    bg_map : str or numpy.ndarray, optional
        Background image to be plotted on the mesh underneath the
        surf_data in greyscale, most likely a sulcal depth map for
        realistic shading.
        If the map contains values outside [0, 1], it will be
        rescaled such that all values are in [0, 1]. Otherwise,
        it will not be modified.
        Default=None.

    %(bg_on_data)s
        Default=False.

    %(darkness)s
        Default=1.

    threshold : str, number or None, optional
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only values of amplitude above the
        given percentile will be shown.

    cmap : str or matplotlib colormap, optional
        You might want to change it to 'gnist_ncar' if plotting a
        surface atlas. Default=cm.cold_hot.

    black_bg : bool, optional
        If True, image is plotted on a black background. Otherwise on a
        white background. Default=False.

    symmetric_cmap : bool, optional
        Make colormap symmetric (ranging from -vmax to vmax).
        Set it to False if you are plotting a surface atlas. Default=True.

    vmax : float or None, optional
        upper bound for the colorbar. if None, use the absolute max of the
        brain map.

    vmin : float or None, optional
        min value for mapping colors.
        If `symmetric_cmap` is `True`, `vmin` is always equal to `-vmax` and
        cannot be chosen.
        If `symmetric_cmap` is `False`, `vmin` defaults to the min of the
        image, or 0 when a threshold is used.

    colorbar : bool, optional
        Add a colorbar or not. Default=True.

    colorbar_height : float, optional
        Height of the colorbar, relative to the figure height. Default=0.5.

    colorbar_fontsize : int, optional
        Fontsize of the colorbar tick labels. Default=25.

    title : str, optional
        Title for the plot.

    title_fontsize : int, optional
        Fontsize of the title. Default=25.

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
    surf_mesh = surface.load_surf_mesh(surf_mesh)
    if surf_map is None:
        surf_map = np.ones(len(surf_mesh[0]))
    else:
        surf_mesh, surf_map = surface.check_mesh_and_data(
            surf_mesh, surf_map)
    if bg_map is not None:
        _, bg_map = surface.check_mesh_and_data(surf_mesh, bg_map)
    info = one_mesh_info(
        surf_map=surf_map, surf_mesh=surf_mesh, threshold=threshold,
        cmap=cmap, black_bg=black_bg, bg_map=bg_map,
        bg_on_data=bg_on_data, darkness=darkness,
        symmetric_cmap=symmetric_cmap, vmax=vmax, vmin=vmin)
    info['colorbar'] = colorbar
    info['cbar_height'] = colorbar_height
    info['cbar_fontsize'] = colorbar_fontsize
    info['title'] = title
    info['title_fontsize'] = title_fontsize
    return _fill_html_template(info, embed_js=True)
