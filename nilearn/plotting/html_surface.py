import json
import collections

import numpy as np
import matplotlib as mpl
from matplotlib import cm as mpl_cm

from .._utils.niimg_conversions import check_niimg_3d
from .. import datasets, surface
from . import cm
from .js_plotting_utils import (
    HTMLDocument, colorscale, mesh_to_plotly, get_html_template, add_js_lib,
    to_color_strings)


class SurfaceView(HTMLDocument):
    pass


def _get_vertexcolor(surf_map, cmap, norm,
                     absolute_threshold=None, bg_map=None):
    vertexcolor = cmap(norm(surf_map).data)
    if absolute_threshold is None:
        return to_color_strings(vertexcolor)
    if bg_map is None:
        bg_map = np.ones(len(surf_map)) * .5
        bg_vmin, bg_vmax = 0, 1
    else:
        bg_map = surface.load_surf_data(bg_map)
        bg_vmin, bg_vmax = np.min(bg_map), np.max(bg_map)
    bg_norm = mpl.colors.Normalize(vmin=bg_vmin, vmax=bg_vmax)
    bg_color = mpl_cm.get_cmap('Greys')(bg_norm(bg_map))
    vertexcolor[np.abs(surf_map) < absolute_threshold] = bg_color[
        np.abs(surf_map) < absolute_threshold]
    return to_color_strings(vertexcolor)


def one_mesh_info(surf_map, surf_mesh, threshold=None, cmap=cm.cold_hot,
                  black_bg=False, bg_map=None, symmetric_cmap=True,
                  vmax=None):
    """
    Prepare info for plotting one surface map on a single mesh.


    This computes the dictionary that gets inserted in the web page,
    which contains the encoded mesh, colors, min and max values, and
    background color.

    """
    info = {}
    colors = colorscale(
        cmap, surf_map, threshold, symmetric_cmap=symmetric_cmap, vmax=vmax)
    info['inflated_left'] = mesh_to_plotly(surf_mesh)
    info['vertexcolor_left'] = _get_vertexcolor(
        surf_map, colors['cmap'], colors['norm'],
        colors['abs_threshold'], bg_map)
    info["cmin"], info["cmax"] = float(colors['vmin']), float(colors['vmax'])
    info['black_bg'] = black_bg
    info['full_brain_mesh'] = False
    info['colorscale'] = colors['colors']
    return info


def _check_mesh(mesh):
    if isinstance(mesh, str):
        return datasets.fetch_surf_fsaverage(mesh)
    if not isinstance(mesh, collections.Mapping):
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
                    vmax=None, vol_to_surf_kwargs={}):
    """
    Project 3D map on cortex; prepare info to plot both hemispheres.


    This computes the dictionary that gets inserted in the web page,
    which contains encoded meshes, colors, min and max values, and
    background color.

    """
    info = {}
    mesh = _check_mesh(mesh)
    surface_maps = {
        h: surface.vol_to_surf(volume_img, mesh['pial_{}'.format(h)],
                               **vol_to_surf_kwargs)
        for h in ['left', 'right']
    }
    colors = colorscale(
        cmap, np.asarray(list(surface_maps.values())).ravel(), threshold,
        symmetric_cmap=symmetric_cmap, vmax=vmax)

    for hemi, surf_map in surface_maps.items():
        bg_map = surface.load_surf_data(mesh['sulc_{}'.format(hemi)])
        info['pial_{}'.format(hemi)] = mesh_to_plotly(
            mesh['pial_{}'.format(hemi)])
        info['inflated_{}'.format(hemi)] = mesh_to_plotly(
            mesh['infl_{}'.format(hemi)])

        info['vertexcolor_{}'.format(hemi)] = _get_vertexcolor(
            surf_map, colors['cmap'], colors['norm'],
            colors['abs_threshold'], bg_map)
    info["cmin"], info["cmax"] = float(colors['vmin']), float(colors['vmax'])
    info['black_bg'] = black_bg
    info['full_brain_mesh'] = True
    info['colorscale'] = colors['colors']
    return info


def _fill_html_template(info, embed_js=True):
    as_json = json.dumps(info)
    as_html = get_html_template('surface_plot_template.html').replace(
        'INSERT_STAT_MAP_JSON_HERE', as_json)
    as_html = add_js_lib(as_html, embed_js=embed_js)
    return SurfaceView(as_html)


def view_img_on_surf(stat_map_img, surf_mesh='fsaverage5',
                     threshold=None, cmap=cm.cold_hot,
                     black_bg=False, vmax=None):
    """
    Insert a surface plot of a statistical map into an HTML page.

    Parameters
    ----------
    stat_map_img : Niimg-like object, 3D
        See http://nilearn.github.io/manipulating_images/input_output.html

    surf_mesh : str or dict, optional.
        if 'fsaverage5', use fsaverage5 mesh from nilearn.datasets
        if 'fsaverage', use fsaverage mesh from nilearn.datasets
        if a dictionary, it should have the same structure as those returned by
        nilearn.datasets.fetch_surf_fsaverage, i.e. keys should be 'infl_left',
        'pial_left', 'sulc_left', 'infl_right', 'pial_right', and 'sulc_right',
        containing inflated and pial meshes, and sulcal depth values for left
        and right hemispheres.

    threshold : str, number or None, optional (default=None)
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only values of amplitude above the
        given percentile will be shown.

    cmap : str or matplotlib colormap, optional

    black_bg : bool, optional (default=False)
        If True, image is plotted on a black background. Otherwise on a
        white background.

    vmax : float or None, optional (default=None)
        upper bound for the colorbar. if None, use the absolute max of the
        brain map.

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
        cmap=cmap, black_bg=black_bg, vmax=vmax)
    return _fill_html_template(info, embed_js=True)


def view_surf(surf_mesh, surf_map=None, bg_map=None, threshold=None,
              cmap=cm.cold_hot, black_bg=False, vmax=None,
              symmetric_cmap=True):
    """
    Insert a surface plot of a surface map into an HTML page.

    Parameters
    ----------
    surf_mesh: str or list of two numpy.ndarray
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces.

    surf_map: str or numpy.ndarray, optional.
        Data to be displayed on the surface mesh. Can be a file (valid formats
        are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .curv, .sulc, .annot, .label) or
        a Numpy array

    bg_map: Surface data, optional,
        Background image to be plotted on the mesh underneath the
        surf_data in greyscale, most likely a sulcal depth map for
        realistic shading.

    threshold : str, number or None, optional (default=None)
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only values of amplitude above the
        given percentile will be shown.

    cmap : str or matplotlib colormap, optional
        You might want to change it to 'gnist_ncar' if plotting a
        surface atlas.

    black_bg : bool, optional (default=False)
        If True, image is plotted on a black background. Otherwise on a
        white background.

    symmetric_cmap : bool, optional (default=True)
        Make colormap symmetric (ranging from -vmax to vmax).
        Set it to False if you are plotting a surface atlas.

    vmax : float or None, optional (default=None)
        upper bound for the colorbar. if None, use the absolute max of the
        brain map.

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
    if surf_map is not None:
        surface.check_mesh_and_data(surf_mesh, surf_map)
    if bg_map is not None:
        surface.check_mesh_and_data(surf_mesh, bg_map)
    info = one_mesh_info(
        surf_map=surf_map, surf_mesh=surf_mesh, threshold=threshold,
        cmap=cmap, black_bg=black_bg, bg_map=bg_map,
        symmetric_cmap=symmetric_cmap, vmax=vmax)
    return _fill_html_template(info, embed_js=True)
