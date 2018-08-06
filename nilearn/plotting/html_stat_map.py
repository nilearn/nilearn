"""
Visualizing 3D stat maps in a Papaya viewer
"""
import os
import tempfile
import base64
import json

import numpy as np

from .. import image, datasets
from .._utils import check_niimg_3d
from .._utils.extmath import fast_abs_percentile
from .._utils.param_validation import check_threshold
from .js_plotting_utils import get_html_template, HTMLDocument, colorscale


def _encode_nii(img):
    # downcast to float32 to save memory
    img = image.load_img(img, dtype='auto')
    fd, temp = tempfile.mkstemp(suffix='.nii')
    os.close(fd)
    try:
        img.to_filename(temp)
        with open(temp, 'rb') as f:
            binary = f.read()
            encoded = base64.b64encode(binary).decode('utf-8')
            return encoded
    finally:
        os.remove(temp)


def _decode_nii(encoded):
    fd, temp = tempfile.mkstemp(suffix='.nii')
    os.close(fd)
    try:
        with open(temp, 'wb') as f:
            f.write(base64.b64decode(encoded.encode('utf-8')))
        img = image.load_img(temp)
        # The asarray/copy is to get rid of memmapping
        loaded = image.new_img_like(img,
                                    np.asarray(img.get_data()).copy())
        del img
        return loaded
    finally:
        os.remove(temp)


class StatMapView(HTMLDocument):
    pass


def _to_papaya(cmap, norm, abs_min, vmax, symmetric_cmap):
    """Create colormaps ('lookup tables') given to papaya viewer"""
    n_colors = 100
    if not symmetric_cmap:
        x = np.linspace(norm(abs_min), norm(vmax), n_colors)
        rgb = cmap(x)[:, :3]
        return {'positive_cmap': _rgb_to_papaya(x, rgb),
                'negative_cmap': None}
    papaya = {}
    x = np.linspace(norm(-vmax), norm(-abs_min), n_colors)
    y = np.linspace(0, 1, n_colors)
    rgb = cmap(x)[::-1, :3]
    papaya['negative_cmap'] = _rgb_to_papaya(y, rgb)
    x = np.linspace(norm(abs_min), norm(vmax), n_colors)
    rgb = cmap(x)[:, :3]
    papaya['positive_cmap'] = _rgb_to_papaya(y, rgb)
    return papaya


def _rgb_to_papaya(x, cmap):
    """Transform (values, rgb array) to format expected by papaya"""
    cmap = [[float(m)] + list(map(float, col)) for m, col in zip(x, cmap)]
    return cmap


def view_stat_map(stat_map_img, threshold=None, bg_img='MNI152',
                  vmax=None, cmap='cold_hot', symmetric_cmap=True):
    """
    Insert a surface plot of a surface map into an HTML page.

    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image. Should be 3D or
        4D with exactly one time point (i.e. stat_map_img.shape[-1] = 1)

    threshold : str, number or None, optional (default=None)
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only values of amplitude above the
        given percentile will be shown.

    bg_img : Niimg-like object, optional (default='MNI152')
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the stat map will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.

    vmax : float, optional (default=None)
        Upper bound for plotting

    cmap : str or matplotlib colormap, optional

    symmetric_cmap : bool, optional (default=True)
        Make colormap symmetric (ranging from -vmax to vmax).
        Set it to False if you are plotting an atlas or an anatomical image.

    Returns
    -------
    StatMapView : plot of the stat map.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook.

    """
    stat_map_img = check_niimg_3d(stat_map_img, dtype='auto')
    if bg_img == 'MNI152':
        bg_img = datasets.load_mni152_template()
    if bg_img is not None and bg_img is not False:
        bg_img = image.load_img(bg_img)
        stat_map_img = image.resample_to_img(stat_map_img, bg_img)
        bg_mask = image.new_img_like(bg_img, bg_img.get_data() != 0)
        stat_map_img = image.new_img_like(
            stat_map_img, stat_map_img.get_data() * bg_mask.get_data())
        encoded_bg = '"{}"'.format(_encode_nii(bg_img))
    else:
        encoded_bg = 'null'
    colors = colorscale(
        cmap, stat_map_img.get_data().ravel(),
        threshold=threshold, symmetric_cmap=symmetric_cmap, vmax=vmax)
    abs_threshold = colors['abs_threshold']
    abs_threshold = 0. if abs_threshold is None else float(abs_threshold)
    papaya_cmaps = _to_papaya(
        colors['cmap'], colors['norm'],
        abs_threshold, colors['vmax'], colors['symmetric_cmap'])
    stat_map_params = {
        'min': abs_threshold, 'max': float(colors['vmax']),
        'cmap': papaya_cmaps, 'symmetric': colors['symmetric_cmap']}
    html = get_html_template('stat_map_template.html')
    html = html.replace(
        'INSERT_STAT_MAP_DATA_HERE', '"{}"'.format(_encode_nii(stat_map_img)))
    html = html.replace('INSERT_MNI_DATA_HERE', encoded_bg)
    html = html.replace('INSERT_STAT_MAP_PARAMS_HERE',
                        json.dumps(stat_map_params))
    return StatMapView(html)
