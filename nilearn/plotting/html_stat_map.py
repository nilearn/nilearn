"""
Visualizing 3D stat maps in a Papaya viewer
"""
import os
import tempfile
import base64

import numpy as np

from .. import image, datasets
from .._utils.extmath import fast_abs_percentile
from .._utils.param_validation import check_threshold
from .js_plotting_utils import get_html_template, HTMLDocument


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


def view_stat_map(stat_map_img, threshold=None, bg_img=None, vmax=None):
    """
    Insert a surface plot of a surface map into an HTML page.

    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image

    threshold : str, number or None, optional (default=None)
        If None, no thresholding.
        If it is a number only values of amplitude greater
        than threshold will be shown.
        If it is a string it must finish with a percent sign,
        e.g. "25.3%", and only values of amplitude above the
        given percentile will be shown.

    bg_img : Niimg-like object, optional (default=None)
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the stat map will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.

    vmax : float, optional (default=None)
        Upper bound for plotting

    Returns
    -------
    StatMapView : plot of the stat map.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook.

    """
    if bg_img is None:
        bg_img = datasets.load_mni152_template()
        bg_mask = datasets.load_mni152_brain_mask()
    else:
        bg_img = image.load_img(bg_img)
        bg_mask = image.new_img_like(bg_img, bg_img.get_data() != 0)
    stat_map_img = image.resample_to_img(stat_map_img, bg_img)
    stat_map_img = image.new_img_like(
        stat_map_img, stat_map_img.get_data() * bg_mask.get_data())
    if threshold is None:
        abs_threshold = 'null'
    else:
        abs_threshold = check_threshold(
            threshold, stat_map_img.get_data(), fast_abs_percentile)
        abs_threshold = str(abs_threshold)
    if vmax is None:
        vmax = np.abs(stat_map_img.get_data()).max()
    html = get_html_template('stat_map_template.html')
    html = html.replace('INSERT_STAT_MAP_DATA_HERE', _encode_nii(stat_map_img))
    html = html.replace('INSERT_MNI_DATA_HERE', _encode_nii(bg_img))
    html = html.replace('INSERT_ABS_MIN_HERE', abs_threshold)
    html = html.replace('INSERT_ABS_MAX_HERE', str(vmax))
    return StatMapView(html)
