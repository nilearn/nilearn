import os
import tempfile
import base64

import numpy as np
from nilearn import image, datasets

from .._utils.extmath import fast_abs_percentile
from .._utils.param_validation import check_threshold
from .html_surface import _get_html_template, SurfaceView


def _encode_nii(img):
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


def view_stat_map(stat_map_img, threshold=None,
                  bg_img=None, vmax=None, img_name="stat_map"):
    if bg_img is None:
        bg_img = datasets.load_mni152_template()
        bg_mask = datasets.load_mni152_brain_mask()
    else:
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
    html = _get_html_template('stat_map_template.html')
    html = html.replace('INSERT_STAT_MAP_DATA_HERE', _encode_nii(stat_map_img))
    html = html.replace('INSERT_MNI_DATA_HERE', _encode_nii(bg_img))
    html = html.replace('INSERT_STAT_MAP_NAME_HERE', img_name)
    html = html.replace('INSERT_ABS_MIN_HERE', abs_threshold)
    html = html.replace('INSERT_ABS_MAX_HERE', str(vmax))
    return SurfaceView(html)
