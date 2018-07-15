import os
import tempfile
import base64

from nilearn import image, datasets

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


def view_stat_map(stat_map_img, img_name="stat_map"):
    mni = datasets.load_mni152_template()
    stat_map_img = image.resample_to_img(stat_map_img, mni)
    html = _get_html_template('stat_map_template.html')
    html = html.replace('INSERT_STAT_MAP_DATA_HERE', _encode_nii(stat_map_img))
    html = html.replace('INSERT_MNI_DATA_HERE', _encode_nii(mni))
    html = html.replace('INSERT_STAT_MAP_NAME_HERE', img_name)
    return SurfaceView(html)
