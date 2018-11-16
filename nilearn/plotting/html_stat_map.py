"""
Visualizing 3D stat maps in a Brainsprite viewer
"""
import os
import json
from io import BytesIO
from string import Template

import numpy as np
from matplotlib.image import imsave

from nibabel.affines import apply_affine

from ..image import resample_to_img, new_img_like, reorder_img
from .js_plotting_utils import get_html_template, HTMLDocument, colorscale
from ..plotting import cm
from ..plotting.find_cuts import find_xyz_cut_coords
from ..plotting.img_plotting import _load_anat
from .._utils.niimg_conversions import check_niimg_3d
from .._utils.param_validation import check_threshold
from .._utils.extmath import fast_abs_percentile
from .._utils.niimg import _safe_get_data
from .._utils.compat import _encodebytes
from ..datasets import load_mni152_template


def _data_to_sprite(data):
    """ Convert a 3D array into a sprite of sagittal slices.
        Returns: sprite (2D numpy array)
        If each sagittal slice is nz (height) x ny (width) pixels, the sprite
        size is (M x nz) x (N x ny), where M and N are computed to be roughly
        equal. All slices are pasted together row by row, from top left to
        bottom right. The last row is completed with empty slices.
    """

    nx, ny, nz = data.shape
    nrows = int(np.ceil(np.sqrt(nx)))
    ncolumns = int(np.ceil(nx / float(nrows)))

    sprite = np.zeros((nrows * nz, ncolumns * ny))
    indrow, indcol = np.where(np.ones((nrows, ncolumns)))

    for xx in range(nx):
        # we need to flip the image in the x axis
        sprite[(indrow[xx] * nz):((indrow[xx] + 1) * nz), (indcol[xx] * ny):
               ((indcol[xx] + 1) * ny)] = data[xx, :, ::-1].transpose()

    return sprite


def _threshold_data(data, threshold=None):
    """ Threshold a data array.
        Returns: data (masked array), threshold (updated)
    """
    # If threshold is None, do nothing
    if threshold is None:
        return data, threshold

    # Deal with automatic settings of plot parameters
    if threshold == 'auto':
        # Threshold epsilon below a percentile value, to be sure that some
        # voxels pass the threshold
        threshold = fast_abs_percentile(data) - 1e-5

    # Threshold
    threshold = check_threshold(threshold, data,
                                percentile_func=fast_abs_percentile,
                                name='threshold')

    # Mask data
    if threshold == 0:
        data = np.ma.masked_equal(data, 0, copy=False)
    else:
        data = np.ma.masked_inside(data, -threshold, threshold, copy=False)
    return data, threshold


def _save_sprite(data, output_sprite, vmax, vmin, mask=None, cmap='Greys',
                 format='png'):
    """ Generate a sprite from a 3D Niimg-like object.
        Returns: sprite
    """

    # Create sprite
    sprite = _data_to_sprite(data)

    # Mask the sprite
    if mask is not None:
        mask = _data_to_sprite(mask)
        sprite = np.ma.array(sprite, mask=mask)

    # Save the sprite
    imsave(output_sprite, sprite, vmin=vmin, vmax=vmax, cmap=cmap,
           format=format)

    return sprite


def _bytesIO_to_base64(handle_io):
    """ Encode the content of a bytesIO virtual file as base64.
        Also closes the file.
        Returns: data
    """
    handle_io.seek(0)
    data = _encodebytes(handle_io.read()).decode('utf-8')
    handle_io.close()
    return data


def _save_cm(output_cmap, cmap, format='png', n_colors=256):
    """ Save the colormap of an image as an image file.
    """

    # save the colormap
    data = np.arange(0., n_colors) / (n_colors - 1.)
    data = data.reshape([1, n_colors])
    imsave(output_cmap, data, cmap=cmap, format=format)


class StatMapView(HTMLDocument):
    pass


def _mask_stat_map(stat_map_img, threshold=None):
    """ Load a stat map and apply a threshold.
        Returns: mask_img, stat_map_img, data, threshold
    """
    # Load stat map
    stat_map_img = check_niimg_3d(stat_map_img, dtype='auto')
    data = _safe_get_data(stat_map_img, ensure_finite=True)

    # threshold the stat_map
    if threshold is not None:
        data, threshold = _threshold_data(data, threshold)
        mask_img = new_img_like(stat_map_img, data.mask, stat_map_img.affine)
    else:
        mask_img = new_img_like(stat_map_img, np.zeros(data.shape),
                                stat_map_img.affine)
    return mask_img, stat_map_img, data, threshold


def _load_bg_img(stat_map_img, bg_img='MNI152', black_bg='auto', dim='auto'):
    """ Load and resample bg_img in an isotropic resolution,
        with a positive diagonal affine matrix.
        Returns: bg_img, bg_min, bg_max, black_bg
    """
    if (bg_img is None or bg_img is False) and black_bg == 'auto':
        black_bg = False

    if bg_img is not None and bg_img is not False:
        if isinstance(bg_img, str) and bg_img == "MNI152":
            bg_img = load_mni152_template()
        bg_img, black_bg, bg_min, bg_max = _load_anat(bg_img, dim=dim,
                                                      black_bg=black_bg)
    else:
        bg_img = new_img_like(stat_map_img, np.zeros(stat_map_img.shape),
                              stat_map_img.affine)
        bg_min = 0
        bg_max = 0
    bg_img = reorder_img(bg_img, resample='nearest')
    return bg_img, bg_min, bg_max, black_bg


def _resample_stat_map(stat_map_img, bg_img, mask_img,
                       resampling_interpolation='continuous'):
    """ Resample the stat map and mask to the background.
        Returns: stat_map_img, mask_img
    """
    stat_map_img = resample_to_img(stat_map_img, bg_img,
                                   interpolation=resampling_interpolation)
    mask_img = resample_to_img(mask_img, bg_img, fill_value=1,
                               interpolation='nearest')

    return stat_map_img, mask_img


def _json_view_params(shape, affine, vmin, vmax, cut_slices, black_bg=False,
                      opacity=1, draw_cross=True, annotate=True, title=None,
                      colorbar=True, value=True):
    """ Create a dictionary with all the brainsprite parameters.
        Returns: params
    """

    # Set color parameters
    if black_bg:
        cfont = '#FFFFFF'
        cbg = '#000000'
    else:
        cfont = '#000000'
        cbg = '#FFFFFF'

    # Deal with limitations of json dump regarding types
    if type(vmin).__module__ == 'numpy':
        vmin = vmin.tolist()  # json does not deal with numpy array
    if type(vmax).__module__ == 'numpy':
        vmax = vmax.tolist()  # json does not deal with numpy array

    params = {'canvas': '3Dviewer',
              'sprite': 'spriteImg',
              'nbSlice': {'X': shape[0],
                          'Y': shape[1],
                          'Z': shape[2]},
              'overlay': {'sprite': 'overlayImg',
                          'nbSlice': {'X': shape[0],
                                      'Y': shape[1],
                                      'Z': shape[2]},
                          'opacity': opacity},
              'colorBackground': cbg,
              'colorFont': cfont,
              'crosshair': draw_cross,
              'affine': affine.tolist(),
              'flagCoordinates': annotate,
              'title': title,
              'flagValue': value,
              'numSlice': {'X': cut_slices[0] - 1,
                           'Y': cut_slices[1] - 1,
                           'Z': cut_slices[2] - 1}}

    if colorbar:
        params['colorMap'] = {'img': 'colorMap',
                              'min': vmin,
                              'max': vmax}
    return params


def _json_view_size(params):
    """ Define the size of the viewer.
        Returns: width_view, height_view
    """
    # slices_width = sagittal_width (y) + coronal_width (x) + axial_width (x)
    slices_width = params['nbSlice']['Y'] + 2 * params['nbSlice']['X']

    # slices_height = max of sagittal_height (z), coronal_height (z), and
    # axial_height (y).
    # Also add 20% extra height for annotation and margin
    slices_height = np.max([params['nbSlice']['Y'], params['nbSlice']['Z']])
    slices_height = 1.20 * slices_height

    # Get the final size of the viewer
    width_view = 600
    ratio = slices_height / slices_width
    height_view = np.ceil(ratio * width_view)

    return width_view, height_view


def _json_view_data(bg_img, stat_map_img, mask_img, bg_min, bg_max, colors,
                    cmap, colorbar):
    """ Create a json-like viewer object, and populate with base64 data.
        Returns: json_view
    """
    # Initialise brainsprite data structure
    json_view = dict.fromkeys(['bg_base64', 'stat_map_base64', 'cm_base64',
                              'params', 'js_jquery', 'js_brainsprite'])

    # Create a base64 sprite for the background
    bg_sprite = BytesIO()
    bg_data = _safe_get_data(bg_img, ensure_finite=True)
    _save_sprite(bg_data, bg_sprite, bg_max, bg_min, None, 'gray', 'png')
    json_view['bg_base64'] = _bytesIO_to_base64(bg_sprite)

    # Create a base64 sprite for the stat map
    stat_map_sprite = BytesIO()
    data = _safe_get_data(stat_map_img, ensure_finite=True)
    mask = _safe_get_data(mask_img, ensure_finite=True)
    _save_sprite(data, stat_map_sprite, colors['vmax'], colors['vmin'],
                 mask, cmap, 'png')
    json_view['stat_map_base64'] = _bytesIO_to_base64(stat_map_sprite)

    # Create a base64 colormap
    if colorbar:
        stat_map_cm = BytesIO()
        _save_cm(stat_map_cm, colors['cmap'], 'png')
        json_view['cm_base64'] = _bytesIO_to_base64(stat_map_cm)
    else:
        json_view['cm_base64'] = ''

    return json_view


def _json_view_to_html(json_view):
    """ Fill a brainsprite html template with relevant parameters and data.
        Returns: html_view
    """

    # Fix the size of the viewer
    width, height = _json_view_size(json_view['params'])

    # Populate all missing keys with html-ready data
    json_view['params'] = json.dumps(json_view['params'])
    js_dir = os.path.join(os.path.dirname(__file__), 'data', 'js')
    with open(os.path.join(js_dir, 'jquery.min.js')) as f:
        json_view['js_jquery'] = f.read()
    with open(os.path.join(js_dir, 'brainsprite.min.js')) as f:
        json_view['js_brainsprite'] = f.read()

    # Load the html template, and plug in all the data
    html_view = get_html_template('stat_map_template.html')
    html_view = Template(html_view).safe_substitute(json_view)

    return StatMapView(html_view, width=width, height=height)


def _get_cut_slices(stat_map_img, cut_coords=None, threshold=None):
    """ For internal use.
        Find slice numbers for the cut.
        Based on find_xyz_cut_coords
    """
    # Select coordinates for the cut
    if cut_coords is None:
        cut_coords = find_xyz_cut_coords(
            stat_map_img, activation_threshold=threshold)

    # Convert cut coordinates into cut slices
    try:
        cut_slices = apply_affine(np.linalg.inv(stat_map_img.affine),
                                  cut_coords)
    except ValueError:
        raise ValueError(
            "The input given for display_mode='ortho' needs to be "
            "a list of 3d world coordinates in (x, y, z). "
            "You provided cut_coords={0}".format(cut_coords))
    except IndexError:
        raise ValueError(
            "The input given for display_mode='ortho' needs to be "
            "a list of 3d world coordinates in (x, y, z). "
            "You provided single cut, cut_coords={0}".format(cut_coords))

    return cut_slices


def view_img(stat_map_img, bg_img='MNI152',
             cut_coords=None,
             colorbar=True,
             title=None,
             threshold=1e-6,
             annotate=True,
             draw_cross=True,
             black_bg='auto',
             cmap=cm.cold_hot,
             symmetric_cmap=True,
             dim='auto',
             vmax=None,
             vmin=None,
             resampling_interpolation='continuous',
             opacity=1,
             **kwargs
             ):
    """
    Interactive html viewer of a statistical map, with optional background

    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image. Can be either a 3D volume or a 4D volume
        with exactly one time point.
    bg_img : Niimg-like object (default='MNI152')
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the stat map will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".
    cut_coords : None, or a tuple of floats (default None)
        The MNI coordinates of the point where the cut is performed
        as a 3-tuple: (x, y, z). If None is given, the cuts are calculated
        automaticaly.
    colorbar : boolean, optional (default True)
        If True, display a colorbar on top of the plots.
    title : string or None (default=None)
        The title displayed on the figure (or None: no title).
    threshold : string, number or None  (default=1e-6)
        If None is given, the image is not thresholded.
        If a string of the form "90%" is given, use the 90-th percentile of
        the absolute value in the image.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        automatically.
    annotate : boolean (default=True)
        If annotate is True, current cuts and value of the map are added to the
        viewer.
    draw_cross : boolean (default=True)
        If draw_cross is True, a cross is drawn on the plot to
        indicate the cuts.
    black_bg : boolean (default='auto')
        If True, the background of the image is set to be black.
        Otherwise, a white background is used.
        If set to auto, an educated guess is made to find if the background
        is white or black.
    cmap : matplotlib colormap, optional
        The colormap for specified image.
    symmetric_cmap : bool, optional (default=True)
        True: make colormap symmetric (ranging from -vmax to vmax).
        False: the colormap will go from the minimum of the volume to vmax.
        Set it to False if you are plotting a positive volume, e.g. an atlas
        or an anatomical image.
    dim : float, 'auto' (default='auto')
        Dimming factor applied to background image. By default, automatic
        heuristics are applied based upon the background image intensity.
        Accepted float values, where a typical scan is between -2 and 2
        (-2 = increase constrast; 2 = decrease contrast), but larger values
        can be used for a more pronounced effect. 0 means no dimming.
    vmax : float, or None (default=None)
        max value for mapping colors.
        If vmax is None and symmetric_cmap is True, vmax is the max
        absolute value of the volume.
        If vmax is None and symmetric_cmap is False, vmax is the max
        value of the volume.
    vmin : float, or None (default=None)
        min value for mapping colors.
        If `symmetric_cmap` is `True`, `vmin` is always equal to `-vmax` and
        cannot be chosen.
        If `symmetric_cmap` is `False`, `vmin` defaults to the min of the
        image, or 0 when a threshold is used.
    resampling_interpolation : string, optional (default continuous)
        The interpolation method for resampling.
        Can be 'continuous', 'linear', or 'nearest'.
        See nilearn.image.resample_img
    opacity : float in [0,1] (default 1)
        The level of opacity of the overlay (0: transparent, 1: opaque)

    Returns
    -------
    html_view : the html viewer object.
        It can be saved as an html page `html_view.save_as_html('test.html')`,
        or opened in a browser `html_view.open_in_browser()`.
        If the output is not requested and the current environment is a Jupyter
        notebook, the viewer will be inserted in the notebook.

    See Also
    --------
    nilearn.plotting.plot_stat_map:
        static plot of brain volume, on a single or multiple planes.
    nilearn.plotting.view_connectome:
        interactive 3d view of a connectome.
    nilearn.plotting.view_markers:
        interactive plot of colored markers.
    nilearn.plotting.view_surf, nilearn.plotting.view_img_on_surf:
        interactive view of statistical maps or surface atlases on the cortical
        surface.
    """

    # Prepare the color map and thresholding
    mask_img, stat_map_img, data, threshold = _mask_stat_map(
        stat_map_img, threshold)
    colors = colorscale(cmap, data.ravel(), threshold=threshold,
                        symmetric_cmap=symmetric_cmap, vmax=vmax,
                        vmin=vmin)

    # Prepare the data for the cuts
    bg_img, bg_min, bg_max, black_bg = _load_bg_img(stat_map_img, bg_img,
                                                    black_bg, dim)
    stat_map_img, mask_img = _resample_stat_map(stat_map_img, bg_img, mask_img,
                                                resampling_interpolation)
    cut_slices = _get_cut_slices(stat_map_img, cut_coords, threshold)

    # Now create a json-like object for the viewer, and converts in html
    json_view = _json_view_data(bg_img, stat_map_img, mask_img, bg_min, bg_max,
                                colors, cmap, colorbar)
    json_view['params'] = _json_view_params(
        stat_map_img.shape, stat_map_img.affine, colors['vmin'],
        colors['vmax'], cut_slices, black_bg, opacity, draw_cross, annotate,
        title, colorbar, value=False)
    html_view = _json_view_to_html(json_view)

    return html_view
