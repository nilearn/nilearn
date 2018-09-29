"""
Visualizing 3D stat maps in a Brainsprite viewer
"""
import warnings
import os
import numpy as np
import json
import numpy as np
import numbers
import matplotlib.pyplot as plt

from matplotlib.image import imsave
from nilearn.image import resample_img, resample_to_img , new_img_like
from io import BytesIO , StringIO
from .js_plotting_utils import get_html_template, HTMLDocument
from matplotlib import cm as mpl_cm
from matplotlib import colors
from base64 import encodebytes
from nibabel.affines import apply_affine
from ..plotting import cm
from ..plotting.find_cuts import find_xyz_cut_coords
from ..plotting.img_plotting import _load_anat, _get_colorbar_and_data_ranges
from .._utils import check_niimg_3d
from .._utils.extmath import fast_abs_percentile
from .._utils.param_validation import check_threshold
from .._utils.niimg import _safe_get_data
from ..datasets import load_mni152_template

def _resample_to_self(img,interpolation):
    u,s,vh = np.linalg.svd(img.affine[0:3,0:3])
    vsize = np.min(np.abs(s))
    img = resample_img(img,target_affine=np.diag([vsize,vsize,vsize]),interpolation=interpolation)
    return img

def _data2sprite(data):
    nx, ny, nz = data.shape
    nrows = int(np.ceil(np.sqrt(nx)))
    ncolumns = int(np.ceil(nx / float(nrows)))

    sprite = np.zeros((nrows * nz, ncolumns * ny))
    indrow, indcol = np.where(np.ones((nrows, ncolumns)))

    for xx in range(nx):
        # we need to flip the image in the x axis
        sprite[(indrow[xx] * nz):((indrow[xx] + 1) * nz),
        (indcol[xx] * ny):((indcol[xx] + 1) * ny)] = data[xx, :,::-1].transpose()

    return sprite

def save_sprite(img , output_sprite , output_cmap=None , output_json=None,
                vmax=None , vmin=None , cmap='Greys' , threshold=None ,
                n_colors=256 , format = 'png', resample=True ,
                interpolation = 'nearest') :
    """ Generate a sprite from a 3D Niimg-like object.

        Parameters
        ----------
        img :  Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
        output_file : string or file-like
            Path string to a filename, or a Python file-like object.
            If *format* is *None* and *fname* is a string, the output
            format is deduced from the extension of the filename.
        output_cmap : string, file-like or None, optional (default None)
            Path string to a filename, or a Python file-like object.
            The color map will be saved in that file (unless it is None).
            If *format* is *None* and *fname* is a string, the output format is
            deduced from the extension of the filename.
        output_json : string, file-like or None, optional (default None)
            Path string to a filename, or a Python file-like object.
            The parameters of the sprite will be saved in that file
            (unless it is None): Y and Z sizes, vmin, vmax, affine transform.
        vmax : float, or None, optional (default None)
            max value for mapping colors.
        vmin : float, or None, optional (default None)
            min value for mapping color.
        cmap : name of a matplotlib colormap, optional (default 'Greys')
            The colormap for the sprite. A matplotlib colormap can also
            be passed directly in cmap.
        threshold : a number, None, or 'auto', optional (default None)
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image:
            values below the threshold (in absolute value) are plotted
            as transparent. If auto is given, the threshold is determined
            magically by analysis of the image.
        n_colors : integer, optional (default 256)
            The number of discrete colors to use in the colormap, if it is
            generated.
        format : string, optional (default 'png')
            One of the file extensions supported by the active backend.  Most
            backends support png, pdf, ps, eps and svg.
        resample : boolean, optional (default True)
            Resample to isotropic voxels, with a LR/AP/VD orientation.
            This is necessary for proper rendering of arbitrary Niimg volumes,
            but not necessary if the image is in an isotropic standard space.
        interpolation : string, optional (default nearest)
            The interpolation method for resampling
            See nilearn.image.resample_img
        black_bg : boolean, optional
            If True, the background of the image is set to be black.

        Returns
        ----------
        sprite : numpy array with the sprite
    """

    # Get cmap
    if isinstance(cm,str):
        cmap = plt.cm.get_cmap(cmap)

    img = check_niimg_3d(img, dtype='auto')

    # resample to isotropic voxel with standard orientation
    if resample:
        img = _resample_to_self(img,interpolation)

    # Read data
    data = _safe_get_data(img, ensure_finite=True)
    if np.isnan(np.sum(data)):
        data = np.nan_to_num(data)

    # Deal with automatic settings of plot parameters
    if threshold == 'auto':
        # Threshold epsilon below a percentile value, to be sure that some
        # voxels pass the threshold
        threshold = fast_abs_percentile(data) - 1e-5

    # threshold
    threshold = float(threshold) if threshold is not None else None

    # Get vmin vmax
    show_nan_msg = False
    if vmax is not None and np.isnan(vmax):
        vmax = None
        show_nan_msg = True
    if vmin is not None and np.isnan(vmin):
        vmin = None
        show_nan_msg = True
    if show_nan_msg:
        nan_msg = ('NaN is not permitted for the vmax and vmin arguments.\n'
                   'Tip: Use np.nanmax() instead of np.max().')
        warnings.warn(nan_msg)

    if vmax is None:
        vmax = np.nanmax(data)
    if vmin is None:
        vmin = np.nanmin(data)

    # Create sprite
    sprite = _data2sprite(data)

    # Mask sprite
    if threshold is not None:
        if threshold == 0:
            sprite = np.ma.masked_equal(sprite, 0, copy=False)
        else:
            sprite = np.ma.masked_inside(sprite, -threshold, threshold,
                                           copy=False)
    # Save the sprite
    imsave(output_sprite,sprite,vmin=vmin,vmax=vmax,cmap=cmap,format=format)

    # Save the parameters
    if type(vmin).__module__ == 'numpy':
        vmin = vmin.tolist() # json does not deal with numpy array
    if type(vmax).__module__ == 'numpy':
        vmax = vmax.tolist() # json does not deal with numpy array

    if output_json is not None:
        params = {
                    'nbSlice': {
                        'X': data.shape[0],
                        'Y': data.shape[1],
                        'Z': data.shape[2]
                    },
                    'min': vmin,
                    'max': vmax,
                    'affine': img.affine.tolist()
                 }
        if isinstance(output_json,str):
            f = open(output_json,'w')
            f.write(json.dumps(params))
            f.close
        else:
            output_json.write(json.dumps(params))

    # save the colormap
    if output_cmap is not None:
        data = np.arange(0,n_colors)/(n_colors-1)
        data = data.reshape([1,n_colors])
        imsave(output_cmap,data,cmap=cmap,format=format)

    return sprite

def _custom_cmap(cmap,vmin,vmax,threshold=None):
    # For internal use only

    # This is taken from https://github.com/nilearn/nilearn/blob/master/nilearn/plotting/displays.py#L754-L772
    # Ideally this code would be reused in plot_stat_map, not just view_stat_map

    our_cmap = mpl_cm.get_cmap(cmap)
    # edge case where the data has a single value
    # yields a cryptic matplotlib error message
    # when trying to plot the color bar
    norm = colors.Normalize(vmin=vmin,vmax=vmax)
    nb_ticks = 5 if norm.vmin != norm.vmax else 1
    ticks = np.linspace(norm.vmin, norm.vmax, nb_ticks)
    bounds = np.linspace(norm.vmin, norm.vmax, our_cmap.N)

    # some colormap hacking
    if threshold is None:
        offset = 0
    else:
        offset = threshold
    if offset > norm.vmax:
        offset = norm.vmax
    cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
    istart = int(norm(-offset, clip=True) * (our_cmap.N - 1))
    istop = int(norm(offset, clip=True) * (our_cmap.N - 1))
    for i in range(istart, istop):
        cmaplist[i] = (0.5, 0.5, 0.5, 1.)  # just an average gray color
    if norm.vmin == norm.vmax:  # len(np.unique(data)) == 1 ?
        return
    else:
        our_cmap = colors.LinearSegmentedColormap.from_list(
            'Custom cmap', cmaplist, our_cmap.N)
        return our_cmap

class StatMapView(HTMLDocument):
    pass

def view_stat_map(stat_map_img, bg_img='MNI152', cut_coords=None,
                colorbar=True, title=None, threshold=None, annotate=True,
                draw_cross=True, black_bg='auto', cmap=cm.cold_hot,
                symmetric_cbar='auto', dim='auto',vmax=None,
                resampling_interpolation='continuous', n_colors=256, opacity=1,
                **kwargs):
    """
    Intarctive viewer of a statistical map, with optional background

    Parameters
    ----------
    stat_map_img : Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        The statistical map image.
    bg_img : Niimg-like object (default='MNI152')
        See http://nilearn.github.io/manipulating_images/input_output.html
        The background image that the stat map will be plotted on top of.
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".
    cut_coords : None, a tuple of floats, or an integer (default None)
        The MNI coordinates of the point where the cut is performed
        as a 3-tuple: (x, y, z). If None is given, the cuts is calculated
        automaticaly.
        This parameter is not currently supported.
    colorbar : boolean, optional (default True)
        If True, display a colorbar next to the plots.
    title : string or None (default=None)
        The title displayed on the figure (or None: no title).
        This parameter is not currently supported.
    threshold : str, number or None  (default=None)
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        magically by analysis of the image.
    annotate : boolean (default=True)
        If annotate is True, positions and left/right annotation
        are added to the plot.
    draw_cross : boolean (default=True)
        If draw_cross is True, a cross is drawn on the plot to
        indicate the cut plosition.
    black_bg : boolean (default='auto')
        If True, the background of the image is set to be black.
        Otherwise, a white background is used.
        If set to auto, an educated guess is made to find if the background
        is white or black.
    cmap : matplotlib colormap, optional
        The colormap for specified image. The colormap *must* be
        symmetrical.
    symmetric_cbar : boolean or 'auto' (default='auto')
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter if
        the range of the whole image is either positive or negative.
        Note: The colormap will always be set to range from -vmax to vmax.
    dim : float, 'auto' (default='auto')
        Dimming factor applied to background image. By default, automatic
        heuristics are applied based upon the background image intensity.
        Accepted float values, where a typical scan is between -2 and 2
        (-2 = increase constrast; 2 = decrease contrast), but larger values
        can be used for a more pronounced effect. 0 means no dimming.
    vmax : float, or None (default=)
        max value for mapping colors.
    resampling_interpolation : string, optional (default nearest)
        The interpolation method for resampling
        See nilearn.image.resample_img
    n_colors : integer (default=256)
        The number of discrete colors to use in the colormap, if it is
        generated.
    opacity : float in [0,1] (default 1)
        The level of opacity of the overlay (0: transparent, 1: opaque)

    Returns
    -------
    StatMapView : plot of the stat map.
        It can be saved as an html page or rendered (transparently) by the
        Jupyter notebook.
    """

    # Load stat map
    stat_map_img = check_niimg_3d(stat_map_img, dtype='auto')

    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        _safe_get_data(stat_map_img, ensure_finite=True),
        vmax,
        symmetric_cbar,
        kwargs)

    # load background image, and resample stat map
    if bg_img is not None and bg_img is not False :
        if isinstance(bg_img, str) and bg_img == "MNI152":
            bg_img = load_mni152_template()
        bg_img, black_bg, bg_min, bg_max = _load_anat(bg_img,dim=dim,black_bg = black_bg)
        bg_img = _resample_to_self(bg_img,interpolation='nearest')
        stat_map_img = resample_to_img(stat_map_img, bg_img , interpolation=resampling_interpolation)

    else:
        stat_map_img = _resample_to_self(stat_map_img,interpolation=resampling_interpolation)
        bg_img = new_img_like(stat_map_img,np.zeros(stat_map_img.shape),stat_map_img.affine)
        bg_min = 0
        bg_max = 0
        if black_bg is 'auto':
            black_bg = False

    # Set color parameters
    if black_bg:
        cfont = '#FFFFFF'
        cbg = '#000000'
    else:
        cfont = '#000000'
        cbg = '#FFFFFF'

    # Select coordinates for the cut
    # https://github.com/nilearn/nilearn/blob/master/nilearn/plotting/displays.py#L943
    if isinstance(cut_coords, numbers.Number):
        raise ValueError(
            "The input given for display_mode='ortho' needs to be "
            "a list of 3d world coordinates in (x, y, z). "
            "You provided single cut, cut_coords={0}".format(cut_coords))
    if cut_coords is None:
        cut_coords = find_xyz_cut_coords(
                    stat_map_img, activation_threshold=threshold)

    # Create a base64 sprite for the background
    bg_sprite = BytesIO()
    save_sprite(bg_img,output_sprite=bg_sprite,cmap='gray',format='png',resample=False,vmin=bg_min, vmax=bg_max)
    bg_sprite.seek(0)
    bg_base64 = encodebytes(bg_sprite.read()).decode('utf-8')
    bg_sprite.close()

    # Create a base64 sprite for the stat map
    # Possibly, also generate a file with the colormap
    stat_map_sprite = BytesIO()
    stat_map_json = StringIO()
    if colorbar:
        stat_map_cm = BytesIO()
    else:
        stat_map_cm = None
    cmap_c = _custom_cmap(cmap,vmin,vmax,threshold)
    save_sprite(stat_map_img,stat_map_sprite,stat_map_cm,stat_map_json,vmax,vmin,cmap_c,
                    threshold,n_colors,'png',False)

    # Convert the sprite and colormap to base64
    stat_map_sprite.seek(0)
    stat_map_base64 = encodebytes(stat_map_sprite.read()).decode('utf-8')
    stat_map_sprite.close()

    if colorbar:
        stat_map_cm.seek(0)
        cm_base64 = encodebytes(stat_map_cm.read()).decode('utf-8')
        stat_map_cm.close()
    else:
        cm_base64 = ''

    # Load the sprite meta-data from the json dump
    stat_map_json.seek(0)
    params = json.load(stat_map_json)
    stat_map_json.close()

    # Convet cut coordinates into cut slices
    cut_slices = np.round(apply_affine(
            np.linalg.inv(stat_map_img.affine),cut_coords))

    # Create a json-like structure
    # with all the brain sprite parameters
    sprite_params = {
                        'canvas': '3Dviewer',
                        'sprite': 'spriteImg',
                        'nbSlice': params['nbSlice'],
                        'overlay': {
                                'sprite': 'overlayImg',
                                'nbSlice': params['nbSlice'],
                                'opacity': opacity
                                },
                        'colorBackground': cbg,
                        'colorFont': cfont,
                        'crosshair': draw_cross,
                        'affine': params['affine'],
                        'flagCoordinates': annotate,
                        'title': title,
                        'flagValue': annotate,
                        'numSlice': {
                            'X': cut_slices[0],
                            'Y': cut_slices[1],
                            'Z': cut_slices[2]
                        },
                    }
    if colorbar:
        sprite_params['colorMap'] = {
                    'img' : 'colorMap',
                    'min' : params['min'],
                    'max' : params['max']
                }

    # Load javascript libraries
    js_dir = os.path.join(os.path.dirname(__file__), 'data', 'js')
    with open(os.path.join(js_dir, 'jquery.min.js')) as f:
      js_jquery = f.read()
    with open(os.path.join(js_dir, 'brainsprite.min.js')) as f:
      js_brainsprite = f.read()

    # Load the html template, and plug base64 data and meta-data
    html = get_html_template('stat_map_template.html')
    html = html.replace('INSERT_SPRITE_PARAMS_HERE', json.dumps(sprite_params))
    html = html.replace('INSERT_BG_DATA_HERE', bg_base64)
    html = html.replace('INSERT_STAT_MAP_DATA_HERE',stat_map_base64)
    html = html.replace('INSERT_CM_DATA_HERE',cm_base64)
    html = html.replace('INSERT_JQUERY_HERE',js_jquery)
    html = html.replace('INSERT_BRAINSPRITE_HERE',js_brainsprite)
    return StatMapView(html)
