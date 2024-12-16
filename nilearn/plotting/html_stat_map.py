"""Visualizing 3D stat maps in a Brainsprite viewer."""

import copy
import json
import warnings
from base64 import b64encode
from io import BytesIO
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.image import imsave
from nibabel.affines import apply_affine

from nilearn.plotting.html_document import HTMLDocument

from .._utils import fill_doc
from .._utils.extmath import fast_abs_percentile
from .._utils.niimg import safe_get_data
from .._utils.niimg_conversions import check_niimg_3d
from .._utils.param_validation import check_threshold
from ..datasets import load_mni152_template
from ..image import get_data, new_img_like, reorder_img, resample_to_img
from ..plotting import cm
from ..plotting.find_cuts import find_xyz_cut_coords
from ..plotting.img_plotting import load_anat
from .js_plotting_utils import colorscale, get_html_template


def _data_to_sprite(data):
    """Convert a 3D array into a sprite of sagittal slices.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        Input data to convert to sprite.

    Returns
    -------
    sprite : 2D :class:`numpy.ndarray`
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
        sprite[
            (indrow[xx] * nz) : ((indrow[xx] + 1) * nz),
            (indcol[xx] * ny) : ((indcol[xx] + 1) * ny),
        ] = data[xx, :, ::-1].transpose()

    return sprite


def _threshold_data(data, threshold=None):
    """Threshold a data array.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        Data to apply threshold on.

    threshold : :obj:`float`, optional
        Threshold to apply to data.

    Returns
    -------
    data : :class:`numpy.ndarray`
        Thresholded data.

    mask : :class:`numpy.ndarray` of :obj:`bool`
        Boolean mask.

    threshold : :obj:`float`
        Updated threshold value.

    """
    # If threshold is None, do nothing
    if threshold is None:
        mask = np.full(data.shape, False)
        return data, mask, threshold

    # Deal with automatic settings of plot parameters
    if threshold == "auto":
        # Threshold epsilon below a percentile value, to be sure that some
        # voxels pass the threshold
        threshold = fast_abs_percentile(data) - 1e-5

    # Threshold
    threshold = check_threshold(
        threshold, data, percentile_func=fast_abs_percentile, name="threshold"
    )

    if threshold == 0:
        mask = data == 0
    else:
        mask = (data >= -threshold) & (data <= threshold)
    data = data * np.logical_not(mask)
    if not np.any(mask):
        warnings.warn(
            f"Threshold given was {threshold}, "
            f"but the data has no values below {data.min()}. "
        )
    return data, mask, threshold


def _save_sprite(
    data, output_sprite, vmax, vmin, mask=None, cmap="Greys", format="png"
):
    """Generate a sprite from a 3D Niimg-like object.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        Input data.

    output_sprite : :class:`numpy.ndarray`
        Output sprite.

    vmax, vmin : :obj:`float`
        ???

    mask : :class:`numpy.ndarray`, optional
        Mask to use.

    cmap : :obj:`str` or colormap, default='Greys'
        Colormap to use.

    format : :obj:`str`, default='png'
        Format to use for output image.

    Returns
    -------
    sprite : :class:`numpy.ndarray`
        Returned sprite.

    """
    # Create sprite
    sprite = _data_to_sprite(data)

    # Mask the sprite
    if mask is not None:
        mask = _data_to_sprite(mask)
        sprite = np.ma.array(sprite, mask=mask)

    # Save the sprite
    imsave(
        output_sprite, sprite, vmin=vmin, vmax=vmax, cmap=cmap, format=format
    )

    return sprite


def _bytes_io_to_base64(handle_io):
    """Encode the content of a bytesIO virtual file as base64.

    Also closes the file.

    Returns
    -------
    data
    """
    handle_io.seek(0)
    data = b64encode(handle_io.read()).decode("utf-8")
    handle_io.close()
    return data


def _save_cm(output_cmap, cmap, format="png", n_colors=256):
    """Save the colormap of an image as an image file."""
    # save the colormap
    data = np.arange(0.0, n_colors) / (n_colors - 1.0)
    data = data.reshape([1, n_colors])
    imsave(output_cmap, data, cmap=cmap, format=format)


class StatMapView(HTMLDocument):  # noqa: D101
    pass


def _mask_stat_map(stat_map_img, threshold=None):
    """Load a stat map and apply a threshold.

    Returns
    -------
    mask_img

    stat_map_img

    data

    threshold
    """
    # Load stat map
    stat_map_img = check_niimg_3d(stat_map_img, dtype="auto")
    data = safe_get_data(stat_map_img, ensure_finite=True)

    # threshold the stat_map
    if threshold is not None:
        data, mask, threshold = _threshold_data(data, threshold)
        mask_img = new_img_like(stat_map_img, mask, stat_map_img.affine)
    else:
        mask_img = new_img_like(
            stat_map_img, np.zeros(data.shape), stat_map_img.affine
        )
    return mask_img, stat_map_img, data, threshold


def _load_bg_img(stat_map_img, bg_img="MNI152", black_bg="auto", dim="auto"):
    """Load and resample bg_img in an isotropic resolution, \
    with a positive diagonal affine matrix.

    Returns
    -------
    bg_img

    bg_min

    bg_max

    black_bg
    """
    if bg_img is None or bg_img is False:
        if black_bg == "auto":
            black_bg = False
        bg_img = new_img_like(
            stat_map_img, np.ma.masked_all(stat_map_img.shape)
        )
        bg_min, bg_max = 0, 0
    else:
        if isinstance(bg_img, str) and bg_img == "MNI152":
            bg_img = load_mni152_template(resolution=2)
        else:
            bg_img = check_niimg_3d(bg_img)
        masked_data = np.ma.masked_inside(
            safe_get_data(bg_img, ensure_finite=True), -1e-6, 1e-6, copy=False
        )
        bg_img = new_img_like(bg_img, masked_data)
        bg_img, black_bg, bg_min, bg_max = load_anat(
            bg_img, dim=dim, black_bg=black_bg
        )
    bg_img = reorder_img(bg_img, resample="nearest", copy_header=True)
    return bg_img, bg_min, bg_max, black_bg


def _resample_stat_map(
    stat_map_img, bg_img, mask_img, resampling_interpolation="continuous"
):
    """Resample the stat map and mask to the background.

    Returns
    -------
    stat_map_img

    mask_img
    """
    stat_map_img = resample_to_img(
        stat_map_img,
        bg_img,
        interpolation=resampling_interpolation,
        copy_header=True,
        force_resample=False,  # TODO set to True in 0.13.0
    )
    mask_img = resample_to_img(
        mask_img,
        bg_img,
        fill_value=1,
        interpolation="nearest",
        copy_header=True,
        force_resample=False,  # TODO set to True in 0.13.0
    )

    return stat_map_img, mask_img


def _json_view_params(
    shape,
    affine,
    vmin,
    vmax,
    cut_slices,
    black_bg=False,
    opacity=1,
    draw_cross=True,
    annotate=True,
    title=None,
    colorbar=True,
    value=True,
):
    """Create a dictionary with all the brainsprite parameters.

    Returns
    -------
    params
    """
    # Set color parameters
    if black_bg:
        cfont = "#FFFFFF"
        cbg = "#000000"
    else:
        cfont = "#000000"
        cbg = "#FFFFFF"

    # Deal with limitations of json dump regarding types
    if type(vmin).__module__ == "numpy":
        vmin = vmin.tolist()  # json does not deal with numpy array
    if type(vmax).__module__ == "numpy":
        vmax = vmax.tolist()  # json does not deal with numpy array

    params = {
        "canvas": "3Dviewer",
        "sprite": "spriteImg",
        "nbSlice": {"X": shape[0], "Y": shape[1], "Z": shape[2]},
        "overlay": {
            "sprite": "overlayImg",
            "nbSlice": {"X": shape[0], "Y": shape[1], "Z": shape[2]},
            "opacity": opacity,
        },
        "colorBackground": cbg,
        "colorFont": cfont,
        "crosshair": draw_cross,
        "affine": affine.tolist(),
        "flagCoordinates": annotate,
        "title": title,
        "flagValue": value,
        "numSlice": {
            "X": cut_slices[0] - 1,
            "Y": cut_slices[1] - 1,
            "Z": cut_slices[2] - 1,
        },
    }

    if colorbar:
        params["colorMap"] = {"img": "colorMap", "min": vmin, "max": vmax}
    return params


def _json_view_size(params, width_view=600):
    """Define the size of the viewer.

    Returns
    -------
    width_view

    height_view
    """
    # slices_width = sagittal_width (y) + coronal_width (x) + axial_width (x)
    slices_width = params["nbSlice"]["Y"] + 2 * params["nbSlice"]["X"]

    # slices_height = max of sagittal_height (z), coronal_height (z), and
    # axial_height (y).
    # Also add 20% extra height for annotation and margin
    slices_height = np.max([params["nbSlice"]["Y"], params["nbSlice"]["Z"]])
    slices_height = 1.20 * slices_height

    # Get the final size of the viewer
    ratio = slices_height / slices_width
    height_view = np.ceil(ratio * width_view)

    return width_view, height_view


def _get_bg_mask_and_cmap(bg_img, black_bg):
    """Get background data for _json_view_data."""
    bg_mask = np.ma.getmaskarray(get_data(bg_img))
    bg_cmap = copy.copy(matplotlib.pyplot.get_cmap("gray"))
    if black_bg:
        bg_cmap.set_bad("black")
    else:
        bg_cmap.set_bad("white")
    return bg_mask, bg_cmap


def _json_view_data(
    bg_img,
    stat_map_img,
    mask_img,
    bg_min,
    bg_max,
    black_bg,
    colors,
    cmap,
    colorbar,
):
    """Create a json-like viewer object, and populate with base64 data.

    Returns
    -------
    json_view
    """
    # Initialize brainsprite data structure
    json_view = dict.fromkeys(
        [
            "bg_base64",
            "stat_map_base64",
            "cm_base64",
            "params",
            "js_jquery",
            "js_brainsprite",
        ]
    )

    # Create a base64 sprite for the background
    bg_sprite = BytesIO()
    bg_data = safe_get_data(bg_img, ensure_finite=True).astype(float)
    bg_mask, bg_cmap = _get_bg_mask_and_cmap(bg_img, black_bg)
    _save_sprite(bg_data, bg_sprite, bg_max, bg_min, bg_mask, bg_cmap, "png")
    json_view["bg_base64"] = _bytes_io_to_base64(bg_sprite)

    # Create a base64 sprite for the stat map
    stat_map_sprite = BytesIO()
    data = safe_get_data(stat_map_img, ensure_finite=True)
    mask = safe_get_data(mask_img, ensure_finite=True)
    _save_sprite(
        data,
        stat_map_sprite,
        colors["vmax"],
        colors["vmin"],
        mask,
        cmap,
        "png",
    )
    json_view["stat_map_base64"] = _bytes_io_to_base64(stat_map_sprite)

    # Create a base64 colormap
    if colorbar:
        stat_map_cm = BytesIO()
        _save_cm(stat_map_cm, colors["cmap"], "png")
        json_view["cm_base64"] = _bytes_io_to_base64(stat_map_cm)
    else:
        json_view["cm_base64"] = ""

    return json_view


def _json_view_to_html(json_view, width_view=600):
    """Fill a brainsprite html template with relevant parameters and data.

    Returns
    -------
    html_view
    """
    # Fix the size of the viewer
    width, height = _json_view_size(json_view["params"], width_view)

    # Populate all missing keys with html-ready data
    json_view["INSERT_PAGE_TITLE_HERE"] = (
        json_view["params"]["title"] or "Slice viewer"
    )
    json_view["params"] = json.dumps(json_view["params"])
    js_dir = Path(__file__).parent / "data" / "js"
    with (js_dir / "jquery.min.js").open() as f:
        json_view["js_jquery"] = f.read()
    with (js_dir / "brainsprite.min.js").open() as f:
        json_view["js_brainsprite"] = f.read()

    # Load the html template, and plug in all the data
    html_view = get_html_template("stat_map_template.html")
    html_view = html_view.safe_substitute(json_view)

    return StatMapView(html_view, width=width, height=height)


def _get_cut_slices(stat_map_img, cut_coords=None, threshold=None):
    """For internal use.

    Find slice numbers for the cut.
    Based on find_xyz_cut_coords
    """
    # Select coordinates for the cut
    if cut_coords is None:
        cut_coords = find_xyz_cut_coords(
            stat_map_img, activation_threshold=threshold
        )

    # Convert cut coordinates into cut slices
    try:
        cut_slices = apply_affine(
            np.linalg.inv(stat_map_img.affine), cut_coords
        )
    except ValueError:
        raise ValueError(
            "The input given for display_mode='ortho' "
            "needs to be a list of 3d world coordinates in (x, y, z). "
            f"You provided cut_coords={cut_coords}"
        )
    except IndexError:
        raise ValueError(
            "The input given for display_mode='ortho' "
            "needs to be a list of 3d world coordinates in (x, y, z). "
            f"You provided single cut, cut_coords={cut_coords}"
        )

    return cut_slices


@fill_doc
def view_img(
    stat_map_img,
    bg_img="MNI152",
    cut_coords=None,
    colorbar=True,
    title=None,
    threshold=1e-6,
    annotate=True,
    draw_cross=True,
    black_bg="auto",
    cmap=cm.cold_hot,
    symmetric_cmap=True,
    dim="auto",
    vmax=None,
    vmin=None,
    resampling_interpolation="continuous",
    width_view=600,
    opacity=1,
):
    """Interactive html viewer of a statistical map, with optional background.

    Parameters
    ----------
    stat_map_img : Niimg-like object
        See :ref:`extracting_data`.
        The statistical map image. Can be either a 3D volume or a 4D volume
        with exactly one time point.
    %(bg_img)s
        If nothing is specified, the MNI152 template will be used.
        To turn off background image, just pass "bg_img=False".
        Default='MNI152'.

    cut_coords : None, or a :obj:`tuple` of :obj:`float`, default=None
        The :term:`MNI` coordinates of the point where the cut is performed
        as a 3-tuple: (x, y, z). If None is given, the cuts are calculated
        automatically.

    colorbar : :obj:`bool`, default=True
        If True, display a colorbar on top of the plots.
    %(title)s
    threshold : :obj:`str`, number or None, default=1e-06
        If None is given, the image is not thresholded.
        If a string of the form "90%%" is given, use the 90-th percentile of
        the absolute value in the image.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        automatically.

    annotate : :obj:`bool`, default=True
        If annotate is True, current cuts are added to the viewer.
    %(draw_cross)s
    black_bg : :obj:`bool` or 'auto', default='auto'
        If True, the background of the image is set to be black.
        Otherwise, a white background is used.
        If set to auto, an educated guess is made to find if the background
        is white or black.
    %(cmap)s
        Default=`plt.cm.cold_hot`.
    symmetric_cmap : :obj:`bool`, default=True
        True: make colormap symmetric (ranging from -vmax to vmax).
        False: the colormap will go from the minimum of the volume to vmax.
        Set it to False if you are plotting a positive volume, e.g. an atlas
        or an anatomical image.
    %(dim)s
        Default='auto'.
    vmax : :obj:`float`, or None, default=None
        max value for mapping colors.
        If vmax is None and symmetric_cmap is True, vmax is the max
        absolute value of the volume.
        If vmax is None and symmetric_cmap is False, vmax is the max
        value of the volume.

    vmin : :obj:`float`, or None, default=None
        min value for mapping colors.
        If `symmetric_cmap` is `True`, `vmin` is always equal to `-vmax` and
        cannot be chosen.
        If `symmetric_cmap` is `False`, `vmin` is equal to the min of the
        image, or 0 when a threshold is used.
    %(resampling_interpolation)s
        Default='continuous'.

    width_view : :obj:`int`, default=600
        Default=600.
        Width of the viewer in pixels.

    opacity : :obj:`float` in [0,1], default=1
        The level of opacity of the overlay (0: transparent, 1: opaque).

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
        stat_map_img, threshold
    )
    colors = colorscale(
        cmap,
        data.ravel(),
        threshold=threshold,
        symmetric_cmap=symmetric_cmap,
        vmax=vmax,
        vmin=vmin,
    )

    # Prepare the data for the cuts
    bg_img, bg_min, bg_max, black_bg = _load_bg_img(
        stat_map_img, bg_img, black_bg, dim
    )
    stat_map_img, mask_img = _resample_stat_map(
        stat_map_img, bg_img, mask_img, resampling_interpolation
    )
    cut_slices = _get_cut_slices(stat_map_img, cut_coords, threshold)

    # Now create a json-like object for the viewer, and converts in html
    json_view = _json_view_data(
        bg_img,
        stat_map_img,
        mask_img,
        bg_min,
        bg_max,
        black_bg,
        colors,
        cmap,
        colorbar,
    )

    json_view["params"] = _json_view_params(
        stat_map_img.shape,
        stat_map_img.affine,
        colors["vmin"],
        colors["vmax"],
        cut_slices,
        black_bg,
        opacity,
        draw_cross,
        annotate,
        title,
        colorbar,
        value=False,
    )

    html_view = _json_view_to_html(json_view, width_view)

    return html_view
