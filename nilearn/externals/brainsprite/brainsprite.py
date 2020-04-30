"""Python interface for the brainsprite.js library.
"""
import os
import json
import warnings
from pathlib import Path
from io import BytesIO
from base64 import b64encode

import numpy as np
from matplotlib.image import imsave

from nibabel.affines import apply_affine

from nilearn.image import resample_to_img, new_img_like, reorder_img
from nilearn.plotting.js_plotting_utils import get_html_template, colorscale
from nilearn.plotting import cm
from nilearn.plotting.find_cuts import find_xyz_cut_coords
from nilearn.plotting.img_plotting import _load_anat
from nilearn.reporting import HTMLDocument
from nilearn._utils.niimg_conversions import check_niimg_3d
from nilearn._utils.param_validation import check_threshold
from nilearn._utils.extmath import fast_abs_percentile
from nilearn._utils.niimg import _safe_get_data
from nilearn.datasets import load_mni152_template
from nilearn.externals import tempita


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
        sprite[
            (indrow[xx] * nz) : ((indrow[xx] + 1) * nz),
            (indcol[xx] * ny) : ((indcol[xx] + 1) * ny),
        ] = data[xx, :, ::-1].transpose()

    return sprite


def _threshold_data(data, threshold=None):
    """ Threshold a data array.
        Returns: data (array), mask (boolean array) threshold (updated)
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

    # Mask data
    if threshold == 0:
        mask = data == 0
        data = data * np.logical_not(mask)
    else:
        mask = (data >= -threshold) & (data <= threshold)
        data = data * np.logical_not(mask)

    if not np.any(mask):
        warnings.warn(
            "Threshold given was {0}, but "
            "the data has no values below {1}. ".format(threshold, data.min())
        )
    return data, mask, threshold


def _bytesIO_to_base64(handle_io):
    """ Encode the content of a bytesIO virtual file as base64.
        Also closes the file.
        Returns: data
    """
    handle_io.seek(0)
    data = b64encode(handle_io.read()).decode("utf-8")
    handle_io.close()
    return data


def _mask_stat_map(stat_map_img, threshold=None):
    """ Load a stat map and apply a threshold.
        Returns: mask_img, stat_map_img, data, threshold
    """
    # Load stat map
    stat_map_img = check_niimg_3d(stat_map_img, dtype="auto")
    data = _safe_get_data(stat_map_img, ensure_finite=True)

    # threshold the stat_map
    if threshold is not None:
        data, mask, threshold = _threshold_data(data, threshold)
        mask_img = new_img_like(stat_map_img, mask, stat_map_img.affine)
    else:
        mask_img = new_img_like(stat_map_img, np.zeros(data.shape), stat_map_img.affine)
    return mask_img, stat_map_img, data, threshold


def _load_bg_img(stat_map_img, bg_img="MNI152", black_bg="auto", dim="auto"):
    """ Load and resample bg_img in an isotropic resolution,
        with a positive diagonal affine matrix.
        Returns: bg_img, bg_min, bg_max, black_bg
    """
    if (bg_img is None or bg_img is False) and black_bg == "auto":
        black_bg = False

    if bg_img is not None and bg_img is not False:
        if isinstance(bg_img, str) and bg_img == "MNI152":
            bg_img = load_mni152_template()
        bg_img, black_bg, bg_min, bg_max = _load_anat(
            bg_img, dim=dim, black_bg=black_bg
        )
    else:
        bg_img = new_img_like(
            stat_map_img, np.zeros(stat_map_img.shape), stat_map_img.affine
        )
        bg_min = 0
        bg_max = 0
    bg_img = reorder_img(bg_img, resample="nearest")
    return bg_img, bg_min, bg_max, black_bg


def _resample_stat_map(
    stat_map_img, bg_img, mask_img, resampling_interpolation="continuous"
):
    """ Resample the stat map and mask to the background.
        Returns: stat_map_img, mask_img
    """
    stat_map_img = resample_to_img(
        stat_map_img, bg_img, interpolation=resampling_interpolation
    )
    mask_img = resample_to_img(mask_img, bg_img, fill_value=1, interpolation="nearest")

    return stat_map_img, mask_img


def _viewer_size(shape):
    """ Define the size of the viewer.
        Returns: width_view, height_view
    """
    # slices_width = sagittal_width (y) + coronal_width (x) + axial_width (x)
    slices_width = shape[1] + 2 * shape[0]

    # slices_height = max of sagittal_height (z), coronal_height (z), and
    # axial_height (y).
    # Also add 20% extra height for annotation and margin
    slices_height = np.max([shape[1], shape[2]])
    slices_height = 1.20 * slices_height

    # Get the final size of the viewer
    width_view = 600
    ratio = slices_height / slices_width
    height_view = np.ceil(ratio * width_view)

    return width_view, height_view


def _get_cut_slices(stat_map_img, cut_coords=None, threshold=None):
    """For internal use. Find slice numbers for the cut.
    """
    # Select coordinates for the cut
    if cut_coords is None:
        cut_coords = find_xyz_cut_coords(stat_map_img, activation_threshold=threshold)

    # Convert cut coordinates into cut slices
    try:
        cut_slices = apply_affine(np.linalg.inv(stat_map_img.affine), cut_coords)
    except ValueError:
        raise ValueError(
            "The input given for display_mode='ortho' needs to be "
            "a list of 3d world coordinates in (x, y, z). "
            "You provided cut_coords={0}".format(cut_coords)
        )
    except IndexError:
        raise ValueError(
            "The input given for display_mode='ortho' needs to be "
            "a list of 3d world coordinates in (x, y, z). "
            "You provided single cut, cut_coords={0}".format(cut_coords)
        )

    return cut_slices


def _save_sprite(
    img, vmax, vmin, output_sprite=None, mask=None, cmap="Greys", format="png"
):
    """ Generate a sprite from a 3D Niimg-like object.
        Returns: sprite
    """

    # Create sprite
    sprite = _data_to_sprite(_safe_get_data(img, ensure_finite=True))

    # Mask the sprite
    if mask is not None:
        mask = _data_to_sprite(_safe_get_data(mask, ensure_finite=True))
        sprite = np.ma.array(sprite, mask=mask)

    # Save the sprite
    if output_sprite is None:
        output_sprite = BytesIO()
        imsave(output_sprite, sprite, vmin=vmin, vmax=vmax, cmap=cmap, format=format)
        output_sprite = _bytesIO_to_base64(output_sprite)
    else:
        imsave(output_cmap, data, cmap=cmap, format=format)

    return output_sprite


def _save_cm(cmap, output_cmap=None, format="png", n_colors=256):
    """ Save the colormap of an image as an image file.
    """

    # the colormap
    data = np.arange(0.0, n_colors) / (n_colors - 1.0)
    data = data.reshape([1, n_colors])

    if output_cmap is None:
        output_cmap = BytesIO()
        imsave(output_cmap, data, cmap=cmap, format=format)
        output_cmap = _bytesIO_to_base64(output_cmap)
    else:
        imsave(output_cmap, data, cmap=cmap, format=format)
    return output_cmap


class StatMapView(HTMLDocument):
    pass


class viewer_substitute:
    """
    Templating tool to insert a brainsprite viewer in an HTML document

    :param canvas: The label for the brainsprite html canvas.
    :type canvas: str, optional
    :param sprite: The label for the html sprite background image.
    :type sprite: str, optional
    :param sprite_overlay: The label for the html sprite overlay image.
    :type sprite_overlay: str, optional
    :param img_colorMap: The label for the html colormap image.
    :type img_colorMap: str, optional
    :param cut_coords: The MNI coordinates of the point where the cut is performed
        as a 3-tuple: (x, y, z). If None is given, the cuts are calculated
        automaticaly.
    :type cut_coords: None, or a tuple of floats, optional
    :param colorbar: If True, display a colorbar on top of the plots.
    :type colorbar: boolean, optional
    :param title: The title displayed on the figure (or None: no title).
    :type title: string or None, optional
    :param threshold: If None is given, the image is not thresholded.
        If a string of the form "90%" is given, use the 90-th percentile of
        the absolute value in the image.
        If a number is given, it is used to threshold the image:
        values below the threshold (in absolute value) are plotted
        as transparent. If auto is given, the threshold is determined
        automatically.
    :type threshold: string, number or None, optional
    :param annotate: If annotate is True, current cuts are added to the viewer.
    :type annotate: boolean, optional
    :param draw_cross: If draw_cross is True, a cross is drawn on the plot to
        indicate the cuts.
    :type draw_cross: boolean, optional
    :param black_bg: If True, the background of the image is set to be black.
        Otherwise, a white background is used.
        If set to auto, an educated guess is made to find if the background
        is white or black.
    :type black_bg: boolean, optional
    :param cmap: The colormap for specified image.
    :type cmap:  matplotlib colormap, optional
    :param symmetric_cmap: True: make colormap symmetric (ranging from -vmax to vmax).
        False: the colormap will go from the minimum of the volume to vmax.
        Set it to False if you are plotting a positive volume, e.g. an atlas
        or an anatomical image.
    :type symmetric_cmap: bool, optional
    :param dim: Dimming factor applied to background image. By default, automatic
        heuristics are applied based upon the background image intensity.
        Accepted float values, where a typical scan is between -2 and 2
        (-2 = increase constrast; 2 = decrease contrast), but larger values
        can be used for a more pronounced effect. 0 means no dimming.
    :type dim: float or 'auto', optional
    :param vmax: max value for mapping colors.
        If vmax is None and symmetric_cmap is True, vmax is the max
        absolute value of the volume.
        If vmax is None and symmetric_cmap is False, vmax is the max
        value of the volume.
    :type vmax: float, or None, optional
    :param vmin: min value for mapping colors.
        If `symmetric_cmap` is `True`, `vmin` is always equal to `-vmax` and
        cannot be chosen.
        If `symmetric_cmap` is `False`, `vmin` defaults to the min of the
        image, or 0 when a threshold is used.
    :type vmin: float, or None, optional
    :param resampling_interpolation: The interpolation method for resampling.
        Can be 'continuous', 'linear', or 'nearest'.
        See nilearn.image.resample_img
    :type resampling_interpolation: string, optional
    :param opacity: The level of opacity of the overlay (0: transparent, 1: opaque)
    :type opacity: float in [0,1], optional
    :param value: dislay the value of the overlay at the current voxel.
    :type value: boolean, optional
    :param base64: turn on/off embedding of sprites in the html using base64 encoding.
        If the flag is off, the sprites (and the colorbar) will be saved in
        files named based on parameters sprite, sprite_overlay and img_colorMap.
    :type base64: boolean (default True)

    :return bsprite: a brainsprite viewer template substitution tool.

    """

    def __init__(
        self,
        canvas="3Dviewer",
        sprite="spriteImg",
        sprite_overlay="overlayImg",
        img_colorMap="colorMap",
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
        opacity=1,
        value=True,
        base64=True,
    ):
        """Set up default attributes for the class."""
        self.canvas = canvas
        self.sprite = sprite
        self.sprite_overlay = sprite_overlay
        self.img_colorMap = img_colorMap
        self.cut_coords = cut_coords
        self.colorbar = colorbar
        self.title = title
        self.threshold = threshold
        self.annotate = annotate
        self.draw_cross = draw_cross
        self.black_bg = black_bg
        self.cmap = cmap
        self.symmetric_cmap = symmetric_cmap
        self.dim = dim
        self.vmax = vmax
        self.vmin = vmin
        self.resampling_interpolation = resampling_interpolation
        self.opacity = opacity
        self.value = value
        self.base64 = base64

    def fit(self, stat_map_img, bg_img="MNI152"):
        """
        Generate sprite and meta-data from a brain volume. Also optionally
        incorporate a background image.

        :param stat_map_img: The statistical map image. Can be either a 3D volume
            or a 4D volume with exactly one time point.
        :type stat_map_img: stasNiimg-like object, See
            http://nilearn.github.io/manipulating_images/input_output.html
        :param bg_img: The background image that the stat map will be plotted on top of.
            If nothing is specified, the MNI152 template will be used.
            To turn off background image, just pass "bg_img=False".
        :type bg_img: Niimg-like object, optional
            See http://nilearn.github.io/manipulating_images/input_output.html
        """
        # Prepare the color map and thresholding
        mask_img, stat_map_img, data, self.threshold = _mask_stat_map(
            stat_map_img, self.threshold
        )

        self.colors_ = colorscale(
            self.cmap,
            data.ravel(),
            threshold=self.threshold,
            symmetric_cmap=self.symmetric_cmap,
            vmax=self.vmax,
            vmin=self.vmin,
        )

        if self.black_bg:
            cfont = "#FFFFFF"
            cbg = "#000000"
        else:
            cfont = "#000000"
            cbg = "#FFFFFF"

        # Prepare the data for the cuts
        bg_img, self.bg_min_, self.bg_max_, self.black_bg_ = _load_bg_img(
            stat_map_img, bg_img, self.black_bg, self.dim
        )
        stat_map_img, mask_img = _resample_stat_map(
            stat_map_img, bg_img, mask_img, self.resampling_interpolation
        )
        self.cut_slices_ = _get_cut_slices(
            stat_map_img, self.cut_coords, self.threshold
        )

        # Now create the viewer, and populate the sprite data
        self.html_ = self._brainsprite_html(bg_img, stat_map_img, mask_img)

        # Add the javascript snippet
        self.javascript_ = self._brainsprite_js(
            shape=stat_map_img.shape,
            affine=stat_map_img.affine,
            colorFont=cfont,
            colorBackground=cbg,
        )

        # Add the brainsprite.min.js library
        js_dir = os.path.join(os.path.dirname(__file__), "data", "js")
        with open(os.path.join(js_dir, "brainsprite.min.js")) as f:
            self.library_ = f.read()
            f.close()

        # Suggest a size for the viewer
        # width x height, in pixels
        self.width_, self.height_ = _viewer_size(stat_map_img.shape)

    def transform(self, template, javascript, html, library, namespace=None,
        width=None, height=None):
        """ Apply substitution in a template, using tempita.

            :param template: a template where brainsprite data needs to be substitued.
            :type template: tempita template
            :param javascript: the tempita name to substitute with brainsprite javascript snippet.
                If None, javascript is not substitued.
            :type javascript: str or None
            :param html: the tempita name to substitute with brainsprite html snippet.
                If None, html is not substitued.
            :type html: str or None
            :param library: the tempita name to substitue with the brainsprite js library.
                If None, library is not substitued.
            :type library: str or None
            :param namespace: a list of names to substitute, using tempita's substitute method.
            :type namespace: dict
            :param width: the width of the html report.
                If None, the width of the report will be the width of the viewer.
            :type height: int or None
            :param height: the height of the html report.
                If None, the height of the report will be the height of the viewer.
            :type height: int or None
        """
        if namespace == None:
            namespace = {}

        if javascript != None:
            namespace[javascript] = self.javascript_

        if html != None:
            namespace[html] = self.html_

        if library != None:
            namespace[library] = self.library_

        if width == None:
            width = self.width_

        if height == None:
            height = self.height_

        # Populate template
        viewer = template.substitute(namespace)

        return StatMapView(viewer, width=width, height=height)

    def _brainsprite_html(self, bg_img, stat_map_img, mask_img):
        """Create an html snippet for the brainsprite viewer (with sprite data).
        """
        # Initiate template
        resource_path = Path(__file__).resolve().parent.joinpath("data", "html")
        if self.base64:
            file_template = resource_path.joinpath("brainsprite_template_base64.html")
            file_bg = None
            file_overlay = None
            file_colormap = None
        else:
            file_template = resource_path.joinpath("brainsprite_template.html")
            file_bg = self.sprite + ".png"
            file_bg = self.sprite_overlay + ".png"
            file_colormap = self.img_colorMap + ".png"
        tpl = tempita.Template.from_filename(str(file_template), encoding="utf-8")

        # Fill template
        snippet_html = tpl.substitute(
            canvas=self.canvas,
            sprite=self.sprite,
            img_colorMap=self.img_colorMap,
            sprite_overlay=self.sprite_overlay,
            bg_base64=_save_sprite(
                output_sprite=file_bg,
                img=bg_img,
                vmax=self.bg_max_,
                vmin=self.bg_min_,
                cmap="gray",
            ),
            overlay_base64=_save_sprite(
                output_sprite=file_overlay,
                img=stat_map_img,
                vmax=self.colors_["vmax"],
                vmin=self.colors_["vmin"],
                mask=mask_img,
                cmap=self.cmap,
            ),
            colormap_base64=_save_cm(
                output_cmap=file_colormap, cmap=self.colors_["cmap"], format="png"
            ),
        )
        return snippet_html

    def _brainsprite_js(self, shape, affine, colorFont, colorBackground):
        """ Create a js snippet for the brainsprite viewer
        """
        # Initiate template
        resource_path = Path(__file__).resolve().parent.joinpath("data", "js")
        file_template = resource_path.joinpath("brainsprite_template.js")
        tpl = tempita.Template.from_filename(str(file_template), encoding="utf-8")

        return tpl.substitute(
            canvas=self.canvas,
            sprite=self.sprite,
            X=shape[0],
            Y=shape[1],
            Z=shape[2],
            sprite_overlay=self.sprite_overlay,
            X_overlay=shape[0],
            Y_overlay=shape[1],
            Z_overlay=shape[2],
            opacity=self.opacity,
            colorBackground=colorBackground,
            colorFont=colorFont,
            crosshair=float(self.draw_cross),
            affine=affine.tolist(),
            flagCoordinates=float(self.annotate),
            title=self.title,
            flagValue=float(self.value),
            X_num=self.cut_slices_[0] - 1,
            Y_num=self.cut_slices_[1] - 1,
            Z_num=self.cut_slices_[2] - 1,
            img_colorMap=self.img_colorMap,
            min=self.colors_["vmin"],
            max=self.colors_["vmax"],
            colorbar=float(not self.colorbar),
        )
