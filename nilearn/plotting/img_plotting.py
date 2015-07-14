"""
Functions to do automatic visualization of Niimg-like objects
See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.

Only matplotlib is required.
"""

# Author: Gael Varoquaux, Chris Filo Gorgolewski
# License: BSD

# Standard library imports
import functools
import numbers
import warnings

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
from scipy import ndimage
from nibabel.spatialimages import SpatialImage

from .._utils.numpy_conversions import as_ndarray

import matplotlib.pyplot as plt

from .. import _utils
from .._utils import new_img_like
from .._utils.extmath import fast_abs_percentile
from .._utils.fixes.matplotlib_backports import (cbar_outline_get_xy,
                                                 cbar_outline_set_xy)
from .._utils.ndimage import get_border_data
from ..datasets import load_mni152_template
from .displays import get_slicer, get_projector
from . import cm

################################################################################
# Core, usage-agnostic functions


def _plot_img_with_bg(img, bg_img=None, cut_coords=None,
                      output_file=None, display_mode='ortho',
                      colorbar=False, figure=None, axes=None, title=None,
                      threshold=None, annotate=True,
                      draw_cross=True, black_bg=False,
                      vmin=None, vmax=None,
                      bg_vmin=None, bg_vmax=None, interpolation="nearest",
                      display_factory=get_slicer,
                      cbar_vmin=None, cbar_vmax=None,
                      **kwargs):
    """ Internal function, please refer to the docstring of plot_img for parameters
        not listed below.

        Parameters
        ----------
        bg_vmin: float
            vmin for bg_img
        bg_vmax: float
            vmax for bg_img
        interpolation: string
            passed to the add_overlay calls
        display_factory: function
            takes a display_mode argument and return a display class
    """
    show_nan_msg = False
    if vmax is not None and np.isnan(vmax):
        vmax = None
        show_nan_msg = True
    if vmin is not None and np.isnan(vmin):
        vmin = None
        show_nan_msg = True
    if show_nan_msg:
        nan_msg = ('NaN is not permitted for the vmax and vmin arguments.\n'
                   'Tip: Use np.nan_max() instead of np.max().')
        warnings.warn(nan_msg)

    if img is not False and img is not None:
        img = _utils.check_niimg_3d(img, dtype='auto')
        data = img.get_data()
        affine = img.get_affine()

        if np.isnan(np.sum(data)):
            data = np.nan_to_num(data)

        # Deal with automatic settings of plot parameters
        if threshold == 'auto':
            # Threshold epsilon below a percentile value, to be sure that some
            # voxels pass the threshold
            threshold = fast_abs_percentile(data) - 1e-5

        img = new_img_like(img, as_ndarray(data), affine)

    display = display_factory(display_mode)(
        img,
        threshold=threshold,
        cut_coords=cut_coords,
        figure=figure, axes=axes,
        black_bg=black_bg,
        colorbar=colorbar)

    if bg_img is not None:
        bg_img = _utils.check_niimg_3d(bg_img)
        display.add_overlay(bg_img,
                           vmin=bg_vmin, vmax=bg_vmax,
                           cmap=plt.cm.gray, interpolation=interpolation)

    if img is not None and img is not False:
        display.add_overlay(new_img_like(img, data, affine),
                            threshold=threshold, interpolation=interpolation,
                            colorbar=colorbar, vmin=vmin, vmax=vmax,
                            **kwargs)

    if annotate:
        display.annotate()
    if draw_cross:
        display.draw_cross()
    if title is not None and not title == '':
        display.title(title)
    if (cbar_vmax is not None) or (cbar_vmin is not None):
        if hasattr(display, '_cbar'):
            cbar = display._cbar
            cbar_tick_locs = cbar.locator.locs
            if cbar_vmax is None:
                cbar_vmax = cbar_tick_locs.max()
            if cbar_vmin is None:
                cbar_vmin = cbar_tick_locs.min()
            new_tick_locs = np.linspace(cbar_vmin, cbar_vmax,
                                        len(cbar_tick_locs))
            cbar.ax.set_ylim(cbar.norm(cbar_vmin), cbar.norm(cbar_vmax))
            outline = cbar_outline_get_xy(cbar.outline)
            outline[:2, 1] += cbar.norm(cbar_vmin)
            outline[2:6, 1] -= (1. - cbar.norm(cbar_vmax))
            outline[6:, 1] += cbar.norm(cbar_vmin)
            cbar_outline_set_xy(cbar.outline, outline)
            cbar.set_ticks(new_tick_locs, update_ticks=True)

    if output_file is not None:
        display.savefig(output_file)
        display.close()
        display = None
    return display


def plot_img(img, cut_coords=None, output_file=None, display_mode='ortho',
            figure=None, axes=None, title=None, threshold=None,
            annotate=True, draw_cross=True, black_bg=False, colorbar=False, **kwargs):
    """ Plot cuts of a given image (by default Frontal, Axial, and Lateral)

        Parameters
        ----------
        img: Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        cut_coords: None, a tuple of floats, or an integer
            The MNI coordinates of the point where the cut is performed
            If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
            For display_mode == 'x', 'y', or 'z', then these are the
            coordinates of each cut in the corresponding direction.
            If None is given, the cuts is calculated automaticaly.
            If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
            in which case it specifies the number of cuts to perform
        output_file: string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        display_mode: {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts: 'x' - saggital, 'y' - coronal,
            'z' - axial, 'ortho' - three cuts are performed in orthogonal
            directions.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
            The axes, or the coordinates, in matplotlib figure space,
            of the axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
        threshold : a number, None, or 'auto'
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image:
            values below the threshold (in absolute value) are plotted
            as transparent. If auto is given, the threshold is determined
            magically by analysis of the image.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'"
            to matplotlib.pyplot.savefig.
        colorbar: boolean, optional
            If True, display a colorbar on the right of the plots.
        kwargs: extra keyword arguments, optional
            Extra keyword arguments passed to matplotlib.pyplot.imshow
    """
    display = _plot_img_with_bg(img, cut_coords=cut_coords,
                    output_file=output_file, display_mode=display_mode,
                    figure=figure, axes=axes, title=title,
                    threshold=threshold, annotate=annotate,
                    draw_cross=draw_cross, resampling_interpolation='continuous',
                    black_bg=black_bg, colorbar=colorbar, **kwargs)

    return display


################################################################################
# Anatomy image for background

# A constant class to serve as a sentinel for the default MNI template
class _MNI152Template(SpatialImage):
    """ This class is a constant pointing to the MNI152 Template
        provided by nilearn
    """

    data   = None
    affine = None
    vmax   = None
    _shape  = None

    def __init__(self, data=None, affine=None, header=None):
        # Comply with spatial image requirements while allowing empty init
        pass

    def load(self):
        if self.data is None:
            anat_img = load_mni152_template()
            data = anat_img.get_data()
            data = data.astype(np.float)
            anat_mask = ndimage.morphology.binary_fill_holes(data > 0)
            data = np.ma.masked_array(data, np.logical_not(anat_mask))
            self.affine = anat_img.get_affine()
            self.data = data
            self.vmax = data.max()
            self._shape = anat_img.shape

    def get_data(self):
        self.load()
        return self.data

    def get_affine(self):
        self.load()
        return self.affine
    
    @property
    def shape(self):
        self.load()
        return self._shape

    def get_shape(self):
        self.load()
        return self._shape

    def __str__(self):
        return "<MNI152Template>"


# The constant that we use as a default in functions
MNI152TEMPLATE = _MNI152Template()


def _load_anat(anat_img=MNI152TEMPLATE, dim=False, black_bg='auto'):
    """ Internal function used to load anatomy, for optional diming
    """
    vmin = None
    vmax = None
    if anat_img is False or anat_img is None:
        if black_bg == 'auto':
            # No anatomy given: no need to turn black_bg on
            black_bg = False
        return anat_img, black_bg, vmin, vmax

    if anat_img is MNI152TEMPLATE:
        anat_img.load()
        # We special-case the 'canonical anat', as we don't need
        # to do a few transforms to it.
        vmin = 0
        vmax = anat_img.vmax
        if black_bg == 'auto':
            black_bg = False
    else:
        anat_img = _utils.check_niimg_3d(anat_img)
        if dim or black_bg == 'auto':
            # We need to inspect the values of the image
            data = anat_img.get_data()
            vmin = data.min()
            vmax = data.max()
        if black_bg == 'auto':
            # Guess if the background is rather black or light based on
            # the values of voxels near the border
            background = np.median(get_border_data(data, 2))
            if background > .5 * (vmin + vmax):
                black_bg = False
            else:
                black_bg = True
    if dim:
        vmean = .5 * (vmin + vmax)
        ptp = .5 * (vmax - vmin)
        if black_bg:
            if not isinstance(dim, numbers.Number):
                dim = .8
            vmax = vmean + (1 + dim) * ptp
        else:
            if not isinstance(dim, numbers.Number):
                dim = .6
            vmin = vmean - (1 + dim) * ptp
    return anat_img, black_bg, vmin, vmax


################################################################################
# Usage-specific functions


def plot_anat(anat_img=MNI152TEMPLATE, cut_coords=None,
              output_file=None, display_mode='ortho', figure=None,
              axes=None, title=None, annotate=True, threshold=None,
              draw_cross=True, black_bg='auto', dim=False, cmap=plt.cm.gray,
              vmin=None, vmax=None, **kwargs):
    """ Plot cuts of an anatomical image (by default 3 cuts:
        Frontal, Axial, and Lateral)

        Parameters
        ----------
        anat_img : Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            The anatomical image to be used as a background. If None is
            given, nilearn tries to find a T1 template.
        cut_coords: None, a tuple of floats, or an integer
            The MNI coordinates of the point where the cut is performed
            If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
            For display_mode == 'x', 'y', or 'z', then these are the
            coordinates of each cut in the corresponding direction.
            If None is given, the cuts is calculated automaticaly.
            If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
            in which case it specifies the number of cuts to perform
        output_file: string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        display_mode: {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts: 'x' - saggital, 'y' - coronal,
            'z' - axial, 'ortho' - three cuts are performed in orthogonal
            directions.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
            The axes, or the coordinates, in matplotlib figure space,
            of the axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        threshold : a number, None, or 'auto', optional
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image:
            values below the threshold (in absolute value) are plotted
            as transparent. If auto is given, the threshold is determined
            magically by analysis of the image.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'"
            to matplotlib.pyplot.savefig.
        cmap: matplotlib colormap, optional
            The colormap for the anat
        vmin: float
            Lower bound for plotting, passed to matplotlib.pyplot.imshow
        vmax: float
            Upper bound for plotting, passed to matplotlib.pyplot.imshow

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.
    """
    anat_img, black_bg, anat_vmin, anat_vmax = _load_anat(
        anat_img,
        dim=dim, black_bg=black_bg)

    if vmin is None:
        vmin = anat_vmin
    if vmax is None:
        vmax = anat_vmax

    display = plot_img(anat_img, cut_coords=cut_coords,
                       output_file=output_file, display_mode=display_mode,
                       figure=figure, axes=axes, title=title,
                       threshold=threshold, annotate=annotate,
                       draw_cross=draw_cross, black_bg=black_bg,
                       vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    return display


def plot_epi(epi_img=None, cut_coords=None, output_file=None,
             display_mode='ortho', figure=None, axes=None, title=None,
             annotate=True, draw_cross=True, black_bg=True,
             cmap=plt.cm.spectral, vmin=None, vmax=None, **kwargs):
    """ Plot cuts of an EPI image (by default 3 cuts:
        Frontal, Axial, and Lateral)

        Parameters
        ----------
        epi_img : a nifti-image like object or a filename
            The EPI (T2*) image
        cut_coords: None, a tuple of floats, or an integer
            The MNI coordinates of the point where the cut is performed
            If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
            For display_mode == 'x', 'y', or 'z', then these are the
            coordinates of each cut in the corresponding direction.
            If None is given, the cuts is calculated automaticaly.
            If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
            in which case it specifies the number of cuts to perform
        output_file: string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        display_mode: {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts: 'x' - saggital, 'y' - coronal,
            'z' - axial, 'ortho' - three cuts are performed in orthogonal
            directions.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
            The axes, or the coordinates, in matplotlib figure space,
            of the axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'"
            to matplotlib.pyplot.savefig.
        cmap: matplotlib colormap, optional
            The colormap for specified image
        threshold : a number, None, or 'auto'
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image:
            values below the threshold (in absolute value) are plotted
            as transparent. If auto is given, the threshold is determined
            magically by analysis of the image.
        vmin: float
            Lower bound for plotting, passed to matplotlib.pyplot.imshow
        vmax: float
            Upper bound for plotting, passed to matplotlib.pyplot.imshow

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.
    """
    display = plot_img(epi_img, cut_coords=cut_coords,
                      output_file=output_file, display_mode=display_mode,
                      figure=figure, axes=axes, title=title,
                      threshold=None, annotate=annotate,
                      draw_cross=draw_cross, black_bg=black_bg,
                      cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
    return display


def plot_roi(roi_img, bg_img=MNI152TEMPLATE, cut_coords=None,
             output_file=None, display_mode='ortho', figure=None, axes=None,
             title=None, annotate=True, draw_cross=True, black_bg='auto',
             alpha=0.7, cmap=plt.cm.gist_ncar, dim=True, vmin=None, vmax=None,
             **kwargs):
    """ Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and
        Lateral)

        Parameters
        ----------
        roi_img : Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            The ROI/mask image, it could be binary mask or an atlas or ROIs
            with integer values.
        bg_img : Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            The background image that the ROI/mask will be plotted on top of. If
            not specified MNI152 template will be used.
        cut_coords: None, or a tuple of floats
            The MNI coordinates of the point where the cut is performed, in
            MNI coordinates and order.
            If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
            For display_mode == 'x', 'y', or 'z', then these are the
            coordinates of each cut in the corresponding direction.
            If None is given, the cuts is calculated automaticaly.
        output_file: string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        display_mode: {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts: 'x' - saggital, 'y' - coronal,
            'z' - axial, 'ortho' - three cuts are performed in orthogonal
            directions.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
            The axes, or the coordinates, in matplotlib figure space,
            of the axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'"
            to matplotlib.pyplot.savefig.
        threshold : a number, None, or 'auto'
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image:
            values below the threshold (in absolute value) are plotted
            as transparent. If auto is given, the threshold is determined
            magically by analysis of the image.
        vmin: float
            Lower bound for plotting, passed to matplotlib.pyplot.imshow
        vmax: float
            Upper bound for plotting, passed to matplotlib.pyplot.imshow

    """
    bg_img, black_bg, bg_vmin, bg_vmax = _load_anat(bg_img, dim=dim,
                                                    black_bg=black_bg)

    display = _plot_img_with_bg(img=roi_img, bg_img=bg_img,
                                cut_coords=cut_coords,
                                output_file=output_file,
                                display_mode=display_mode,
                                figure=figure, axes=axes, title=title,
                                annotate=annotate,
                                draw_cross=draw_cross,
                                black_bg=black_bg, threshold=0.5,
                                bg_vmin=bg_vmin, bg_vmax=bg_vmax,
                                resampling_interpolation='nearest',
                                alpha=alpha, cmap=cmap,
                                vmin=vmin, vmax=vmax, **kwargs)
    return display


def plot_stat_map(stat_map_img, bg_img=MNI152TEMPLATE, cut_coords=None,
                  output_file=None, display_mode='ortho', colorbar=True,
                  figure=None, axes=None, title=None, threshold=1e-6,
                  annotate=True, draw_cross=True, black_bg='auto',
                  cmap=cm.cold_hot, symmetric_cbar="auto",
                  dim=True, vmax=None, **kwargs):
    """ Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and
        Lateral)

        Parameters
        ----------
        stat_map_img : Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            The statistical map image
        bg_img : Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            The background image that the ROI/mask will be plotted on top of. If
            not specified MNI152 template will be used.
        cut_coords : None, a tuple of floats, or an integer
            The MNI coordinates of the point where the cut is performed
            If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
            For display_mode == 'x', 'y', or 'z', then these are the
            coordinates of each cut in the corresponding direction.
            If None is given, the cuts is calculated automaticaly.
            If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
            in which case it specifies the number of cuts to perform
        output_file : string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        display_mode : {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts: 'x' - saggital, 'y' - coronal,
            'z' - axial, 'ortho' - three cuts are performed in orthogonal
            directions.
        colorbar : boolean, optional
            If True, display a colorbar on the right of the plots.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
            The axes, or the coordinates, in matplotlib figure space,
            of the axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
        threshold : a number, None, or 'auto'
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image:
            values below the threshold (in absolute value) are plotted
            as transparent. If auto is given, the threshold is determined
            magically by analysis of the image.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'"
            to matplotlib.pyplot.savefig.
        cmap: matplotlib colormap, optional
            The colormap for specified image. The ccolormap *must* be
            symmetrical.
        symmetric_cbar: boolean or 'auto', optional, default 'auto'
            Specifies whether the colorbar should range from -vmax to vmax
            or from 0 to vmax. Setting to 'auto' will select the latter if
            the whole image is non-negative.
        vmax: float
            Upper bound for plotting, passed to matplotlib.pyplot.imshow

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.
    """
    # dim the background
    bg_img, black_bg, bg_vmin, bg_vmax = _load_anat(bg_img, dim=dim,
                                                    black_bg=black_bg)

    # make sure that the color range is symmetrical
    if vmax is None or symmetric_cbar in ['auto', False]:
        stat_map_img = _utils.check_niimg_3d(stat_map_img, dtype='auto')
        stat_map_data = stat_map_img.get_data()
        # Avoid dealing with masked_array:
        if hasattr(stat_map_data, '_mask'):
            stat_map_data = np.asarray(stat_map_data[
                    np.logical_not(stat_map_data._mask)])
        stat_map_max = np.nanmax(stat_map_data)
        stat_map_min = np.nanmin(stat_map_data)

    if symmetric_cbar == 'auto':
        symmetric_cbar = (stat_map_min < 0) and (stat_map_max > 0)

    if vmax is None:
        vmax = max(-stat_map_min, stat_map_max)

    if 'vmin' in kwargs:
        raise ValueError('plot_stat_map does not accept a "vmin" '
                         'argument, as it uses a symmetrical range '
                         'defined via the vmax argument. To threshold '
                         'the map, use the "threshold" argument')
    vmin = -vmax

    if not symmetric_cbar:
        negative_range = (stat_map_max <= 0)
        positive_range = (stat_map_min >= 0)
        if positive_range:
            cbar_vmin = 0
            cbar_vmax = None
        elif negative_range:
            cbar_vmax = 0
            cbar_vmin = None
        else:
            cbar_vmin = stat_map_min
            cbar_vmax = stat_map_max
    else:
        cbar_vmin, cbar_vmax = None, None

    display = _plot_img_with_bg(img=stat_map_img, bg_img=bg_img,
                                cut_coords=cut_coords,
                                output_file=output_file,
                                display_mode=display_mode,
                                figure=figure, axes=axes, title=title,
                                annotate=annotate, draw_cross=draw_cross,
                                black_bg=black_bg, threshold=threshold,
                                bg_vmin=bg_vmin, bg_vmax=bg_vmax, cmap=cmap,
                                vmin=vmin, vmax=vmax, colorbar=colorbar,
                                cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax,
                                resampling_interpolation='continuous',
                                **kwargs)

    return display


def plot_glass_brain(stat_map_img,
                     output_file=None, display_mode='ortho', colorbar=False,
                     figure=None, axes=None, title=None, threshold='auto',
                     annotate=True,
                     black_bg=False,
                     cmap=None,
                     alpha=0.7,
                     vmin=None, vmax=None,
                     **kwargs):
    """Plot 2d projections of an ROI/mask image (by default 3 projections:
        Frontal, Axial, and Lateral). The brain glass schematics
        are added on top of the image.

        Parameters
        ----------
        stat_map_img : Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            The statistical map image. It needs to be in MNI space
            in order to align with the brain schematics.
        output_file : string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        display_mode : {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts: 'x' - saggital, 'y' - coronal,
            'z' - axial, 'ortho' - three cuts are performed in orthogonal
            directions.
        colorbar : boolean, optional
            If True, display a colorbar on the right of the plots.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
            The axes, or the coordinates, in matplotlib figure space,
            of the axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
        threshold : a number, None, or 'auto'
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image:
            values below the threshold (in absolute value) are plotted
            as transparent. If auto is given, the threshold is determined
            magically by analysis of the image.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'"
            to matplotlib.pyplot.savefig.
        cmap: matplotlib colormap, optional
            The colormap for specified image
        alpha: float between 0 and 1
            Alpha transparency for the brain schematics
        vmin: float
            Lower bound for plotting, passed to matplotlib.pyplot.imshow
        vmax: float
            Upper bound for plotting, passed to matplotlib.pyplot.imshow

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.

    """
    if cmap is None:
        cmap = plt.cm.hot if black_bg else plt.cm.hot_r

    def display_factory(display_mode):
        return functools.partial(get_projector(display_mode), alpha=alpha)

    display = _plot_img_with_bg(img=stat_map_img,
                                output_file=output_file,
                                display_mode=display_mode,
                                figure=figure, axes=axes, title=title,
                                annotate=annotate,
                                black_bg=black_bg, threshold=threshold,
                                cmap=cmap, colorbar=colorbar,
                                display_factory=display_factory,
                                resampling_interpolation='continuous',
                                vmin=vmin, vmax=vmax,
                                **kwargs)

    return display


def plot_connectome(adjacency_matrix, node_coords,
                    node_color='auto', node_size=50,
                    edge_cmap=cm.bwr,
                    edge_vmin=None, edge_vmax=None,
                    edge_threshold=None,
                    output_file=None, display_mode='ortho',
                    figure=None, axes=None, title=None,
                    annotate=True, black_bg=False,
                    alpha=0.7,
                    edge_kwargs=None, node_kwargs=None):
    """Plot connectome on top of the brain glass schematics.

        Parameters
        ----------
        adjacency_matrix: numpy array of shape (n, n)
            represents the link strengths of the graph. Assumed to be
            a symmetric matrix.
        node_coords: numpy array_like of shape (n, 3)
            3d coordinates of the graph nodes in world space.
        node_color: color or sequence of colors
            color(s) of the nodes.
        node_size: scalar or array_like
            size(s) of the nodes in points^2.
        edge_cmap: colormap
            colormap used for representing the strength of the edges.
        edge_vmin: float, optional, default: None
        edge_vmax: float, optional, default: None
                If not None, either or both of these values will be used to
                as the minimum and maximum values to color edges. If None are
                supplied the maximum absolute value within the given threshold
                will be used as minimum (multiplied by -1) and maximum
                coloring levels.
        edge_threshold: str or number
            If it is a number only the edges with a value greater than
            edge_threshold will be shown.
            If it is a string it must finish with a percent sign,
            e.g. "25.3%", and only the edges with a abs(value) above
            the given percentile will be shown.
        output_file : string, or None, optional
            The name of an image file to export the plot to. Valid extensions
            are .png, .pdf, .svg. If output_file is not None, the plot
            is saved to a file, and the display is closed.
        display_mode : {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts: 'x' - saggital, 'y' - coronal,
            'z' - axial, 'ortho' - three cuts are performed in orthogonal
            directions.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
            The axes, or the coordinates, in matplotlib figure space,
            of the axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title displayed on the figure.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'"
            to matplotlib.pyplot.savefig.
        alpha: float between 0 and 1
            Alpha transparency for the brain schematics.
        edge_kwargs: dict
            will be passed as kwargs for each edge matlotlib Line2D.
        node_kwargs: dict
            will be passed as kwargs to the plt.scatter call that plots all
            the nodes in one go

    """
    display = plot_glass_brain(None,
                               display_mode=display_mode,
                               figure=figure, axes=axes, title=title,
                               annotate=annotate,
                               black_bg=black_bg)

    display.add_graph(adjacency_matrix, node_coords,
                      node_color=node_color, node_size=node_size,
                      edge_cmap=edge_cmap,
                      edge_vmin=edge_vmin, edge_vmax=edge_vmax,
                      edge_threshold=edge_threshold,
                      edge_kwargs=edge_kwargs, node_kwargs=node_kwargs)

    if output_file is not None:
        display.savefig(output_file)
        display.close()
        display = None

    return display
