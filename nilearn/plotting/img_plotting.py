"""
Functions to do automatic visualization of nifti-like images

Only matplotlib is required.
"""

# Author: Gael Varoquaux, Chris Filo Gorgolewski
# License: BSD

# Standard library imports
import operator
import functools

# Standard scientific libraries imports (more specific imports are
# delayed, so that the part module can be used without them).
import numpy as np
from scipy import ndimage

import nibabel

from .._utils.testing import skip_if_running_nose
from .._utils.numpy_conversions import as_ndarray

try:
    import pylab as pl
except ImportError:
    skip_if_running_nose('Could not import matplotlib')

from .. import _utils
from .._utils.extmath import fast_abs_percentile
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
                      bg_vmin=None, bg_vmax=None, interpolation="nearest",
                      display_factory=get_slicer,
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
    if img is not False and img is not None:
        img = _utils.check_niimg(img, ensure_3d=True)
        data = img.get_data()
        affine = img.get_affine()

        # Remove NaNs
        nan_mask = np.isnan(np.asarray(data))
        if np.any(nan_mask):
            data = data.copy()
            data[nan_mask] = 0
        del nan_mask

        # Deal with automatic settings of plot parameters
        if threshold == 'auto':
            # Threshold epsilon below a percentile value, to be sure that some
            # voxels pass the threshold
            threshold = fast_abs_percentile(data) - 1e-5

        img = nibabel.Nifti1Image(as_ndarray(data), affine)

    display = display_factory(display_mode)(
        img,
        threshold=threshold,
        cut_coords=cut_coords,
        figure=figure, axes=axes,
        black_bg=black_bg,
        colorbar=colorbar)

    if bg_img is not None:
        display.add_overlay(bg_img,
                           vmin=bg_vmin, vmax=bg_vmax,
                           cmap=pl.cm.gray, interpolation=interpolation)

    if img is not None and img is not False:
        if threshold:
            data = np.ma.masked_inside(data, -threshold, threshold, copy=False)
        display.add_overlay(nibabel.Nifti1Image(data, affine),
                           interpolation=interpolation, colorbar=colorbar,
                           **kwargs)

    if annotate:
        display.annotate()
    if draw_cross:
        display.draw_cross()
    if title is not None and not title == '':
        display.title(title)
    if output_file is not None:
        display.savefig(output_file)
        display.close()
        display = None
    return display


def plot_img(img, cut_coords=None, output_file=None, display_mode='ortho',
            figure=None, axes=None, title=None, threshold=None,
            annotate=True, draw_cross=True, black_bg=False, **kwargs):
    """ Plot cuts of a given image (by default Frontal, Axial, and Lateral)

        Parameters
        ----------
        img: Niimg-like object
            See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
            Path to a nifti file or nifti-like object
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
            The title dispayed on the figure.
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
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        kwargs: extra keyword arguments, optional
            Extra keyword arguments passed to pylab.imshow
    """
    display = _plot_img_with_bg(img, cut_coords=cut_coords,
                    output_file=output_file, display_mode=display_mode,
                    figure=figure, axes=axes, title=title,
                    threshold=threshold, annotate=annotate,
                    draw_cross=draw_cross,
                    black_bg=black_bg, **kwargs)

    return display


################################################################################
# Anatomy image for background

# A constant class to serve as a sentinel for the default MNI template
class _MNI152Template(object):
    """ This class is a constant pointing to the MNI152 Template
        provided by nilearn
    """

    data   = None
    affine = None
    vmax   = None
    _shape  = None

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
    if anat_img is not False and anat_img is not None:
        if anat_img is MNI152TEMPLATE:
            anat_img.load()
            # We special-case the 'canonical anat', as we don't need
            # to do a few transforms to it.
            vmin = 0
            vmax = anat_img.vmax
            if black_bg == 'auto':
                black_bg = False
        else:
            anat_img = _utils.check_niimg(anat_img, ensure_3d=True)
            if dim or black_bg == 'auto':
                # We need to inspect the values of the image
                data = anat_img.get_data()
                vmin = data.min()
                vmax = data.max()
            if black_bg == 'auto':
                # Guess if the background is rather black or light based on
                # the values of voxels near the border
                border_size = 2
                border_data = np.concatenate([
                        data[:border_size, :, :].ravel(),
                        data[-border_size:, :, :].ravel(),
                        data[:, :border_size, :].ravel(),
                        data[:, -border_size:, :].ravel(),
                        data[:, :, :border_size].ravel(),
                        data[:, :, -border_size:].ravel(),
                    ])
                background = np.median(border_data)
                if background > .5 * (vmin + vmax):
                    black_bg = False
                else:
                    black_bg = True
        if dim:
            vmean = .5 * (vmin + vmax)
            ptp = .5 * (vmax - vmin)
            if black_bg:
                if not operator.isNumberType(dim):
                    dim = .8
                vmax = vmean + (1 + dim) * ptp
            else:
                if not operator.isNumberType(dim):
                    dim = .6
                vmin = vmean - (1 + dim) * ptp
    if black_bg == 'auto':
        # No anatomy given: no need to turn black_bg on
        black_bg = False
    return anat_img, black_bg, vmin, vmax


################################################################################
# Usage-specific functions


def plot_anat(anat_img=MNI152TEMPLATE, cut_coords=None,
              output_file=None, display_mode='ortho', figure=None,
              axes=None, title=None, annotate=True, draw_cross=True,
              black_bg='auto', dim=False, cmap=pl.cm.gray, **kwargs):
    """ Plot cuts of an anatomical image (by default 3 cuts:
        Frontal, Axial, and Lateral)

        Parameters
        ----------
        anat_img : a nifti-image like object or a filename
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
            The title dispayed on the figure.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        cmap: matplotlib colormap, optional
            The colormap for the anat
        threshold : a number, None, or 'auto'
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image:
            values below the threshold (in absolute value) are plotted
            as transparent. If auto is given, the threshold is determined
            magically by analysis of the image.

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.
    """
    anat_img, black_bg, vmin, vmax = _load_anat(anat_img,
                                                dim=dim, black_bg=black_bg)
    # vmin and/or vmax could have been provided in the kwargs
    vmin = kwargs.pop('vmin', vmin)
    vmax = kwargs.pop('vmax', vmax)
    display = plot_img(anat_img, cut_coords=cut_coords,
                      output_file=output_file, display_mode=display_mode,
                      figure=figure, axes=axes, title=title,
                      threshold=None, annotate=annotate,
                      draw_cross=draw_cross, black_bg=black_bg,
                      vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    return display


def plot_epi(epi_img=None, cut_coords=None, output_file=None,
             display_mode='ortho', figure=None, axes=None, title=None,
             annotate=True, draw_cross=True, black_bg=True,
             cmap=pl.cm.spectral, **kwargs):
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
            The title dispayed on the figure.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        cmap: matplotlib colormap, optional
            The colormap for the anat
        threshold : a number, None, or 'auto'
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image:
            values below the threshold (in absolute value) are plotted
            as transparent. If auto is given, the threshold is determined
            magically by analysis of the image.

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
                      cmap=cmap, **kwargs)
    return display


def plot_roi(roi_img, bg_img=MNI152TEMPLATE, cut_coords=None,
             output_file=None, display_mode='ortho', figure=None, axes=None,
             title=None, annotate=True, draw_cross=True, black_bg='auto',
             alpha=0.7, cmap=pl.cm.gist_ncar, dim=True, **kwargs):
    """ Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and
        Lateral)

        Parameters
        ----------
        roi_img : a nifti-image like object or a filename
            The ROI/mask image, it could be binary mask or an atlas or ROIs
            with integer values.
        bg_img : a nifti-image like object or a filename
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
            The title dispayed on the figure.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        threshold : a number, None, or 'auto'
            If None is given, the image is not thresholded.
            If a number is given, it is used to threshold the image:
            values below the threshold (in absolute value) are plotted
            as transparent. If auto is given, the threshold is determined
            magically by analysis of the image.

    """
    bg_img, black_bg, bg_vmin, bg_vmax = _load_anat(bg_img, dim=dim,
                                                    black_bg=black_bg)

    display = _plot_img_with_bg(img=roi_img, bg_img=bg_img,
                               cut_coords=cut_coords,
                               output_file=output_file,
                               display_mode=display_mode,
                               figure=figure, axes=axes, title=title,
                               annotate=annotate, draw_cross=draw_cross,
                               black_bg=black_bg, threshold=0.5,
                               bg_vmin=bg_vmin, bg_vmax=bg_vmax,
                               alpha=alpha, cmap=cmap, **kwargs)
    return display


def plot_stat_map(stat_map_img, bg_img=MNI152TEMPLATE, cut_coords=None,
                  output_file=None, display_mode='ortho', colorbar=True,
                  figure=None, axes=None, title=None, threshold=1e-6,
                  annotate=True, draw_cross=True, black_bg='auto',
                  cmap=cm.cold_hot, dim=True, **kwargs):
    """ Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and
        Lateral)

        Parameters
        ----------
        stat_map_img : a nifti-image like object or a filename
            The statistical map image
        bg_img : a nifti-image like object or a filename
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
            The title dispayed on the figure.
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
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        cmap: matplotlib colormap, optional
            The colormap for the anat

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.
    """
    # dim the background
    bg_img, black_bg, bg_vmin, bg_vmax = _load_anat(bg_img, dim=dim,
                                                    black_bg=black_bg)

    # make sure that the color range is symmetrical
    if 'vmax' in kwargs:
        vmax = kwargs.pop('vmax')
    else:
        stat_map_img = _utils.check_niimg(stat_map_img, ensure_3d=True)
        stat_map_data = stat_map_img.get_data()
        # Avoid dealing with masked_array:
        if hasattr(stat_map_data, '_mask'):
            stat_map_data = np.asarray(stat_map_data[
                    np.logical_not(stat_map_data._mask)])
        stat_map_max = np.nanmax(stat_map_data)
        stat_map_min = np.nanmin(stat_map_data)
        vmax = max(-stat_map_min, stat_map_max)
    if 'vmin' in kwargs:
        raise ValueError('plot_stat_map does not accept a "vmin" '
                         'argument, as it uses a symmetrical range '
                         'defined via the vmax argument. To threshold '
                         'the map, use the "threshold" argument')
    vmin = -vmax

    display = _plot_img_with_bg(img=stat_map_img, bg_img=bg_img,
                               cut_coords=cut_coords,
                               output_file=output_file,
                               display_mode=display_mode,
                               figure=figure, axes=axes, title=title,
                               annotate=annotate, draw_cross=draw_cross,
                               black_bg=black_bg, threshold=threshold,
                               bg_vmin=bg_vmin, bg_vmax=bg_vmax, cmap=cmap,
                               vmin=vmin, vmax=vmax, colorbar=colorbar,
                               **kwargs)
    return display


def plot_glass_brain(stat_map_img,
                     output_file=None, display_mode='ortho',
                     figure=None, axes=None, title=None, threshold='auto',
                     annotate=True,
                     black_bg=False,
                     cmap=None,
                     alpha=0.7,
                     **kwargs):
    """Plot 2d projections of an ROI/mask image (by default 3 projections:
        Frontal, Axial, and Lateral). The brain glass schematics
        are added on top of the image.

        Parameters
        ----------
        stat_map_img : a nifti-image like object or a filename
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
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), optional
            The axes, or the coordinates, in matplotlib figure space,
            of the axes used to display the plot. If None, the complete
            figure is used.
        title : string, optional
            The title dispayed on the figure.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you wish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        cmap: matplotlib colormap, optional
            The colormap for the anat
        alpha: float between 0 and 1
            Alpha transparency for the brain schematics

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.

    """
    if cmap is None:
        cmap = pl.cm.hot if black_bg else pl.cm.hot_r

    def display_factory(display_mode):
        return functools.partial(get_projector(display_mode), alpha=alpha)

    display = _plot_img_with_bg(img=stat_map_img,
                                output_file=output_file,
                                display_mode=display_mode,
                                figure=figure, axes=axes, title=title,
                                annotate=annotate,
                                black_bg=black_bg, threshold=threshold,
                                cmap=cmap, colorbar=False,
                                display_factory=display_factory,
                                **kwargs)

    return display
