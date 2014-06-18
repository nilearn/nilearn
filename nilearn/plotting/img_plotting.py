"""
Functions to do automatic visualization of nifti-like images

Only matplotlib is required.
"""

# Author: Gael Varoquaux, Chris Filo Gorgolewski
# License: BSD

# Standard library imports
import operator

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
from ..image.resampling import coord_transform
from .._utils.fast_maths import fast_abs_percentile
from ..datasets import load_mni152_template
from .slicers import get_slicer
from . import cm
from .find_cuts import find_cut_slices

################################################################################
# Core, usage-agnostic functions

def _plot_img_with_bg(img, bg_img=None, cut_coords=None, display_mode='ortho',
             figure=None, axes=None, title=None, threshold=None,
             annotate=True, draw_cross=True, black_bg=False,
             bg_vmin=None, bg_vmax=None, interpolation="nearest", 
             colorbar=False, **kwargs):
    """ Internal function, please refer to the docstring of plot_img
    """
    if img is not False and img is not None:
        img = _utils.check_niimg(img)
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
            # Threshold epsilon above a percentile value, to be sure that some
            # voxels are indeed threshold
            threshold = fast_abs_percentile(data) + 1e-5

        if display_mode in 'xyz':
            # Here we use a heuristic for cut indices that is well suited to
            # finding a small number of objects
            if cut_coords is None:
                cut_coords = 12
            if operator.isNumberType(cut_coords):
                cut_coords = find_cut_slices(nibabel.Nifti1Image(data, affine),
                                             direction=display_mode,
                                             n_cuts=cut_coords)

        if len(data.shape) > 3:
            if len(data.shape) == 4 and data.shape[3] == 1:
                data = data[:,:,:,0]
            else:
                raise ValueError("The provided volume has %d dimensions. Only" \
                                 " three dimensional volumes volumes are " \
                                 "supported." % len(data.shape))

        img = nibabel.Nifti1Image(as_ndarray(data), affine)

    slicer = get_slicer(display_mode).init_with_figure(img,
                                              threshold=threshold,
                                              cut_coords=cut_coords,
                                              figure=figure, axes=axes,
                                              black_bg=black_bg)


    if bg_img is not None:
        slicer.add_overlay(bg_img,
                           vmin=bg_vmin, vmax=bg_vmax,
                           cmap=pl.cm.gray, interpolation=interpolation)

    if img is not None and img is not False:
        if threshold:
            data = np.ma.masked_inside(data, -threshold, threshold, copy=False)
        slicer.add_overlay(nibabel.Nifti1Image(data, affine), 
                           interpolation=interpolation, colorbar=colorbar,
                           **kwargs)

    if black_bg:
        # To have a black background in PDF, we need to create a
        # patch in black for the background
        for ax in slicer.axes.values():
            ax.ax.imshow(np.zeros((2, 2, 3)),
                         extent=[-5000, 5000, -5000, 5000],
                         zorder=-500)

    if annotate:
        slicer.annotate()
    if draw_cross:
        slicer.draw_cross()
    if title is not None and not title == '':
        slicer.title(title)
    return slicer


def plot_img(niimg, cut_coords=None, display_mode='ortho', figure=None,
             axes=None, title=None, threshold=None,
             annotate=True, draw_cross=True, black_bg=False, **kwargs):
    """ Plot cuts of a given image (by default Frontal, Axial, and Lateral)

        Parameters
        ----------
        niimg: a nifti-image like object or a filename
            Path to a nifti file or nifti-like object
        cut_coords: None, a tuple of floats, or an integer
            The MNI coordinates of the point where the cut is performed
            If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
            For display_mode == 'x', 'y', or 'z', then these are the
            coordinates of each cut in the corresponding direction.
            If None is given, the cuts is calculated automaticaly.
            If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
            in which case it specifies the number of cuts to perform
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
            values below the threshold are plotted as transparent. If
            auto is given, the threshold is determined magically by
            analysis of the image.
        annotate: boolean, optional
            If annotate is True, positions and left/right annotation
            are added to the plot.
        draw_cross: boolean, optional
            If draw_cross is True, a cross is drawn on the plot to
            indicate the cut plosition.
        black_bg: boolean, optional
            If True, the background of the image is set to be black. If
            you whish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        kwargs: extra keyword arguments, optional
            Extra keyword arguments passed to pylab.imshow
    """
    slicer = _plot_img_with_bg(niimg, cut_coords=cut_coords,
                    display_mode=display_mode, figure=figure, axes=axes,
                    title=title, threshold=threshold, annotate=annotate,
                    draw_cross=draw_cross,
                    black_bg=black_bg, **kwargs)
    return slicer


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
            anat_img = _utils.check_niimg(anat_img)
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
            if not operator.isNumberType(dim):
                dim = .6
            if black_bg:
                vmax = vmean + (1 + dim) * ptp
            else:
                vmin = vmean - (1 + dim) * ptp
    if black_bg == 'auto':
        # No anatomy given: no need to turn black_bg on
        black_bg = False
    return anat_img, black_bg, vmin, vmax


################################################################################
# Usage-specific functions


def plot_anat(anat_img=MNI152TEMPLATE, cut_coords=None, display_mode='ortho',
              figure=None, axes=None, title=None, annotate=True,
              draw_cross=True, black_bg='auto', dim=False, cmap=pl.cm.gray,
              **kwargs):
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
        display_mode: {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts: 'x' - saggital, 'y' - coronal, 
            'z' - axial, 'ortho' - three cuts are performed in orthogonal 
            directions.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), 
            optional
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
            you whish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        cmap: matplotlib colormap, optional
            The colormap for the anat

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.
    """
    anat_img, black_bg, vmin, vmax = _load_anat(anat_img,
                                                dim=dim, black_bg=black_bg)
    slicer = plot_img(anat_img, cut_coords=cut_coords,
                      display_mode=display_mode,
                      figure=figure, axes=axes, title=title,
                      threshold=None, annotate=annotate,
                      draw_cross=draw_cross, black_bg=black_bg,
                      vmin=vmin, vmax=vmax, cmap=cmap, **kwargs)
    return slicer


def plot_epi(epi_img=None, cut_coords=None, display_mode='ortho',
             figure=None, axes=None, title=None, annotate=True,
             draw_cross=True, black_bg=True, cmap=pl.cm.spectral, **kwargs):
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
            you whish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        cmap: matplotlib colormap, optional
            The colormap for the anat

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.
    """
    slicer = plot_img(epi_img, cut_coords=cut_coords,
                      display_mode=display_mode,
                      figure=figure, axes=axes, title=title,
                      threshold=None, annotate=annotate,
                      draw_cross=draw_cross, black_bg=black_bg,
                      cmap=cmap, **kwargs)
    return slicer


def plot_roi(roi_img, bg_img=MNI152TEMPLATE, cut_coords=None,
             display_mode='ortho', figure=None, axes=None, title=None,
             annotate=True, draw_cross=True, black_bg='auto', alpha=0.7,
             cmap=pl.cm.gist_ncar, dim=True, **kwargs):
    """ Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and
        Lateral)

        Parameters
        ----------
        roi_img : a nifti-image like object or a filename
            The ROI/mask image, it could be binary mask or an atlas or ROIs with 
            integer values.
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
        display_mode: {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts: 'x' - saggital, 'y' - coronal, 
            'z' - axial, 'ortho' - three cuts are performed in orthogonal 
            directions.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), 
            optional
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
            you whish to save figures with a black background, you
            will need to pass "facecolor='k', edgecolor='k'" to pylab's
            savefig.
        cmap: matplotlib colormap, optional
            The colormap for the anat

        Notes
        -----
        Arrays should be passed in numpy convention: (x, y, z)
        ordered.
    """
    bg_img, black_bg, bg_vmin, bg_vmax = _load_anat(bg_img, dim=dim,
                                                    black_bg=black_bg)

    slicer = _plot_img_with_bg(img=roi_img, bg_img=bg_img,
                               cut_coords=cut_coords,
                               display_mode=display_mode,
                               figure=figure, axes=axes, title=title,
                               annotate=annotate, draw_cross=draw_cross,
                               black_bg=black_bg, threshold=0.5,
                               bg_vmin=bg_vmin, bg_vmax=bg_vmax,
                               alpha=alpha, cmap=cmap, **kwargs)
    return slicer


def plot_stat_map(stat_map_img, bg_img=MNI152TEMPLATE, cut_coords=None,
                  display_mode='ortho', figure=None, axes=None, title=None,
                  threshold=1e-6, annotate=True, draw_cross=True,
                  black_bg='auto', cmap=cm.cold_hot, dim=True, colorbar=True,
                  **kwargs):
    """ Plot cuts of an ROI/mask image (by default 3 cuts: Frontal, Axial, and 
        Lateral)

        Parameters
        ----------
        stat_map_img : a nifti-image like object or a filename
            The ROI/mask image, it could be binary mask or an atlas or ROIs with 
            integer values.
        bg_img : a nifti-image like object or a filename
            The background image that the ROI/mask will be plotted on top of. If
            not specified MNI152 template will be used.
        cut_coords: None, a tuple of floats, or an integer
            The MNI coordinates of the point where the cut is performed
            If display_mode is 'ortho', this should be a 3-tuple: (x, y, z)
            For display_mode == 'x', 'y', or 'z', then these are the
            coordinates of each cut in the corresponding direction.
            If None is given, the cuts is calculated automaticaly.
            If display_mode is 'x', 'y' or 'z', cut_coords can be an integer,
            in which case it specifies the number of cuts to perform
        display_mode: {'ortho', 'x', 'y', 'z'}
            Choose the direction of the cuts: 'x' - saggital, 'y' - coronal,
            'z' - axial, 'ortho' - three cuts are performed in orthogonal
            directions.
        figure : integer or matplotlib figure, optional
            Matplotlib figure used or its number. If None is given, a
            new figure is created.
        axes : matplotlib axes or 4 tuple of float: (xmin, ymin, width, height), 
            optional
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
            you whish to save figures with a black background, you
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
    stat_map_img = _utils.check_niimg(stat_map_img)
    stat_map_data = stat_map_img.get_data()
    stat_map_max = np.nanmax(stat_map_data)
    stat_map_min = np.nanmin(stat_map_data)
    vmax = max(-stat_map_min, stat_map_max)
    vmin = -vmax

    slicer = _plot_img_with_bg(img=stat_map_img, bg_img=bg_img,
                               cut_coords=cut_coords,
                               display_mode=display_mode,
                               figure=figure, axes=axes, title=title,
                               annotate=annotate, draw_cross=draw_cross,
                               black_bg=black_bg, threshold=threshold,
                               bg_vmin=bg_vmin, bg_vmax=bg_vmax, cmap=cmap,
                               vmin=vmin, vmax=vmax, colorbar=colorbar,
                               **kwargs)
    return slicer

################################################################################
# Demo functions

def demo_plot_roi(**kwargs):
    """ Demo plotting an ROI
    """
    mni_affine = MNI152TEMPLATE.get_affine()
    data = np.zeros((91, 109, 91))
    # Color a asymetric rectangle around Broca area:
    x, y, z = -52, 10, 22
    x_map, y_map, z_map = coord_transform(x, y, z,
                                          np.linalg.inv(mni_affine))
    data[int(x_map)-5:int(x_map)+5, int(y_map)-3:int(y_map)+3, 
         int(z_map)-10:int(z_map)+10] = 1
    img = nibabel.Nifti1Image(data, mni_affine)
    return plot_roi(img, title="Broca's area", **kwargs)
