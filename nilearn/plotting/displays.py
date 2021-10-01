"""
The Slicer classes.

The main purpose of these classes is to have auto adjust of axes size to
the data with different layout of cuts.
"""

import collections.abc
import numbers
from distutils.version import LooseVersion

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
from matplotlib import cm as mpl_cm
from matplotlib import (colors,
                        lines,
                        transforms,
                        )
from matplotlib.colorbar import ColorbarBase
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrow
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from scipy import sparse, stats

from . import cm, glass_brain
from .edge_detect import _edge_map
from .find_cuts import find_xyz_cut_coords, find_cut_slices
from .. import _utils
from ..image import new_img_like
from ..image.resampling import (get_bounds, reorder_img, coord_transform,
                                get_mask_bounds)
from nilearn.image import get_data


###############################################################################
# class BaseAxes
###############################################################################

class BaseAxes(object):
    """ An MPL axis-like object that displays a 2D view of 3D volumes
    """

    def __init__(self, ax, direction, coord):
        """ An MPL axis-like object that displays a cut of 3D volumes

        Parameters
        ----------
        ax : A MPL axes instance
            The axes in which the plots will be drawn.

        direction : {'x', 'y', 'z'}
            The directions of the view.

        coord : float
            The coordinate along the direction of the cut.

        """
        self.ax = ax
        self.direction = direction
        self.coord = coord
        self._object_bounds = list()
        self.shape = None

    def transform_to_2d(self, data, affine):
        raise NotImplementedError("'transform_to_2d' needs to be implemented "
                                  "in derived classes'")

    def add_object_bounds(self, bounds):
        """Ensures that axes get rescaled when adding object bounds

        """
        old_object_bounds = self.get_object_bounds()
        self._object_bounds.append(bounds)
        new_object_bounds = self.get_object_bounds()

        if new_object_bounds != old_object_bounds:
            self.ax.axis(self.get_object_bounds())

    def draw_2d(self, data_2d, data_bounds, bounding_box,
                type='imshow', **kwargs):
        # kwargs messaging
        kwargs['origin'] = 'upper'

        if self.direction == 'y':
            (xmin, xmax), (_, _), (zmin, zmax) = data_bounds
            (xmin_, xmax_), (_, _), (zmin_, zmax_) = bounding_box
        elif self.direction in 'xlr':
            (_, _), (xmin, xmax), (zmin, zmax) = data_bounds
            (_, _), (xmin_, xmax_), (zmin_, zmax_) = bounding_box
        elif self.direction == 'z':
            (xmin, xmax), (zmin, zmax), (_, _) = data_bounds
            (xmin_, xmax_), (zmin_, zmax_), (_, _) = bounding_box
        else:
            raise ValueError('Invalid value for direction %s' %
                             self.direction)
        ax = self.ax
        # Here we need to do a copy to avoid having the image changing as
        # we change the data
        im = getattr(ax, type)(data_2d.copy(),
                               extent=(xmin, xmax, zmin, zmax),
                               **kwargs)

        self.add_object_bounds((xmin_, xmax_, zmin_, zmax_))
        self.shape = data_2d.T.shape

        # The bounds of the object do not take into account a possible
        # inversion of the axis. As such, we check that the axis is properly
        # inverted when direction is left
        if self.direction == 'l' and not (ax.get_xlim()[0] > ax.get_xlim()[1]):
            ax.invert_xaxis()

        return im

    def get_object_bounds(self):
        """ Return the bounds of the objects on this axes.
        """
        if len(self._object_bounds) == 0:
            # Nothing plotted yet
            return -.01, .01, -.01, .01
        xmins, xmaxs, ymins, ymaxs = np.array(self._object_bounds).T
        xmax = max(xmaxs.max(), xmins.max())
        xmin = min(xmins.min(), xmaxs.min())
        ymax = max(ymaxs.max(), ymins.max())
        ymin = min(ymins.min(), ymaxs.min())

        return xmin, xmax, ymin, ymax

    def draw_left_right(self, size, bg_color, **kwargs):
        if self.direction in 'xlr':
            return
        ax = self.ax
        ax.text(.1, .95, 'L',
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='top',
                size=size,
                bbox=dict(boxstyle="square,pad=0",
                          ec=bg_color, fc=bg_color, alpha=1),
                **kwargs)

        ax.text(.9, .95, 'R',
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                size=size,
                bbox=dict(boxstyle="square,pad=0", ec=bg_color, fc=bg_color),
                **kwargs)

    def draw_scale_bar(self, bg_color, size=5.0, units='cm',
                       fontproperties=None, frameon=False, loc=4, pad=.1,
                       borderpad=.5, sep=5, size_vertical=0, label_top=False,
                       color='black', fontsize=None, **kwargs):
        """ Adds a scale bar annotation to the display

        Parameters
        ----------
        bgcolor : matplotlib color: str or (r, g, b) value
            The background color of the scale bar annotation.

        size : float, optional
            Horizontal length of the scale bar, given in `units`.
            Default=5.0.

        units : str, optional
            Physical units of the scale bar (`'cm'` or `'mm'`).
            Default='cm'.

        fontproperties : ``matplotlib.font_manager.FontProperties`` or dict, optional
            Font properties for the label text.

        frameon : Boolean, optional
            Whether the scale bar is plotted with a border. Default=False.

        loc : int, optional
            Location of this scale bar. Valid location codes are documented
            `here <https://matplotlib.org/mpl_toolkits/axes_grid/\
            api/anchored_artists_api.html#mpl_toolkits.axes_grid1.\
            anchored_artists.AnchoredSizeBar>`__.
            Default=4.

        pad : int of float, optional
            Padding around the label and scale bar, in fraction of the font
            size. Default=0.1.

        borderpad : int or float, optional
            Border padding, in fraction of the font size. Default=0.5.

        sep : int or float, optional
            Separation between the label and the scale bar, in points.
            Default=5.

        size_vertical : int or float, optional
            Vertical length of the size bar, given in `units`. Default=0.

        label_top : bool, optional
            If True, the label will be over the scale bar. Default=False.

        color : str, optional
            Color for the scale bar and label. Default='black'.

        fontsize : int, optional
            Label font size (overwrites the size passed in through the
            ``fontproperties`` argument).

        **kwargs :
            Keyworded arguments to pass to
            ``matplotlib.offsetbox.AnchoredOffsetbox``.

        """
        axis = self.ax
        fontproperties = fontproperties or FontProperties()
        if fontsize:
            fontproperties.set_size(fontsize)
        width_mm = size
        if units == 'cm':
            width_mm *= 10

        anchor_size_bar = AnchoredSizeBar(
            axis.transData,
            width_mm,
            '%g%s' % (size, units),
            fontproperties=fontproperties,
            frameon=frameon,
            loc=loc,
            pad=pad,
            borderpad=borderpad,
            sep=sep,
            size_vertical=size_vertical,
            label_top=label_top,
            color=color,
            **kwargs)

        if frameon:
            anchor_size_bar.patch.set_facecolor(bg_color)
            anchor_size_bar.patch.set_edgecolor('none')
        axis.add_artist(anchor_size_bar)

    def draw_position(self, size, bg_color, **kwargs):
        raise NotImplementedError("'draw_position' should be implemented "
                                  "in derived classes")


###############################################################################
# class CutAxes
###############################################################################

class CutAxes(BaseAxes):
    """ An MPL axis-like object that displays a cut of 3D volumes
    """
    def transform_to_2d(self, data, affine):
        """ Cut the 3D volume into a 2D slice

        Parameters
        ----------
        data : 3D ndarray
            The 3D volume to cut.

        affine : 4x4 ndarray
            The affine of the volume.

        """
        coords = [0, 0, 0]
        coords['xyz'.index(self.direction)] = self.coord
        x_map, y_map, z_map = [int(np.round(c)) for c in
                               coord_transform(coords[0],
                                               coords[1],
                                               coords[2],
                                               np.linalg.inv(affine))]
        if self.direction == 'y':
            cut = np.rot90(data[:, y_map, :])
        elif self.direction == 'x':
            cut = np.rot90(data[x_map, :, :])
        elif self.direction == 'z':
            cut = np.rot90(data[:, :, z_map])
        else:
            raise ValueError('Invalid value for direction %s' %
                             self.direction)
        return cut

    def draw_position(self, size, bg_color, decimals=False, **kwargs):
        if decimals:
            text = '%s=%.{}f'.format(decimals)
            coord = float(self.coord)
        else:
            text = '%s=%i'
            coord = self.coord
        ax = self.ax
        ax.text(0, 0, text % (self.direction, coord),
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='bottom',
                size=size,
                bbox=dict(boxstyle="square,pad=0",
                          ec=bg_color, fc=bg_color, alpha=1),
                **kwargs)


def _get_index_from_direction(direction):
    """Returns numerical index from direction
    """
    directions = ['x', 'y', 'z']
    try:
        # l and r are subcases of x
        if direction in 'lr':
            index = 0
        else:
            index = directions.index(direction)
    except ValueError:
        message = (
            '{0} is not a valid direction. '
            "Allowed values are 'l', 'r', 'x', 'y' and 'z'").format(direction)
        raise ValueError(message)
    return index


def _coords_3d_to_2d(coords_3d, direction, return_direction=False):
    """Project 3d coordinates into 2d ones given the direction of a cut
    """
    index = _get_index_from_direction(direction)
    dimensions = [0, 1, 2]
    dimensions.pop(index)

    if return_direction:
        return coords_3d[:, dimensions], coords_3d[:, index]

    return coords_3d[:, dimensions]


###############################################################################
# class GlassBrainAxes
###############################################################################

class GlassBrainAxes(BaseAxes):
    """An MPL axis-like object that displays a 2D projection of 3D
    volumes with a schematic view of the brain.

    """
    def __init__(self, ax, direction, coord, plot_abs=True, **kwargs):
        super(GlassBrainAxes, self).__init__(ax, direction, coord)
        self._plot_abs = plot_abs
        if ax is not None:
            object_bounds = glass_brain.plot_brain_schematics(ax,
                                                              direction,
                                                              **kwargs)
            self.add_object_bounds(object_bounds)

    def transform_to_2d(self, data, affine):
        """ Returns the maximum of the absolute value of the 3D volume
        along an axis.

        Parameters
        ----------
        data : 3D ndarray
            The 3D volume.

        affine : 4x4 ndarray
            The affine of the volume.

        """
        if self.direction in 'xlr':
            max_axis = 0
        else:
            max_axis = '.yz'.index(self.direction)

        # set unselected brain hemisphere activations to 0

        if self.direction == 'l':
            x_center, _, _, _ = np.dot(np.linalg.inv(affine),
                                       np.array([0, 0, 0, 1]))
            data_selection = data[:int(x_center), :, :]
        elif self.direction == 'r':
            x_center, _, _, _ = np.dot(np.linalg.inv(affine),
                                       np.array([0, 0, 0, 1]))
            data_selection = data[int(x_center):, :, :]
        else:
            data_selection = data

        # We need to make sure data_selection is not empty in the x axis
        # This should be the case since we expect images in MNI space
        if data_selection.shape[0] == 0:
            data_selection = data

        if not self._plot_abs:
            # get the shape of the array we are projecting to
            new_shape = list(data.shape)
            del new_shape[max_axis]

            # generate a 3D indexing array that points to max abs value in the
            # current projection
            a1, a2 = np.indices(new_shape)
            inds = [a1, a2]
            inds.insert(max_axis, np.abs(data_selection).argmax(axis=max_axis))

            # take the values where the absolute value of the projection
            # is the highest
            maximum_intensity_data = data_selection[tuple(inds)]
        else:
            maximum_intensity_data = np.abs(data_selection).max(axis=max_axis)

        # This work around can be removed bumping matplotlib > 2.1.0. See #1815
        # in nilearn for the invention of this work around

        if self.direction == 'l' and data_selection.min() is np.ma.masked and \
                not (self.ax.get_xlim()[0] > self.ax.get_xlim()[1]):
            self.ax.invert_xaxis()

        return np.rot90(maximum_intensity_data)

    def draw_position(self, size, bg_color, **kwargs):
        # It does not make sense to draw crosses for the position of
        # the cuts since we are taking the max along one axis
        pass

    def _add_markers(self, marker_coords, marker_color, marker_size, **kwargs):
        """Plot markers

        In the case of 'l' and 'r' directions (for hemispheric projections),
        markers in the coordinate x == 0 are included in both hemispheres.

        """
        marker_coords_2d = _coords_3d_to_2d(marker_coords, self.direction)
        xdata, ydata = marker_coords_2d.T

        # Allow markers only in their respective hemisphere when appropriate
        if self.direction in 'lr':
            if not isinstance(marker_color, str) and \
                    not isinstance(marker_color, np.ndarray):
                marker_color = np.asarray(marker_color)
            relevant_coords = []
            xcoords, ycoords, zcoords = marker_coords.T
            for cidx, xc in enumerate(xcoords):
                if self.direction == 'r' and xc >= 0:
                    relevant_coords.append(cidx)
                elif self.direction == 'l' and xc <= 0:
                    relevant_coords.append(cidx)
            xdata = xdata[relevant_coords]
            ydata = ydata[relevant_coords]
            # if marker_color is string for example 'red' or 'blue', then
            # we pass marker_color as it is to matplotlib scatter without
            # making any selection in 'l' or 'r' color.
            # More likely that user wants to display all nodes to be in
            # same color.
            if not isinstance(marker_color, str) and \
                    len(marker_color) != 1:
                marker_color = marker_color[relevant_coords]

            if not isinstance(marker_size, numbers.Number):
                marker_size = np.asarray(marker_size)[relevant_coords]

        defaults = {'marker': 'o',
                    'zorder': 1000}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        self.ax.scatter(xdata, ydata, s=marker_size,
                        c=marker_color, **kwargs)

    def _add_lines(self, line_coords, line_values, cmap,
                   vmin=None, vmax=None, directed=False, **kwargs):
        """Plot lines

        Parameters
        ----------
        line_coords : list of numpy arrays of shape (2, 3)
            3d coordinates of lines start points and end points.

        line_values : array_like
            Values of the lines.

        cmap : colormap
            Colormap used to map line_values to a color.

        vmin, vmax : float, optional
            If not None, either or both of these values will be used to
            as the minimum and maximum values to color lines. If None are
            supplied the maximum absolute value within the given threshold
            will be used as minimum (multiplied by -1) and maximum
            coloring levels.

        directed : boolean, optional
            Add arrows instead of lines if set to True. Use this when plotting
            directed graphs for example. Default=False.

        kwargs : dict
            Additional arguments to pass to matplotlib Line2D.

        """
        # colormap for colorbar
        self.cmap = cmap
        if vmin is None and vmax is None:
            abs_line_values_max = np.abs(line_values).max()
            vmin = -abs_line_values_max
            vmax = abs_line_values_max
        elif vmin is None:
            if vmax > 0:
                vmin = -vmax
            else:
                raise ValueError(
                    "If vmax is set to a non-positive number "
                    "then vmin needs to be specified"
                )
        elif vmax is None:
            if vmin < 0:
                vmax = -vmin
            else:
                raise ValueError(
                    "If vmin is set to a non-negative number "
                    "then vmax needs to be specified"
                )
        norm = colors.Normalize(vmin=vmin,
                                vmax=vmax)
        # normalization useful for colorbar
        self.norm = norm
        abs_norm = colors.Normalize(vmin=0,
                                    vmax=vmax)
        value_to_color = plt.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba

        # Allow lines only in their respective hemisphere when appropriate
        if self.direction in 'lr':
            relevant_lines = []
            for lidx, line in enumerate(line_coords):
                if self.direction == 'r':
                    if line[0, 0] >= 0 and line[1, 0] >= 0:
                        relevant_lines.append(lidx)
                elif self.direction == 'l':
                    if line[0, 0] < 0 and line[1, 0] < 0:
                        relevant_lines.append(lidx)
            line_coords = np.array(line_coords)[relevant_lines]
            line_values = line_values[relevant_lines]

        for start_end_point_3d, line_value in zip(
                line_coords, line_values):
            start_end_point_2d = _coords_3d_to_2d(start_end_point_3d,
                                                  self.direction)

            color = value_to_color(line_value)
            abs_line_value = abs(line_value)
            linewidth = 1 + 2 * abs_norm(abs_line_value)
            # Hacky way to put the strongest connections on top of the weakest
            # note sign does not matter hence using 'abs'
            zorder = 10 + 10 * abs_norm(abs_line_value)
            this_kwargs = {'color': color, 'linewidth': linewidth,
                           'zorder': zorder}
            # kwargs should have priority over this_kwargs so that the
            # user can override the default logic
            this_kwargs.update(kwargs)
            xdata, ydata = start_end_point_2d.T
            # If directed is True, add an arrow
            if directed:
                dx = xdata[1] - xdata[0]
                dy = ydata[1] - ydata[0]
                # Hack to avoid empty arrows to crash with
                # matplotlib versions older than 3.1
                # This can be removed once support for
                # matplotlib pre 3.1 has been dropped.
                if dx == 0 and dy == 0:
                    arrow = FancyArrow(xdata[0], ydata[0],
                                       dx, dy)
                else:
                    arrow = FancyArrow(xdata[0], ydata[0],
                                       dx, dy,
                                       length_includes_head=True,
                                       width=linewidth,
                                       head_width=3*linewidth,
                                       **this_kwargs)
                self.ax.add_patch(arrow)
            # Otherwise a line
            else:
                line = lines.Line2D(xdata, ydata, **this_kwargs)
                self.ax.add_line(line)


###############################################################################
# class BaseSlicer
###############################################################################

class BaseSlicer(object):
    """ The main purpose of these class is to have auto adjust of axes size
        to the data with different layout of cuts.

    """
    # This actually encodes the figsize for only one axe
    _default_figsize = [2.2, 2.6]
    _axes_class = CutAxes

    def __init__(self, cut_coords, axes=None, black_bg=False,
                 brain_color=(0.5, 0.5, 0.5), **kwargs):
        """ Create 3 linked axes for plotting orthogonal cuts.

        Parameters
        ----------
        cut_coords : 3 tuple of ints
            The cut position, in world space.

        axes : matplotlib axes object, optional
            The axes that will be subdivided in 3.

        black_bg : boolean, optional
            If True, the background of the figure will be put to
            black. If you wish to save figures with a black background,
            you will need to pass "facecolor='k', edgecolor='k'"
            to matplotlib.pyplot.savefig. Default=False.

        brain_color : tuple, optional
            The brain color to use as the background color (e.g., for
            transparent colorbars).
            Default=(0.5, 0.5, 0.5)

        """
        self.cut_coords = cut_coords
        if axes is None:
            axes = plt.axes((0., 0., 1., 1.))
            axes.axis('off')
        self.frame_axes = axes
        axes.set_zorder(1)
        bb = axes.get_position()
        self.rect = (bb.x0, bb.y0, bb.x1, bb.y1)
        self._black_bg = black_bg
        self._brain_color = brain_color
        self._colorbar = False
        self._colorbar_width = 0.05 * bb.width
        self._colorbar_margin = dict(left=0.25 * bb.width,
                                     right=0.02 * bb.width,
                                     top=0.05 * bb.height,
                                     bottom=0.05 * bb.height)
        self._init_axes(**kwargs)

    @staticmethod
    def find_cut_coords(img=None, threshold=None, cut_coords=None):
        # Implement this as a staticmethod or a classmethod when
        # subclassing
        raise NotImplementedError

    @classmethod
    def init_with_figure(cls, img, threshold=None,
                         cut_coords=None, figure=None, axes=None,
                         black_bg=False, leave_space=False, colorbar=False,
                         brain_color=(0.5, 0.5, 0.5), **kwargs):
        "Initialize the slicer with an image"
        # deal with "fake" 4D images
        if img is not None and img is not False:
            img = _utils.check_niimg_3d(img)

        cut_coords = cls.find_cut_coords(img, threshold, cut_coords)

        if isinstance(axes, plt.Axes) and figure is None:
            figure = axes.figure

        if not isinstance(figure, plt.Figure):
            # Make sure that we have a figure
            figsize = cls._default_figsize[:]

            # Adjust for the number of axes
            figsize[0] *= len(cut_coords)

            # Make space for the colorbar
            if colorbar:
                figsize[0] += .7

            facecolor = 'k' if black_bg else 'w'

            if leave_space:
                figsize[0] += 3.4
            figure = plt.figure(figure, figsize=figsize,
                                facecolor=facecolor)
        if isinstance(axes, plt.Axes):
            assert axes.figure is figure, ("The axes passed are not "
                                           "in the figure")

        if axes is None:
            axes = [0., 0., 1., 1.]
            if leave_space:
                axes = [0.3, 0, .7, 1.]
        if isinstance(axes, collections.abc.Sequence):
            axes = figure.add_axes(axes)
        # People forget to turn their axis off, or to set the zorder, and
        # then they cannot see their slicer
        axes.axis('off')
        return cls(cut_coords, axes, black_bg, brain_color, **kwargs)

    def title(self, text, x=0.01, y=0.99, size=15, color=None, bgcolor=None,
              alpha=1, **kwargs):
        """ Write a title to the view.

        Parameters
        ----------
        text : string
            The text of the title.

        x : float, optional
            The horizontal position of the title on the frame in
            fraction of the frame width. Default=0.01.

        y : float, optional
            The vertical position of the title on the frame in
            fraction of the frame height. Default=0.99.

        size : integer, optional
            The size of the title text. Default=15.

        color : matplotlib color specifier, optional
            The color of the font of the title.

        bgcolor : matplotlib color specifier, optional
            The color of the background of the title.

        alpha : float, optional
            The alpha value for the background. Default=1.

        kwargs :
            Extra keyword arguments are passed to matplotlib's text
            function.

        """
        if color is None:
            color = 'k' if self._black_bg else 'w'
        if bgcolor is None:
            bgcolor = 'w' if self._black_bg else 'k'
        if hasattr(self, '_cut_displayed'):
            # Adapt to the case of mosaic plotting
            if isinstance(self.cut_coords, dict):
                first_axe = self._cut_displayed[-1]
                first_axe = (first_axe, self.cut_coords[first_axe][0])
            else:
                first_axe = self._cut_displayed[0]
        else:
            first_axe = self.cut_coords[0]
        ax = self.axes[first_axe].ax
        ax.text(x, y, text,
                transform=self.frame_axes.transAxes,
                horizontalalignment='left',
                verticalalignment='top',
                size=size, color=color,
                bbox=dict(boxstyle="square,pad=.3",
                          ec=bgcolor, fc=bgcolor, alpha=alpha),
                zorder=1000,
                **kwargs)
        ax.set_zorder(1000)

    def add_overlay(self, img, threshold=1e-6, colorbar=False, **kwargs):
        """ Plot a 3D map in all the views.

        Parameters
        -----------
        img : Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            If it is a masked array, only the non-masked part will be plotted.

        threshold : Int or Float or None, optional
            If None is given, the maps are not thresholded.
            If a number is given, it is used to threshold the maps:
            values below the threshold (in absolute value) are
            plotted as transparent. Default=1e-6.

        colorbar : boolean, optional
            If True, display a colorbar on the right of the plots.
            Default=False.

        kwargs :
            Extra keyword arguments are passed to imshow.

        """
        if colorbar and self._colorbar:
            raise ValueError("This figure already has an overlay with a "
                             "colorbar.")
        else:
            self._colorbar = colorbar

        img = _utils.check_niimg_3d(img)

        # Make sure that add_overlay shows consistent default behavior
        # with plot_stat_map
        kwargs.setdefault('interpolation', 'nearest')
        ims = self._map_show(img, type='imshow', threshold=threshold, **kwargs)

        # `ims` can be empty in some corner cases, look at test_img_plotting.test_outlier_cut_coords.
        if colorbar and ims:
            self._show_colorbar(ims[0].cmap, ims[0].norm, threshold)

        plt.draw_if_interactive()

    def add_contours(self, img, threshold=1e-6, filled=False, **kwargs):
        """ Contour a 3D map in all the views.

        Parameters
        -----------
        img : Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Provides image to plot.

        threshold : Int or Float or None, optional
            If None is given, the maps are not thresholded.
            If a number is given, it is used to threshold the maps,
            values below the threshold (in absolute value) are plotted
            as transparent. Default=1e-6.

        filled : boolean, optional
            If filled=True, contours are displayed with color fillings.
            Default=False.

        kwargs :
            Extra keyword arguments are passed to contour, see the
            documentation of pylab.contour and see pylab.contourf documentation
            for arguments related to contours with fillings.
            Useful, arguments are typical "levels", which is a
            list of values to use for plotting a contour or contour
            fillings (if filled=True), and
            "colors", which is one color or a list of colors for
            these contours.

        Notes
        -----
        If colors are not specified, default coloring choices
        (from matplotlib) for contours and contour_fillings can be
        different.

        """
        if not filled:
            threshold = None
        self._map_show(img, type='contour', threshold=threshold, **kwargs)
        if filled:
            if 'levels' in kwargs:
                levels = kwargs['levels']
                if len(levels) <= 1:
                    # contour fillings levels should be given as (lower, upper).
                    levels.append(np.inf)

            self._map_show(img, type='contourf', threshold=threshold, **kwargs)

        plt.draw_if_interactive()

    def _map_show(self, img, type='imshow',
                  resampling_interpolation='continuous',
                  threshold=None, **kwargs):
        # In the special case where the affine of img is not diagonal,
        # the function `reorder_img` will trigger a resampling
        # of the provided image with a continuous interpolation
        # since this is the default value here. In the special
        # case where this image is binary, such as when this function
        # is called from `add_contours`, continuous interpolation
        # does not make sense and we turn to nearest interpolation instead.
        if _utils.niimg._is_binary_niimg(img):
            img = reorder_img(img, resample='nearest')
        else:
            img = reorder_img(img, resample=resampling_interpolation)
        threshold = float(threshold) if threshold is not None else None

        if threshold is not None:
            data = _utils.niimg._safe_get_data(img, ensure_finite=True)
            if threshold == 0:
                data = np.ma.masked_equal(data, 0, copy=False)
            else:
                data = np.ma.masked_inside(data, -threshold, threshold,
                                           copy=False)
            img = new_img_like(img, data, img.affine)

        affine = img.affine
        data = _utils.niimg._safe_get_data(img, ensure_finite=True)
        data_bounds = get_bounds(data.shape, affine)
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = data_bounds

        xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = \
            xmin, xmax, ymin, ymax, zmin, zmax

        # Compute tight bounds
        if type in ('contour', 'contourf'):
            # Define a pseudo threshold to have a tight bounding box
            if 'levels' in kwargs:
                thr = 0.9 * np.min(np.abs(kwargs['levels']))
            else:
                thr = 1e-6
            not_mask = np.logical_or(data > thr, data < -thr)
            xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = \
                get_mask_bounds(new_img_like(img, not_mask, affine))
        elif hasattr(data, 'mask') and isinstance(data.mask, np.ndarray):
            not_mask = np.logical_not(data.mask)
            xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = \
                get_mask_bounds(new_img_like(img, not_mask, affine))

        data_2d_list = []
        for display_ax in self.axes.values():
            try:
                data_2d = display_ax.transform_to_2d(data, affine)
            except IndexError:
                # We are cutting outside the indices of the data
                data_2d = None

            data_2d_list.append(data_2d)

        if kwargs.get('vmin') is None:
            kwargs['vmin'] = np.ma.min([d.min() for d in data_2d_list
                                        if d is not None])
        if kwargs.get('vmax') is None:
            kwargs['vmax'] = np.ma.max([d.max() for d in data_2d_list
                                        if d is not None])

        bounding_box = (xmin_, xmax_), (ymin_, ymax_), (zmin_, zmax_)
        ims = []
        to_iterate_over = zip(self.axes.values(), data_2d_list)
        for display_ax, data_2d in to_iterate_over:
            if data_2d is not None and data_2d.min() is not np.ma.masked:
                # If data_2d is completely masked, then there is nothing to
                # plot. Hence, no point to do imshow(). Moreover, we see
                # problem came up with matplotlib 2.1.0 (issue #9280) when
                # data is completely masked or with numpy < 1.14
                # (issue #4595). This work around can be removed when bumping
                # matplotlib version above 2.1.0
                im = display_ax.draw_2d(data_2d, data_bounds, bounding_box,
                                        type=type, **kwargs)
                ims.append(im)
        return ims

    def _show_colorbar(self, cmap, norm, threshold=None):
        """Displays the colorbar.

        Parameters
        ----------
        cmap : a matplotlib colormap
            The colormap used.

        norm : a matplotlib.colors.Normalize object
            This object is typically found as the 'norm' attribute of an
            matplotlib.image.AxesImage.

        threshold : float or None, optional
            The absolute value at which the colorbar is thresholded.

        """
        if threshold is None:
            offset = 0
        else:
            offset = threshold
        if offset > norm.vmax:
            offset = norm.vmax

        # create new  axis for the colorbar
        figure = self.frame_axes.figure
        _, y0, x1, y1 = self.rect
        height = y1 - y0
        x_adjusted_width = self._colorbar_width / len(self.axes)
        x_adjusted_margin = self._colorbar_margin['right'] / len(self.axes)
        lt_wid_top_ht = [x1 - (x_adjusted_width + x_adjusted_margin),
                         y0 + self._colorbar_margin['top'],
                         x_adjusted_width,
                         height - (self._colorbar_margin['top'] +
                                   self._colorbar_margin['bottom'])]
        self._colorbar_ax = figure.add_axes(lt_wid_top_ht)
        if LooseVersion(matplotlib.__version__) >= LooseVersion("1.6"):
            self._colorbar_ax.set_facecolor('w')
        else:
            self._colorbar_ax.set_axis_bgcolor('w')

        our_cmap = mpl_cm.get_cmap(cmap)
        # edge case where the data has a single value
        # yields a cryptic matplotlib error message
        # when trying to plot the color bar
        nb_ticks = 5 if norm.vmin != norm.vmax else 1
        ticks = np.linspace(norm.vmin, norm.vmax, nb_ticks)
        bounds = np.linspace(norm.vmin, norm.vmax, our_cmap.N)

        # some colormap hacking
        cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
        transparent_start = int(norm(-offset, clip=True) * (our_cmap.N - 1))
        transparent_stop = int(norm(offset, clip=True) * (our_cmap.N - 1))
        for i in range(transparent_start, transparent_stop):
            cmaplist[i] = self._brain_color + (0.,)  # transparent
        if norm.vmin == norm.vmax:  # len(np.unique(data)) == 1 ?
            return
        else:
            our_cmap = colors.LinearSegmentedColormap.from_list(
                'Custom cmap', cmaplist, our_cmap.N)

        self._cbar = ColorbarBase(
            self._colorbar_ax, ticks=ticks, norm=norm,
            orientation='vertical', cmap=our_cmap, boundaries=bounds,
            spacing='proportional', format='%.2g')
        self._cbar.ax.set_facecolor(self._brain_color)

        self._colorbar_ax.yaxis.tick_left()
        tick_color = 'w' if self._black_bg else 'k'
        outline_color = 'w' if self._black_bg else 'k'

        for tick in self._colorbar_ax.yaxis.get_ticklabels():
            tick.set_color(tick_color)
        self._colorbar_ax.yaxis.set_tick_params(width=0)
        self._cbar.outline.set_edgecolor(outline_color)

    def add_edges(self, img, color='r'):
        """ Plot the edges of a 3D map in all the views.

        Parameters
        ----------
        img : Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            The 3D map to be plotted.
            If it is a masked array, only the non-masked part will be plotted.

        color : matplotlib color: string or (r, g, b) value
            The color used to display the edge map.
            Default='r'.

        """
        img = reorder_img(img, resample='continuous')
        data = get_data(img)
        affine = img.affine
        single_color_cmap = colors.ListedColormap([color])
        data_bounds = get_bounds(data.shape, img.affine)

        # For each ax, cut the data and plot it
        for display_ax in self.axes.values():
            try:
                data_2d = display_ax.transform_to_2d(data, affine)
                edge_mask = _edge_map(data_2d)
            except IndexError:
                # We are cutting outside the indices of the data
                continue
            display_ax.draw_2d(edge_mask, data_bounds, data_bounds,
                               type='imshow', cmap=single_color_cmap)

        plt.draw_if_interactive()

    def add_markers(self, marker_coords, marker_color='r', marker_size=30,
                    **kwargs):
        """Add markers to the plot.

        Parameters
        ----------
        marker_coords : array of size (n_markers, 3)
            Coordinates of the markers to plot. For each slice, only markers
            that are 2 millimeters away from the slice are plotted.

        marker_color : pyplot compatible color or list of shape (n_markers,), optional
            List of colors for each marker that can be string or matplotlib colors.
            Default='r'.

        marker_size : single float or list of shape (n_markers,), optional
            Size in pixel for each marker. Default=30.

        """
        defaults = {'marker': 'o',
                    'zorder': 1000}
        marker_coords = np.asanyarray(marker_coords)
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        for display_ax in self.axes.values():
            direction = display_ax.direction
            coord = display_ax.coord
            marker_coords_2d, third_d = _coords_3d_to_2d(
                marker_coords, direction, return_direction=True)
            xdata, ydata = marker_coords_2d.T
	        # Allow markers only in their respective hemisphere when appropriate
            marker_color_ = marker_color
            if direction in ('lr'):
                if (not isinstance(marker_color, str) and
	            not isinstance(marker_color, np.ndarray)):
                    marker_color_ = np.asarray(marker_color)
                xcoords, ycoords, zcoords = marker_coords.T
                if direction == 'r':
                    relevant_coords = (xcoords >= 0)
                elif direction == 'l':
                    relevant_coords = (xcoords <= 0)
                xdata = xdata[relevant_coords]
                ydata = ydata[relevant_coords]
                if (not isinstance(marker_color, str) and
                        len(marker_color) != 1):
                    marker_color_ = marker_color_[relevant_coords]
            # Check if coord has integer represents a cut in direction
            # to follow the heuristic. If no foreground image is given
            # coordinate is empty or None. This case is valid for plotting
            # markers on glass brain without any foreground image.
            if isinstance(coord, numbers.Number):
                # Heuristic that plots only markers that are 2mm away
                # from the current slice.
                # XXX: should we keep this heuristic?
                mask = np.abs(third_d - coord) <= 2.
                xdata = xdata[mask]
                ydata = ydata[mask]
            display_ax.ax.scatter(xdata, ydata, s=marker_size,
                                  c=marker_color_, **kwargs)

    def annotate(self, left_right=True, positions=True, scalebar=False,
                 size=12, scale_size=5.0, scale_units='cm', scale_loc=4,
                 decimals=0, **kwargs):
        """Add annotations to the plot.

        Parameters
        ----------
        left_right : boolean, optional
            If left_right is True, annotations indicating which side
            is left and which side is right are drawn. Default=True.

        positions : boolean, optional
            If positions is True, annotations indicating the
            positions of the cuts are drawn. Default=True.

        scalebar : boolean, optional
            If ``True``, cuts are annotated with a reference scale bar.
            For finer control of the scale bar, please check out
            the draw_scale_bar method on the axes in "axes" attribute of
            this object. Default=False.

        size : integer, optional
            The size of the text used. Default=12.

        scale_size : number, optional
            The length of the scalebar, in units of scale_units.
            Default=5.0.

        scale_units : {'cm', 'mm'}, optional
            The units for the scalebar. Default='cm'.

        scale_loc : integer, optional
            The positioning for the scalebar. Default=4.
            Valid location codes are:

            - 'upper right'  : 1
            - 'upper left'   : 2
            - 'lower left'   : 3
            - 'lower right'  : 4
            - 'right'        : 5
            - 'center left'  : 6
            - 'center right' : 7
            - 'lower center' : 8
            - 'upper center' : 9
            - 'center'       : 10

        decimals : integer, optional
            Number of decimal places on slice position annotation. If zero,
            the slice position is integer without decimal point.
            Default=0.

        kwargs :
            Extra keyword arguments are passed to matplotlib's text
            function.

        """
        kwargs = kwargs.copy()
        if 'color' not in kwargs:
            if self._black_bg:
                kwargs['color'] = 'w'
            else:
                kwargs['color'] = 'k'

        bg_color = ('k' if self._black_bg else 'w')

        if left_right:
            for display_axis in self.axes.values():
                display_axis.draw_left_right(size=size, bg_color=bg_color,
                                             **kwargs)

        if positions:
            for display_axis in self.axes.values():
                display_axis.draw_position(size=size, bg_color=bg_color,
                                           decimals=decimals,
                                           **kwargs)

        if scalebar:
            axes = self.axes.values()
            for display_axis in axes:
                display_axis.draw_scale_bar(bg_color=bg_color,
                                            fontsize=size,
                                            size=scale_size,
                                            units=scale_units,
                                            loc=scale_loc,
                                            **kwargs)

    def close(self):
        """ Close the figure. This is necessary to avoid leaking memory.
        """
        plt.close(self.frame_axes.figure.number)

    def savefig(self, filename, dpi=None):
        """ Save the figure to a file

        Parameters
        ----------
        filename : string
            The file name to save to. Its extension determines the
            file type, typically '.png', '.svg' or '.pdf'.

        dpi : None or scalar, optional
            The resolution in dots per inch.

        """
        facecolor = edgecolor = 'k' if self._black_bg else 'w'
        self.frame_axes.figure.savefig(filename, dpi=dpi,
                                       facecolor=facecolor,
                                       edgecolor=edgecolor)


###############################################################################
# class OrthoSlicer
###############################################################################

class OrthoSlicer(BaseSlicer):
    """ A class to create 3 linked axes for plotting orthogonal
    cuts of 3D maps.

    Attributes
    ----------
    axes : dictionary of axes
        The 3 axes used to plot each view.

    frame_axes : axes
        The axes framing the whole set of views.

    Notes
    -----
    The extent of the different axes are adjusted to fit the data
    best in the viewing area.

    """
    _cut_displayed = 'yxz'
    _axes_class = CutAxes

    @classmethod
    def find_cut_coords(cls, img=None, threshold=None, cut_coords=None):
        "Instantiate the slicer and find cut coordinates"
        if cut_coords is None:
            if img is None or img is False:
                cut_coords = (0, 0, 0)
            else:
                cut_coords = find_xyz_cut_coords(
                    img, activation_threshold=threshold)
            cut_coords = [cut_coords['xyz'.find(c)]
                          for c in sorted(cls._cut_displayed)]
        return cut_coords

    def _init_axes(self, **kwargs):
        cut_coords = self.cut_coords
        if len(cut_coords) != len(self._cut_displayed):
            raise ValueError('The number cut_coords passed does not'
                             ' match the display_mode')
        x0, y0, x1, y1 = self.rect
        facecolor = 'k' if self._black_bg else 'w'
        # Create our axes:
        self.axes = dict()
        for index, direction in enumerate(self._cut_displayed):
            fh = self.frame_axes.get_figure()
            ax = fh.add_axes([0.3 * index * (x1 - x0) + x0, y0,
                              .3 * (x1 - x0), y1 - y0], aspect='equal')
            if LooseVersion(matplotlib.__version__) >= LooseVersion("1.6"):
                ax.set_facecolor(facecolor)
            else:
                ax.set_axis_bgcolor(facecolor)

            ax.axis('off')
            coord = self.cut_coords[
                sorted(self._cut_displayed).index(direction)]
            display_ax = self._axes_class(ax, direction, coord, **kwargs)
            self.axes[direction] = display_ax
            ax.set_axes_locator(self._locator)

        if self._black_bg:
            for ax in self.axes.values():
                ax.ax.imshow(np.zeros((2, 2, 3)),
                             extent=[-5000, 5000, -5000, 5000],
                             zorder=-500, aspect='equal')

            # To have a black background in PDF, we need to create a
            # patch in black for the background
            self.frame_axes.imshow(np.zeros((2, 2, 3)),
                                   extent=[-5000, 5000, -5000, 5000],
                                   zorder=-500, aspect='auto')
            self.frame_axes.set_zorder(-1000)

    def _locator(self, axes, renderer):
        """ The locator function used by matplotlib to position axes.
        Here we put the logic used to adjust the size of the axes.

        """
        x0, y0, x1, y1 = self.rect
        width_dict = dict()
        # A dummy axes, for the situation in which we are not plotting
        # all three (x, y, z) cuts
        dummy_ax = self._axes_class(None, None, None)
        width_dict[dummy_ax.ax] = 0
        display_ax_dict = self.axes

        if self._colorbar:
            adjusted_width = self._colorbar_width / len(self.axes)
            right_margin = self._colorbar_margin['right'] / len(self.axes)
            ticks_margin = self._colorbar_margin['left'] / len(self.axes)
            x1 = x1 - (adjusted_width + ticks_margin + right_margin)

        for display_ax in display_ax_dict.values():
            bounds = display_ax.get_object_bounds()
            if not bounds:
                # This happens if the call to _map_show was not
                # successful. As it happens asynchronously (during a
                # refresh of the figure) we capture the problem and
                # ignore it: it only adds a non informative traceback
                bounds = [0, 1, 0, 1]
            xmin, xmax, ymin, ymax = bounds
            width_dict[display_ax.ax] = (xmax - xmin)

        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.items():
            width_dict[ax] = width / total_width * (x1 - x0)

        direction_ax = []
        for d in self._cut_displayed:
            direction_ax.append(display_ax_dict.get(d, dummy_ax).ax)
        left_dict = dict()
        for idx, ax in enumerate(direction_ax):
            left_dict[ax] = x0
            for prev_ax in direction_ax[:idx]:
                left_dict[ax] += width_dict[prev_ax]

        return transforms.Bbox([[left_dict[axes], y0],
                               [left_dict[axes] + width_dict[axes], y1]])

    def draw_cross(self, cut_coords=None, **kwargs):
        """ Draw a crossbar on the plot to show where the cut is
        performed.

        Parameters
        ----------
        cut_coords : 3-tuple of floats, optional
            The position of the cross to draw. If none is passed, the
            ortho_slicer's cut coordinates are used.

        kwargs :
            Extra keyword arguments are passed to axhline

        """
        if cut_coords is None:
            cut_coords = self.cut_coords
        coords = dict()
        for direction in 'xyz':
            coord = None
            if direction in self._cut_displayed:
                coord = cut_coords[
                    sorted(self._cut_displayed).index(direction)]
            coords[direction] = coord
        x, y, z = coords['x'], coords['y'], coords['z']

        kwargs = kwargs.copy()
        if 'color' not in kwargs:
            if self._black_bg:
                kwargs['color'] = '.8'
            else:
                kwargs['color'] = 'k'

        if 'y' in self.axes:
            ax = self.axes['y'].ax
            if x is not None:
                ax.axvline(x, ymin=.05, ymax=.95, **kwargs)
            if z is not None:
                ax.axhline(z, **kwargs)

        if 'x' in self.axes:
            ax = self.axes['x'].ax
            if y is not None:
                ax.axvline(y, ymin=.05, ymax=.95, **kwargs)
            if z is not None:
                ax.axhline(z, xmax=.95, **kwargs)

        if 'z' in self.axes:
            ax = self.axes['z'].ax
            if x is not None:
                ax.axvline(x, ymin=.05, ymax=.95, **kwargs)
            if y is not None:
                ax.axhline(y, **kwargs)


###############################################################################
# class TiledSlicer
###############################################################################

class TiledSlicer(BaseSlicer):
    """ A class to create 3 axes for plotting orthogonal
    cuts of 3D maps, organized in a 2x2 grid.

    Attributes
    ----------
    axes : dictionary of axes
        The 3 axes used to plot each view.

    frame_axes : axes
        The axes framing the whole set of views.

    Notes
    -----
    The extent of the different axes are adjusted to fit the data
    best in the viewing area.

    """
    _cut_displayed = 'yxz'
    _axes_class = CutAxes
    _default_figsize = [2.0, 6.0]

    @classmethod
    def find_cut_coords(cls, img=None, threshold=None, cut_coords=None):
        """Instantiate the slicer and find cut coordinates.

        Parameters
        ----------
        img : 3D Nifti1Image
            The brain map.

        threshold : float, optional
            The lower threshold to the positive activation. If None, the
            activation threshold is computed using the 80% percentile of
            the absolute value of the map.

        cut_coords : list of float, optional
            xyz world coordinates of cuts.

        Returns
        -------
        cut_coords : list of float
            xyz world coordinates of cuts.

        """
        if cut_coords is None:
            if img is None or img is False:
                cut_coords = (0, 0, 0)
            else:
                cut_coords = find_xyz_cut_coords(
                    img, activation_threshold=threshold)
            cut_coords = [cut_coords['xyz'.find(c)]
                          for c in sorted(cls._cut_displayed)]

        return cut_coords

    def _find_initial_axes_coord(self, index):
        """Find coordinates for initial axes placement for xyz cuts.

        Parameters
        ----------
        index : int
            Index corresponding to current cut 'x', 'y' or 'z'.

        Returns
        -------
        [coord1, coord2, coord3, coord4] : list of int
            x0, y0, x1, y1 coordinates used by matplotlib
            to position axes in figure.

        """
        rect_x0, rect_y0, rect_x1, rect_y1 = self.rect

        if index == 0:
                coord1 = rect_x1 - rect_x0
                coord2 = 0.5 * (rect_y1 - rect_y0) + rect_y0
                coord3 = 0.5 * (rect_x1 - rect_x0) + rect_x0
                coord4 = rect_y1 - rect_y0
        elif index == 1:
                coord1 = 0.5 * (rect_x1 - rect_x0) + rect_x0
                coord2 = 0.5 * (rect_y1 - rect_y0) + rect_y0
                coord3 = rect_x1 - rect_x0
                coord4 = rect_y1 - rect_y0
        elif index == 2:
                coord1 = rect_x1 - rect_x0
                coord2 = rect_y1 - rect_y0
                coord3 = 0.5 * (rect_x1 - rect_x0) + rect_x0
                coord4 = 0.5 * (rect_y1 - rect_y0) + rect_y0
        return [coord1, coord2, coord3, coord4]

    def _init_axes(self, **kwargs):
        """Initializes and places axes for display of 'xyz' cuts.

        Parameters
        ----------
        kwargs :
            additional arguments to pass to self._axes_class

        """
        cut_coords = self.cut_coords
        if len(cut_coords) != len(self._cut_displayed):
            raise ValueError('The number cut_coords passed does not'
                             ' match the display_mode')

        facecolor = 'k' if self._black_bg else 'w'

        self.axes = dict()
        for index, direction in enumerate(self._cut_displayed):
            fh = self.frame_axes.get_figure()
            axes_coords = self._find_initial_axes_coord(index)
            ax = fh.add_axes(axes_coords, aspect='equal')

            if LooseVersion(matplotlib.__version__) >= LooseVersion("1.6"):
                ax.set_facecolor(facecolor)
            else:
                ax.set_axis_bgcolor(facecolor)

            ax.axis('off')
            coord = self.cut_coords[
                sorted(self._cut_displayed).index(direction)]
            display_ax = self._axes_class(ax, direction, coord, **kwargs)
            self.axes[direction] = display_ax
            ax.set_axes_locator(self._locator)

    def _adjust_width_height(self, width_dict, height_dict,
                             rect_x0, rect_y0, rect_x1, rect_y1):
        """Adjusts absolute image width and height to ratios.

        Parameters
        ----------
        width_dict : dict
            Width of image cuts displayed in axes.

        height_dict : dict
            Height of image cuts displayed in axes.

        rect_x0, rect_y0, rect_x1, rect_y1 : float
            Matplotlib figure boundaries.

        Returns
        -------
        width_dict : dict
            Width ratios of image cuts for optimal positioning of axes.

        height_dict : dict
            Height ratios of image cuts for optimal positioning of axes.

        """
        total_height = 0
        total_width = 0

        if 'y' in self.axes:
            ax = self.axes['y'].ax
            total_height = total_height + height_dict[ax]
            total_width = total_width + width_dict[ax]

        if 'x' in self.axes:
            ax = self.axes['x'].ax
            total_width = total_width + width_dict[ax]

        if 'z' in self.axes:
            ax = self.axes['z'].ax
            total_height = total_height + height_dict[ax]

        for ax, width in width_dict.items():
            width_dict[ax] = width / total_width * (rect_x1 - rect_x0)

        for ax, height in height_dict.items():
            height_dict[ax] = height / total_height * (rect_y1 - rect_y0)

        return (width_dict, height_dict)

    def _find_axes_coord(self, rel_width_dict, rel_height_dict,
                         rect_x0, rect_y0, rect_x1, rect_y1):
        """"Find coordinates for initial axes placement for xyz cuts.

        Parameters
        ----------
        rel_width_dict : dict
            Width ratios of image cuts for optimal positioning of axes.

        rel_height_dict : dict
            Height ratios of image cuts for optimal positioning of axes.

        rect_x0, rect_y0, rect_x1, rect_y1 : float
            Matplotlib figure boundaries.

        Returns
        -------
        coord1, coord2, coord3, coord4 : dict
            x0, y0, x1, y1 coordinates per axes used by matplotlib
            to position axes in figure.

        """
        coord1 = dict()
        coord2 = dict()
        coord3 = dict()
        coord4 = dict()

        if 'y' in self.axes:
            ax = self.axes['y'].ax
            coord1[ax] = rect_x0
            coord2[ax] = (rect_y1) - rel_height_dict[ax]
            coord3[ax] = rect_x0 + rel_width_dict[ax]
            coord4[ax] = rect_y1

        if 'x' in self.axes:
            ax = self.axes['x'].ax
            coord1[ax] = (rect_x1) - rel_width_dict[ax]
            coord2[ax] = (rect_y1) - rel_height_dict[ax]
            coord3[ax] = rect_x1
            coord4[ax] = rect_y1

        if 'z' in self.axes:
            ax = self.axes['z'].ax
            coord1[ax] = rect_x0
            coord2[ax] = rect_y0
            coord3[ax] = rect_x0 + rel_width_dict[ax]
            coord4[ax] = rect_y0 + rel_height_dict[ax]

        return(coord1, coord2, coord3, coord4)

    def _locator(self, axes, renderer):
        """ The locator function used by matplotlib to position axes.
        Here we put the logic used to adjust the size of the axes.

        """
        rect_x0, rect_y0, rect_x1, rect_y1 = self.rect

        # image width and height
        width_dict = dict()
        height_dict = dict()

        # A dummy axes, for the situation in which we are not plotting
        # all three (x, y, z) cuts
        dummy_ax = self._axes_class(None, None, None)
        width_dict[dummy_ax.ax] = 0
        height_dict[dummy_ax.ax] = 0
        display_ax_dict = self.axes

        if self._colorbar:
            adjusted_width = self._colorbar_width / len(self.axes)
            right_margin = self._colorbar_margin['right'] / len(self.axes)
            ticks_margin = self._colorbar_margin['left'] / len(self.axes)
            rect_x1 = rect_x1 - (adjusted_width + ticks_margin + right_margin)

        for display_ax in display_ax_dict.values():
            bounds = display_ax.get_object_bounds()
            if not bounds:
                # This happens if the call to _map_show was not
                # successful. As it happens asynchronously (during a
                # refresh of the figure) we capture the problem and
                # ignore it: it only adds a non informative traceback
                bounds = [0, 1, 0, 1]
            xmin, xmax, ymin, ymax = bounds
            width_dict[display_ax.ax] = (xmax - xmin)
            height_dict[display_ax.ax] = (ymax - ymin)

        # relative image height and width
        rel_width_dict, rel_height_dict = self._adjust_width_height(
                width_dict, height_dict,
                rect_x0, rect_y0, rect_x1, rect_y1)

        direction_ax = []
        for d in self._cut_displayed:
            direction_ax.append(display_ax_dict.get(d, dummy_ax).ax)

        coord1, coord2, coord3, coord4 = self._find_axes_coord(
                rel_width_dict, rel_height_dict,
                rect_x0, rect_y0, rect_x1, rect_y1)

        return transforms.Bbox([[coord1[axes], coord2[axes]],
                               [coord3[axes], coord4[axes]]])

    def draw_cross(self, cut_coords=None, **kwargs):
        """Draw a crossbar on the plot to show where the cut is performed.

        Parameters
        ----------
        cut_coords : 3-tuple of floats, optional
            The position of the cross to draw. If none is passed, the
            ortho_slicer's cut coordinates are used.

        kwargs :
            Extra keyword arguments are passed to axhline

        """
        if cut_coords is None:
            cut_coords = self.cut_coords
        coords = dict()
        for direction in 'xyz':
            coord_ = None
            if direction in self._cut_displayed:
                sorted_cuts = sorted(self._cut_displayed)
                index = sorted_cuts.index(direction)
                coord_ = cut_coords[index]
            coords[direction] = coord_
        x, y, z = coords['x'], coords['y'], coords['z']

        kwargs = kwargs.copy()
        if 'color' not in kwargs:
            try:
                kwargs['color'] = '.8' if self._black_bg else 'k'
            except KeyError:
                pass

        if 'y' in self.axes:
            ax = self.axes['y'].ax
            if x is not None:
                ax.axvline(x, **kwargs)
            if z is not None:
                ax.axhline(z, **kwargs)

        if 'x' in self.axes:
            ax = self.axes['x'].ax
            if y is not None:
                ax.axvline(y, **kwargs)
            if z is not None:
                ax.axhline(z, **kwargs)

        if 'z' in self.axes:
            ax = self.axes['z'].ax
            if x is not None:
                ax.axvline(x, **kwargs)
            if y is not None:
                ax.axhline(y, **kwargs)

###############################################################################
# class BaseStackedSlicer
###############################################################################

class BaseStackedSlicer(BaseSlicer):
    """ A class to create linked axes for plotting stacked
    cuts of 2D maps.

    Attributes
    ----------
    axes : dictionary of axes
        The axes used to plot each view.

    frame_axes : axes
        The axes framing the whole set of views.

    Notes
    -----
    The extent of the different axes are adjusted to fit the data
    best in the viewing area.

    """
    @classmethod
    def find_cut_coords(cls, img=None, threshold=None, cut_coords=None):
        "Instantiate the slicer and find cut coordinates"
        if cut_coords is None:
            cut_coords = 7

        if img is None or img is False:
            bounds = ((-40, 40), (-30, 30), (-30, 75))
            lower, upper = bounds['xyz'.index(cls._direction)]
            cut_coords = np.linspace(lower, upper, cut_coords).tolist()
        else:
            if (not isinstance(cut_coords, collections.abc.Sequence) and
                    isinstance(cut_coords, numbers.Number)):
                cut_coords = find_cut_slices(img,
                                             direction=cls._direction,
                                             n_cuts=cut_coords)

        return cut_coords

    def _init_axes(self, **kwargs):
        x0, y0, x1, y1 = self.rect
        # Create our axes:
        self.axes = dict()
        fraction = 1. / len(self.cut_coords)
        for index, coord in enumerate(self.cut_coords):
            coord = float(coord)
            fh = self.frame_axes.get_figure()
            ax = fh.add_axes([fraction * index * (x1 - x0) + x0, y0,
                              fraction * (x1 - x0), y1 - y0])
            ax.axis('off')
            display_ax = self._axes_class(ax, self._direction,
                                          coord, **kwargs)
            self.axes[coord] = display_ax
            ax.set_axes_locator(self._locator)

        if self._black_bg:
            for ax in self.axes.values():
                ax.ax.imshow(np.zeros((2, 2, 3)),
                             extent=[-5000, 5000, -5000, 5000],
                             zorder=-500, aspect='equal')

            # To have a black background in PDF, we need to create a
            # patch in black for the background
            self.frame_axes.imshow(np.zeros((2, 2, 3)),
                                   extent=[-5000, 5000, -5000, 5000],
                                   zorder=-500, aspect='auto')
            self.frame_axes.set_zorder(-1000)

    def _locator(self, axes, renderer):
        """ The locator function used by matplotlib to position axes.
        Here we put the logic used to adjust the size of the axes.

        """
        x0, y0, x1, y1 = self.rect
        width_dict = dict()
        display_ax_dict = self.axes

        if self._colorbar:
            adjusted_width = self._colorbar_width / len(self.axes)
            right_margin = self._colorbar_margin['right'] / len(self.axes)
            ticks_margin = self._colorbar_margin['left'] / len(self.axes)
            x1 = x1 - (adjusted_width + right_margin + ticks_margin)

        for display_ax in display_ax_dict.values():
            bounds = display_ax.get_object_bounds()
            if not bounds:
                # This happens if the call to _map_show was not
                # successful. As it happens asynchronously (during a
                # refresh of the figure) we capture the problem and
                # ignore it: it only adds a non informative traceback
                bounds = [0, 1, 0, 1]
            xmin, xmax, ymin, ymax = bounds
            width_dict[display_ax.ax] = (xmax - xmin)
        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.items():
            width_dict[ax] = width / total_width * (x1 - x0)
        left_dict = dict()
        left = float(x0)
        for coord, display_ax in display_ax_dict.items():
            left_dict[display_ax.ax] = left
            this_width = width_dict[display_ax.ax]
            left += this_width
        return transforms.Bbox([[left_dict[axes], y0],
                                [left_dict[axes] + width_dict[axes], y1]])

    def draw_cross(self, cut_coords=None, **kwargs):
        """ Draw a crossbar on the plot to show where the cut is
        performed.

        Parameters
        ----------
        cut_coords : 3-tuple of floats, optional
            The position of the cross to draw. If none is passed, the
            ortho_slicer's cut coordinates are used.

        kwargs :
            Extra keyword arguments are passed to axhline

        """
        return


class XSlicer(BaseStackedSlicer):
    _direction = 'x'
    _default_figsize = [2.6, 2.3]


class YSlicer(BaseStackedSlicer):
    _direction = 'y'
    _default_figsize = [2.2, 2.3]


class ZSlicer(BaseStackedSlicer):
    _direction = 'z'
    _default_figsize = [2.2, 2.3]


class XZSlicer(OrthoSlicer):
    _cut_displayed = 'xz'


class YXSlicer(OrthoSlicer):
    _cut_displayed = 'yx'


class YZSlicer(OrthoSlicer):
    _cut_displayed = 'yz'


class MosaicSlicer(BaseSlicer):
    """ A class to create 3 axes for plotting cuts of 3D maps,
    in multiple rows and columns.

    Attributes
    ----------
    axes : dictionary of axes
        The 3 axes used to plot multiple views.

    frame_axes : axes
        The axes framing the whole set of views.

    """
    _cut_displayed = 'yxz'
    _axes_class = CutAxes
    _default_figsize = [11.1, 7.2]

    @classmethod
    def find_cut_coords(cls, img=None, threshold=None, cut_coords=None):
        """Instantiate the slicer and find cut coordinates for mosaic plotting.

        Parameters
        ----------
        img : 3D Nifti1Image, optional
            The brain image.

        threshold : float, optional
            The lower threshold to the positive activation. If None, the
            activation threshold is computed using the 80% percentile of
            the absolute value of the map.

        cut_coords : list/tuple of 3 floats, integer, optional
            xyz world coordinates of cuts. If cut_coords
            are not provided, 7 coordinates of cuts are automatically
            calculated.

        Returns
        -------
        cut_coords : dict
            xyz world coordinates of cuts in a direction. Each key
            denotes the direction.
        """
        if cut_coords is None:
            cut_coords = 7

        if (not isinstance(cut_coords, collections.abc.Sequence) and
                isinstance(cut_coords, numbers.Number)):
            cut_coords = [cut_coords] * 3
            cut_coords = cls._find_cut_coords(img, cut_coords,
                                              cls._cut_displayed)
        else:
            if len(cut_coords) != len(cls._cut_displayed):
                raise ValueError('The number cut_coords passed does not'
                                 ' match the display_mode. Mosaic plotting '
                                 'expects tuple of length 3.' )
            cut_coords = [cut_coords['xyz'.find(c)]
                          for c in sorted(cls._cut_displayed)]
            cut_coords = cls._find_cut_coords(img, cut_coords,
                                              cls._cut_displayed)
        return cut_coords

    @staticmethod
    def _find_cut_coords(img, cut_coords, cut_displayed):
        """ Find slicing positions along a given axis.

            Helper function to find_cut_coords.

        Parameters
        ----------
        img : 3D Nifti1Image
            The brain image.

        cut_coords : list/tuple of 3 floats, integer, optional
            xyz world coordinates of cuts.

        cut_displayed : str
            Sectional directions 'yxz'

        Returns
        -------
        cut_coords : 1D array of length specified in n_cuts
            The computed cut_coords.
        """
        coords = dict()
        if img is None or img is False:
            bounds = ((-40, 40), (-30, 30), (-30, 75))
            for direction, n_cuts in zip(sorted(cut_displayed),
                                         cut_coords):
                lower, upper = bounds['xyz'.index(direction)]
                coords[direction] = np.linspace(lower, upper,
                                                n_cuts).tolist()
        else:
            for direction, n_cuts in zip(sorted(cut_displayed),
                                         cut_coords):
                coords[direction] = find_cut_slices(img, direction=direction,
                                                    n_cuts=n_cuts)
        return coords

    def _init_axes(self, **kwargs):
        """Initializes and places axes for display of 'xyz' multiple cuts.

        Parameters
        ----------
        kwargs:
            additional arguments to pass to self._axes_class

        """
        if not isinstance(self.cut_coords, dict):
            self.cut_coords = self.find_cut_coords(cut_coords=self.cut_coords)

        if len(self.cut_coords) != len(self._cut_displayed):
            raise ValueError('The number cut_coords passed does not'
                             ' match the mosaic mode')
        x0, y0, x1, y1 = self.rect

        # Create our axes:
        self.axes = dict()
        # portions for main axes
        fraction = y1 / len(self.cut_coords)
        height = fraction
        for index, direction in enumerate(self._cut_displayed):
            coords = self.cut_coords[direction]
            # portions allotment for each of 'x', 'y', 'z' coordinate
            fraction_c = 1. / len(coords)
            fh = self.frame_axes.get_figure()
            indices = [x0, fraction * index * (y1 - y0) + y0,
                       x1, fraction * (y1 - y0)]
            ax = fh.add_axes(indices)
            ax.axis('off')
            this_x0, this_y0, this_x1, this_y1 = indices
            for index_c, coord in enumerate(coords):
                coord = float(coord)
                fh_c = ax.get_figure()
                # indices for each sub axes within main axes
                indices = [fraction_c * index_c * (this_x1 - this_x0) + this_x0,
                           this_y0,
                           fraction_c * (this_x1 - this_x0),
                           height]
                ax = fh_c.add_axes(indices)
                ax.axis('off')
                display_ax = self._axes_class(ax, direction,
                                              coord, **kwargs)
                self.axes[(direction, coord)] = display_ax
                ax.set_axes_locator(self._locator)

    def _locator(self, axes, renderer):
        """ The locator function used by matplotlib to position axes.
            Here we put the logic used to adjust the size of the axes.
        """
        x0, y0, x1, y1 = self.rect
        display_ax_dict = self.axes

        if self._colorbar:
            adjusted_width = self._colorbar_width / len(self.axes)
            right_margin = self._colorbar_margin['right'] / len(self.axes)
            ticks_margin = self._colorbar_margin['left'] / len(self.axes)
            x1 = x1 - (adjusted_width + right_margin + ticks_margin)

        # capture widths for each axes for anchoring Bbox
        width_dict = dict()
        for direction in self._cut_displayed:
            this_width = dict()
            for display_ax in display_ax_dict.values():
                if direction == display_ax.direction:
                    bounds = display_ax.get_object_bounds()
                    if not bounds:
                        # This happens if the call to _map_show was not
                        # successful. As it happens asynchronously (during a
                        # refresh of the figure) we capture the problem and
                        # ignore it: it only adds a non informative traceback
                        bounds = [0, 1, 0, 1]
                    xmin, xmax, ymin, ymax = bounds
                    this_width[display_ax.ax] = (xmax - xmin)
            total_width = float(sum(this_width.values()))
            for ax, w in this_width.items():
                width_dict[ax] = w / total_width * (x1 - x0)

        left_dict = dict()
        # bottom positions in Bbox according to cuts
        bottom_dict = dict()
        # fraction is divided by the cut directions 'y', 'x', 'z'
        fraction = y1 / len(self._cut_displayed)
        height_dict = dict()
        for index, direction in enumerate(self._cut_displayed):
            left = float(x0)
            this_height = fraction + fraction * index
            for coord, display_ax in display_ax_dict.items():
                if direction == display_ax.direction:
                    left_dict[display_ax.ax] = left
                    this_width = width_dict[display_ax.ax]
                    left += this_width
                    bottom_dict[display_ax.ax] = fraction * index * (y1 - y0)
                    height_dict[display_ax.ax] = this_height
        return transforms.Bbox([[left_dict[axes], bottom_dict[axes]],
                                [left_dict[axes] + width_dict[axes],
                                 height_dict[axes]]])


    def draw_cross(self, cut_coords=None, **kwargs):
        """ Draw a crossbar on the plot to show where the cut is
        performed.

        Parameters
        ----------
        cut_coords: 3-tuple of floats, optional
            The position of the cross to draw. If none is passed, the
            ortho_slicer's cut coordinates are used.
        kwargs:
            Extra keyword arguments are passed to axhline
        """
        return


SLICERS = dict(ortho=OrthoSlicer,
               tiled=TiledSlicer,
               mosaic=MosaicSlicer,
               xz=XZSlicer,
               yz=YZSlicer,
               yx=YXSlicer,
               x=XSlicer,
               y=YSlicer,
               z=ZSlicer)


class OrthoProjector(OrthoSlicer):
    """A class to create linked axes for plotting orthogonal projections
    of 3D maps.

    """
    _axes_class = GlassBrainAxes

    @classmethod
    def find_cut_coords(cls, img=None, threshold=None, cut_coords=None):
        return (None, ) * len(cls._cut_displayed)

    def draw_cross(self, cut_coords=None, **kwargs):
        # It does not make sense to draw crosses for the position of
        # the cuts since we are taking the max along one axis
        pass

    def add_graph(self, adjacency_matrix, node_coords,
                  node_color='auto', node_size=50,
                  edge_cmap=cm.bwr,
                  edge_vmin=None, edge_vmax=None,
                  edge_threshold=None,
                  edge_kwargs=None, node_kwargs=None, colorbar=False,
                  ):
        """Plot undirected graph on each of the axes

        Parameters
        ----------
        adjacency_matrix : numpy array of shape (n, n)
            Represents the edges strengths of the graph.
            The matrix can be symmetric which will result in
            an undirected graph, or not symmetric which will
            result in a directed graph.

        node_coords : numpy array_like of shape (n, 3)
            3d coordinates of the graph nodes in world space.

        node_color : color or sequence of colors, optional
            Color(s) of the nodes. Default='auto'.

        node_size : scalar or array_like, optional
            Size(s) of the nodes in points^2. Default=50.

        edge_cmap : colormap, optional
            Colormap used for representing the strength of the edges.
            Default=cm.bwr.

        edge_vmin, edge_vmax : float, optional
            If not None, either or both of these values will be used to
            as the minimum and maximum values to color edges. If None are
            supplied the maximum absolute value within the given threshold
            will be used as minimum (multiplied by -1) and maximum
            coloring levels.

        edge_threshold : str or number, optional
            If it is a number only the edges with a value greater than
            edge_threshold will be shown.
            If it is a string it must finish with a percent sign,
            e.g. "25.3%", and only the edges with a abs(value) above
            the given percentile will be shown.

        edge_kwargs : dict, optional
            Will be passed as kwargs for each edge matlotlib Line2D.

        node_kwargs : dict
            Will be passed as kwargs to the plt.scatter call that plots all
            the nodes in one go.

        """
        # set defaults
        if edge_kwargs is None:
            edge_kwargs = {}
        if node_kwargs is None:
            node_kwargs = {}
        if isinstance(node_color, str) and node_color == 'auto':
            nb_nodes = len(node_coords)
            node_color = mpl_cm.Set2(np.linspace(0, 1, nb_nodes))
        node_coords = np.asarray(node_coords)

        # decompress input matrix if sparse
        if sparse.issparse(adjacency_matrix):
            adjacency_matrix = adjacency_matrix.toarray()

        # make the lines below well-behaved
        adjacency_matrix = np.nan_to_num(adjacency_matrix)

        # safety checks
        if 's' in node_kwargs:
            raise ValueError("Please use 'node_size' and not 'node_kwargs' "
                             "to specify node sizes")
        if 'c' in node_kwargs:
            raise ValueError("Please use 'node_color' and not 'node_kwargs' "
                             "to specify node colors")

        adjacency_matrix_shape = adjacency_matrix.shape
        if (len(adjacency_matrix_shape) != 2 or
                adjacency_matrix_shape[0] != adjacency_matrix_shape[1]):
            raise ValueError(
                "'adjacency_matrix' is supposed to have shape (n, n)."
                ' Its shape was {0}'.format(adjacency_matrix_shape))

        node_coords_shape = node_coords.shape
        if len(node_coords_shape) != 2 or node_coords_shape[1] != 3:
            message = (
                "Invalid shape for 'node_coords'. You passed an "
                "'adjacency_matrix' of shape {0} therefore "
                "'node_coords' should be a array with shape ({0[0]}, 3) "
                'while its shape was {1}').format(adjacency_matrix_shape,
                                                  node_coords_shape)

            raise ValueError(message)

        if isinstance(node_color, (list, np.ndarray)) and len(node_color) != 1:
            if len(node_color) != node_coords_shape[0]:
                raise ValueError(
                    "Mismatch between the number of nodes ({0}) "
                    "and and the number of node colors ({1})."
                    .format(node_coords_shape[0], len(node_color)))

        if node_coords_shape[0] != adjacency_matrix_shape[0]:
            raise ValueError(
                "Shape mismatch between 'adjacency_matrix' "
                "and 'node_coords'"
                "'adjacency_matrix' shape is {0}, 'node_coords' shape is {1}"
                .format(adjacency_matrix_shape, node_coords_shape))

        # If the adjacency matrix is not symmetric, give a warning
        symmetric = True
        if not np.allclose(adjacency_matrix, adjacency_matrix.T, rtol=1e-3):
            symmetric = False
            warnings.warn(("'adjacency_matrix' is not symmetric. "
                           "A directed graph will be plotted."))

        # For a masked array, masked values are replaced with zeros
        if hasattr(adjacency_matrix, 'mask'):
            if not (adjacency_matrix.mask == adjacency_matrix.mask.T).all():
                symmetric = False
                warnings.warn(("'adjacency_matrix' was masked with "
                               "a non symmetric mask. A directed "
                               "graph will be plotted."))
            adjacency_matrix = adjacency_matrix.filled(0)

        if edge_threshold is not None:
            if symmetric:
                # Keep a percentile of edges with the highest absolute
                # values, so only need to look at the covariance
                # coefficients below the diagonal
                lower_diagonal_indices = np.tril_indices_from(adjacency_matrix,
                                                              k=-1)
                lower_diagonal_values = adjacency_matrix[
                    lower_diagonal_indices]
                edge_threshold = _utils.param_validation.check_threshold(
                    edge_threshold, np.abs(lower_diagonal_values),
                    stats.scoreatpercentile, 'edge_threshold')
            else:
                edge_threshold = _utils.param_validation.check_threshold(
                    edge_threshold, np.abs(adjacency_matrix.ravel()),
                    stats.scoreatpercentile, 'edge_threshold')

            adjacency_matrix = adjacency_matrix.copy()
            threshold_mask = np.abs(adjacency_matrix) < edge_threshold
            adjacency_matrix[threshold_mask] = 0

        if symmetric:
            lower_triangular_adjacency_matrix = np.tril(adjacency_matrix, k=-1)
            non_zero_indices = lower_triangular_adjacency_matrix.nonzero()
        else:
            non_zero_indices = adjacency_matrix.nonzero()

        line_coords = [node_coords[list(index)]
                       for index in zip(*non_zero_indices)]

        adjacency_matrix_values = adjacency_matrix[non_zero_indices]
        for ax in self.axes.values():
            ax._add_markers(node_coords, node_color, node_size, **node_kwargs)
            if line_coords:
                ax._add_lines(line_coords, adjacency_matrix_values, edge_cmap,
                              vmin=edge_vmin, vmax=edge_vmax, directed=(not symmetric),
                              **edge_kwargs)
            # To obtain the brain left view, we simply invert the x axis
            if ax.direction == 'l' and not (ax.ax.get_xlim()[0] > ax.ax.get_xlim()[1]):
                ax.ax.invert_xaxis()

        if colorbar:
            self._colorbar = colorbar
            self._show_colorbar(ax.cmap, ax.norm, threshold=edge_threshold)

        plt.draw_if_interactive()


class XProjector(OrthoProjector):
    _cut_displayed = 'x'
    _default_figsize = [2.6, 2.3]


class YProjector(OrthoProjector):
    _cut_displayed = 'y'
    _default_figsize = [2.2, 2.3]


class ZProjector(OrthoProjector):
    _cut_displayed = 'z'
    _default_figsize = [2.2, 2.3]


class XZProjector(OrthoProjector):
    _cut_displayed = 'xz'


class YXProjector(OrthoProjector):
    _cut_displayed = 'yx'


class YZProjector(OrthoProjector):
    _cut_displayed = 'yz'


class LYRZProjector(OrthoProjector):
    _cut_displayed = 'lyrz'


class LZRYProjector(OrthoProjector):
    _cut_displayed = 'lzry'


class LZRProjector(OrthoProjector):
    _cut_displayed = 'lzr'


class LYRProjector(OrthoProjector):
    _cut_displayed = 'lyr'


class LRProjector(OrthoProjector):
    _cut_displayed = 'lr'


class LProjector(OrthoProjector):
    _cut_displayed = 'l'
    _default_figsize = [2.6, 2.3]


class RProjector(OrthoProjector):
    _cut_displayed = 'r'
    _default_figsize = [2.6, 2.3]


PROJECTORS = dict(ortho=OrthoProjector,
                  xz=XZProjector,
                  yz=YZProjector,
                  yx=YXProjector,
                  x=XProjector,
                  y=YProjector,
                  z=ZProjector,
                  lzry=LZRYProjector,
                  lyrz=LYRZProjector,
                  lyr=LYRProjector,
                  lzr=LZRProjector,
                  lr=LRProjector,
                  l=LProjector,
                  r=RProjector)


def get_create_display_fun(display_mode, class_dict):
    try:
        return class_dict[display_mode].init_with_figure
    except KeyError:
        message = ('{0} is not a valid display_mode. '
                   'Valid options are {1}').format(
                        display_mode, sorted(class_dict.keys()))
        raise ValueError(message)


def get_slicer(display_mode):
    "Internal function to retrieve a slicer"
    return get_create_display_fun(display_mode, SLICERS)


def get_projector(display_mode):
    "Internal function to retrieve a projector"
    return get_create_display_fun(display_mode, PROJECTORS)
