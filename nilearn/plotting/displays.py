"""
The Slicer classes.

The main purpose of these classes is to have auto adjust of axes size to
the data with different layout of cuts.
"""

import operator

import numpy as np

import nibabel
from .._utils.testing import skip_if_running_nose
from .. import _utils

try:
    import pylab as pl
    from matplotlib import transforms, colors
    from matplotlib.colorbar import ColorbarBase
except ImportError:
    skip_if_running_nose('Could not import matplotlib')


# Local imports
from .find_cuts import find_xyz_cut_coords, find_cut_slices
from .edge_detect import _edge_map
from ..image.resampling import get_bounds, reorder_img, coord_transform,\
            get_mask_bounds

from . import glass_brain

################################################################################
# class BaseAxes
################################################################################

class BaseAxes(object):
    """ An MPL axis-like object that displays a 2D view of 3D volumes
    """

    def __init__(self, ax, direction, coord):
        """ An MPL axis-like object that displays a cut of 3D volumes

            Parameters
            ==========
            ax: a MPL axes instance
                The axes in which the plots will be drawn
            direction: {'x', 'y', 'z'}
                The directions of the view
            coord: float
                The coordinate along the direction of the cut

        """
        self.ax = ax
        self.direction = direction
        self.coord = coord
        self._object_bounds = list()

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
        # kwargs massaging
        kwargs['origin'] = 'upper'

        if self.direction == 'y':
            (xmin, xmax), (_, _), (zmin, zmax) = data_bounds
            (xmin_, xmax_), (_, _), (zmin_, zmax_) = bounding_box
        elif self.direction == 'x':
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
        if self.direction == 'x':
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

    def draw_position(self, size, bg_color, **kwargs):
        raise NotImplementedError("'draw_position' should be implemented "
                                  "in derived classes")

################################################################################
# class CutAxes
################################################################################

class CutAxes(BaseAxes):
    """ An MPL axis-like object that displays a cut of 3D volumes
    """
    def transform_to_2d(self, data, affine):
        """ Cut the 3D volume into a 2D slice

            Parameters
            ==========
            data: 3D ndarray
                The 3D volume to cut
            affine: 4x4 ndarray
                The affine of the volume
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

    def draw_position(self, size, bg_color, **kwargs):
        ax = self.ax
        ax.text(0, 0, '%s=%i' % (self.direction, self.coord),
                transform=ax.transAxes,
                horizontalalignment='left',
                verticalalignment='bottom',
                size=size,
                bbox=dict(boxstyle="square,pad=0",
                          ec=bg_color, fc=bg_color, alpha=1),
                **kwargs)


class GlassBrainAxes(BaseAxes):
    """An MPL axis-like object that displays a 2D projection of 3D
    volumes with a schematic view of the brain.

    """
    def __init__(self, ax, direction, coord, **kwargs):
        super(GlassBrainAxes, self).__init__(ax, direction, coord)
        if ax is not None:
            object_bounds = glass_brain.plot_brain_schematics(ax,
                                                              direction,
                                                              **kwargs)
            self.add_object_bounds(object_bounds)

    def transform_to_2d(self, data, affine):
        """ Returns the maximum of the absolute value of the 3D volume
            along an axis.

            Parameters
            ==========
            data: 3D ndarray
                The 3D volume
            affine: 4x4 ndarray
                The affine of the volume

        """
        max_axis = 'xyz'.index(self.direction)
        maximum_intensity_data = np.abs(data).max(axis=max_axis)
        return np.rot90(maximum_intensity_data)

    def draw_position(self, size, bg_color, **kwargs):
        # It does not make sense to draw crosses for the position of
        # the cuts since we are taking the max along one axis
        pass


################################################################################
# class BaseSlicer
################################################################################

class BaseSlicer(object):
    """ The main purpose of these class is to have auto adjust of axes size
        to the data with different layout of cuts.
    """
    # This actually encodes the figsize for only one axe
    _default_figsize = [2.2, 2.6]
    _colorbar = False
    # pseudo absolute value
    _colorbar_width = 0.06
    _colorbar_labels_margin = 2.8
    _axes_class = CutAxes

    def __init__(self, cut_coords, axes=None, black_bg=False, **kwargs):
        """ Create 3 linked axes for plotting orthogonal cuts.

            Parameters
            ----------
            cut_coords: 3 tuple of ints
                The cut position, in world space.
            axes: matplotlib axes object, optional
                The axes that will be subdivided in 3.
            black_bg: boolean, optional
                If True, the background of the figure will be put to
                black. If you wish to save figures with a black background,
                you will need to pass "facecolor='k', edgecolor='k'" to 
                pylab's savefig.

        """
        self.cut_coords = cut_coords
        if axes is None:
            axes = pl.axes((0., 0., 1., 1.))
            axes.axis('off')
        self.frame_axes = axes
        axes.set_zorder(1)
        bb = axes.get_position()
        self.rect = (bb.x0, bb.y0, bb.x1, bb.y1)
        self._black_bg = black_bg
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
                         **kwargs):
        # deal with "fake" 4D images
        if img is not None and img is not False:
            img = _utils.check_niimg(img, ensure_3d=True)

        cut_coords = cls.find_cut_coords(img, threshold, cut_coords)

        if isinstance(axes, pl.Axes) and figure is None:
            figure = axes.figure

        if not isinstance(figure, pl.Figure):
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
            figure = pl.figure(figure, figsize=figsize,
                            facecolor=facecolor)
        if isinstance(axes, pl.Axes):
            assert axes.figure is figure, ("The axes passed are not "
                    "in the figure")

        if axes is None:
            axes = [0., 0., 1., 1.]
            if leave_space:
                axes = [0.3, 0, .7, 1.]
        if operator.isSequenceType(axes):
            axes = figure.add_axes(axes)
        # People forget to turn their axis off, or to set the zorder, and
        # then they cannot see their slicer
        axes.axis('off')
        return cls(cut_coords, axes, black_bg, **kwargs)


    def title(self, text, x=0.01, y=0.99, size=15, color=None, bgcolor=None,
              alpha=1, **kwargs):
        """ Write a title to the view.

            Parameters
            ----------
            text: string
                The text of the title
            x: float, optional
                The horizontal position of the title on the frame in 
                fraction of the frame width.
            y: float, optional
                The vertical position of the title on the frame in 
                fraction of the frame height.
            size: integer, optional
                The size of the title text.
            color: matplotlib color specifier, optional
                The color of the font of the title.
            bgcolor: matplotlib color specifier, optional
                The color of the background of the title.
            alpha: float, optional
                The alpha value for the background.
            kwargs:
                Extra keyword arguments are passed to matplotlib's text
                function.
        """
        if color is None:
            color = 'k' if self._black_bg else 'w'
        if bgcolor is None:
            bgcolor = 'w' if self._black_bg else 'k'
        if hasattr(self, '_cut_displayed'):
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
            img: Niimg-like object
                See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
                If it is a masked array, only the non-masked part will be
                plotted.
            threshold : a number, None
                If None is given, the maps are not thresholded.
                If a number is given, it is used to threshold the maps:
                values below the threshold (in absolute value) are
                plotted as transparent.
            colorbar: boolean, optional
                If True, display a colorbar on the right of the plots.
            kwargs:
                Extra keyword arguments are passed to imshow.
        """
        if colorbar and self._colorbar:
            raise ValueError("This figure already has an overlay with a "
                             "colorbar.")
        else:
            self._colorbar = colorbar

        img = _utils.check_niimg(img, ensure_3d=True)

        if threshold is not None:
            data = img.get_data()
            if threshold == 0:
                data = np.ma.masked_equal(data, 0, copy=False)
            else:
                data = np.ma.masked_inside(data, -threshold, threshold,
                                           copy=False)
            img = nibabel.Nifti1Image(data, img.get_affine())

        # To make sure that add_overlay has a consistant default behavior
        # with plot_stat_map
        kwargs.setdefault('interpolation', 'nearest')
        ims = self._map_show(img, type='imshow', **kwargs)

        if colorbar:
            self._colorbar_show(ims[0], threshold)

        pl.draw_if_interactive()

    def add_contours(self, img, **kwargs):
        """ Contour a 3D map in all the views.

            Parameters
            -----------
            img: Niimg-like object
                See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
                Provides image to plot.
            kwargs:
                Extra keyword arguments are passed to contour, see the
                documentation of pylab.contour
                Useful, arguments are typical "levels", which is a
                list of values to use for plotting a contour, and
                "colors", which is one color or a list of colors for
                these contours.
        """
        self._map_show(img, type='contour', **kwargs)
        pl.draw_if_interactive()

    def _map_show(self, img, type='imshow', resampling_interpolation='continuous', **kwargs):
        img = reorder_img(img, resample=resampling_interpolation)

        affine = img.get_affine()
        data = img.get_data()
        data_bounds = get_bounds(data.shape, affine)
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = data_bounds

        xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = \
                                        xmin, xmax, ymin, ymax, zmin, zmax

        if hasattr(data, 'mask') and isinstance(data.mask, np.ndarray):
            not_mask = np.logical_not(data.mask)
            xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = \
                    get_mask_bounds(nibabel.Nifti1Image(not_mask.astype(np.int),
                                    affine))

        data_2d_list = []
        for display_ax in self.axes.itervalues():
            try:
                data_2d = display_ax.transform_to_2d(data, affine)
            except IndexError:
                # We are cutting outside the indices of the data
                data_2d = None

            data_2d_list.append(data_2d)

        if 'vmin' not in kwargs:
            kwargs['vmin'] = min(d.min() for d in data_2d_list
                                 if d is not None)
        if 'vmax' not in kwargs:
            kwargs['vmax'] = max(d.max() for d in data_2d_list
                                 if d is not None)

        bounding_box = (xmin_, xmax_), (ymin_, ymax_), (zmin_, zmax_)

        ims = []
        to_iterate_over = zip(self.axes.values(), data_2d_list)
        for display_ax, data_2d in to_iterate_over:
            if data_2d is not None:
                im = display_ax.draw_2d(data_2d, data_bounds, bounding_box,
                                        type=type, **kwargs)
                ims.append(im)
        return ims

    def _colorbar_show(self, im, threshold):
        if threshold is None:
            offset = 0
        else:
            offset = threshold
        if offset > im.norm.vmax:
            offset = im.norm.vmax

        # create new  axis for the colorbar
        x_adjusted_width = self._colorbar_width / len(self.axes)
        x_adjusted_right_margin = 0.01 / len(self.axes)
        figure = self.frame_axes.figure
        _, y0, x1, y1 = self.rect
        y_width = y1 - y0
        y_margin = 0.05 * y_width

        self._colorbar_ax = figure.add_axes([
            x1 - (x_adjusted_width + x_adjusted_right_margin),
            y0 + y_margin,
            x_adjusted_width - x_adjusted_right_margin,
            y_width - 2 * y_margin])

        our_cmap = im.cmap
        # edge case where the data has a single value
        # yields a cryptic matplotlib error message
        # when trying to plot the color bar
        nb_ticks = 5 if im.norm.vmin != im.norm.vmax else 1
        ticks = np.linspace(im.norm.vmin, im.norm.vmax, nb_ticks)
        bounds = np.linspace(im.norm.vmin, im.norm.vmax, our_cmap.N)

        # some colormap hacking
        cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
        istart = int(im.norm(-offset) * (our_cmap.N - 1))
        istop = int(im.norm(offset) * (our_cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.)  # just an average gray color
        our_cmap = our_cmap.from_list('Custom cmap', cmaplist, our_cmap.N)

        self._cbar = ColorbarBase(
            self._colorbar_ax, ticks=ticks, norm=im.norm,
            orientation='vertical', cmap=our_cmap, boundaries=bounds,
            spacing='proportional')

        self._colorbar_ax.yaxis.tick_left()
        self._colorbar_ax.set_yticklabels(["% 2.2g" % t for t in ticks])

        tick_color = 'w' if self._black_bg else 'k'
        for tick in self._colorbar_ax.yaxis.get_ticklabels():
            tick.set_color(tick_color)
        self._colorbar_ax.yaxis.set_tick_params(width=0)

    def add_edges(self, img, color='r'):
        """ Plot the edges of a 3D map in all the views.

            Parameters
            -----------
            map: 3D ndarray
                The 3D map to be plotted. If it is a masked array, only
                the non-masked part will be plotted.
            affine: 4x4 ndarray
                The affine matrix giving the transformation from voxel
                indices to world space.
            color: matplotlib color: string or (r, g, b) value
                The color used to display the edge map
        """
        img = reorder_img(img)
        data = img.get_data()
        affine = img.get_affine()
        single_color_cmap = colors.ListedColormap([color])
        data_bounds = get_bounds(data.shape, img.get_affine())

        # For each ax, cut the data and plot it
        for display_ax in self.axes.itervalues():
            try:
                data_2d = display_ax.transform_to_2d(data, affine)
                edge_mask = _edge_map(data_2d)
            except IndexError:
                # We are cutting outside the indices of the data
                continue
            display_ax.draw_2d(edge_mask, data_bounds, data_bounds,
                               type='imshow', cmap=single_color_cmap)

        pl.draw_if_interactive()

    def annotate(self, left_right=True, positions=True, size=12, **kwargs):
        """ Add annotations to the plot.

            Parameters
            ----------
            left_right: boolean, optional
                If left_right is True, annotations indicating which side
                is left and which side is right are drawn.
            positions: boolean, optional
                If positions is True, annotations indicating the
                positions of the cuts are drawn.
            size: integer, optional
                The size of the text used.
            kwargs:
                Extra keyword arguments are passed to matplotlib's text
                function.
        """
        kwargs = kwargs.copy()
        if not 'color' in kwargs:
            if self._black_bg:
                kwargs['color'] = 'w'
            else:
                kwargs['color'] = 'k'

        bg_color = ('k' if self._black_bg else 'w')
        if left_right:
            for display_ax in self.axes.values():
                display_ax.draw_left_right(size=size, bg_color=bg_color,
                                       **kwargs)

        if positions:
            for display_ax in self.axes.values():
                display_ax.draw_position(size=size, bg_color=bg_color,
                                       **kwargs)

    def close(self):
        """ Close the figure. This is necessary to avoid leaking memory.
        """
        pl.close(self.frame_axes.figure.number)

    def savefig(self, filename, dpi=None):
        """ Save the figure to a file

            Parameters
            ==========
            filename: string
                The file name to save to. It's extension determines the
                file type, typically '.png', '.svg' or '.pdf'.

            dpi: None or scalar
                The resolution in dots per inch.
        """
        facecolor = edgecolor = 'k' if self._black_bg else 'w'
        self.frame_axes.figure.savefig(filename, dpi=dpi,
                                       facecolor=facecolor,
                                       edgecolor=edgecolor)

################################################################################
# class OrthoSlicer
################################################################################

class OrthoSlicer(BaseSlicer):
    """ A class to create 3 linked axes for plotting orthogonal
        cuts of 3D maps.

        Attributes
        ----------

        axes: dictionnary of axes
            The 3 axes used to plot each view.
        frame_axes: axes
            The axes framing the whole set of views.

        Notes
        -----

        The extent of the different axes are adjusted to fit the data
        best in the viewing area.
    """
    _cut_displayed = 'yxz'
    _axes_class = CutAxes

    @classmethod
    def find_cut_coords(self, img=None, threshold=None, cut_coords=None):
        if cut_coords is None:
            if img is None or img is False:
                cut_coords = (0, 0, 0)
            else:
                cut_coords = find_xyz_cut_coords(img,
                                                 activation_threshold=threshold)
            cut_coords = [cut_coords['xyz'.find(c)]
                          for c in sorted(self._cut_displayed)]
        return cut_coords

    def _init_axes(self, **kwargs):
        cut_coords = self.cut_coords
        if len(cut_coords) != len(self._cut_displayed):
            raise ValueError('The number cut_coords passed does not'
                             'match the display_mode')
        x0, y0, x1, y1 = self.rect
        axisbg = 'k' if self._black_bg else 'w'
        # Create our axes:
        self.axes = dict()
        for index, direction in enumerate(self._cut_displayed):
            ax = pl.axes([0.3*index*(x1 - x0) + x0, y0, .3*(x1 - x0), y1 - y0],
                         axisbg=axisbg)
            ax.axis('off')
            coord = self.cut_coords[sorted(self._cut_displayed).index(direction)]
            display_ax = self._axes_class(ax, direction, coord, **kwargs)
            self.axes[direction] = display_ax
            ax.set_axes_locator(self._locator)

        if self._black_bg:
            for ax in self.axes.values():
                ax.ax.imshow(np.zeros((2, 2, 3)),
                            extent=[-5000, 5000, -5000, 5000],
                            zorder=-500, aspect='auto')

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
            ticks_margin = adjusted_width * self._colorbar_labels_margin
            x1 = x1 - (adjusted_width + ticks_margin)

        for display_ax in display_ax_dict.itervalues():
            bounds = display_ax.get_object_bounds()
            if not bounds:
                # This happens if the call to _map_show was not
                # succesful. As it happens asyncroniously (during a
                # refresh of the figure) we capture the problem and
                # ignore it: it only adds a non informative traceback
                bounds = [0, 1, 0, 1]
            xmin, xmax, ymin, ymax = bounds
            width_dict[display_ax.ax] = (xmax - xmin)
        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.iteritems():
            width_dict[ax] = width/total_width*(x1 -x0)
        x_ax = display_ax_dict.get('x', dummy_ax)
        y_ax = display_ax_dict.get('y', dummy_ax)
        z_ax = display_ax_dict.get('z', dummy_ax)
        left_dict = dict()
        left_dict[y_ax.ax] = x0
        left_dict[x_ax.ax] = x0 + width_dict[y_ax.ax]
        left_dict[z_ax.ax] = x0 + width_dict[x_ax.ax] + width_dict[y_ax.ax]
        return transforms.Bbox([[left_dict[axes], y0],
                          [left_dict[axes] + width_dict[axes], y1]])

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
        if cut_coords is None:
            cut_coords = self.cut_coords
        coords = dict()
        for direction in 'xyz':
            coord = None
            if direction in self._cut_displayed:
                coord = cut_coords[sorted(self._cut_displayed).index(direction)]
            coords[direction] = coord
        x, y, z = coords['x'], coords['y'], coords['z']

        kwargs = kwargs.copy()
        if not 'color' in kwargs:
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


################################################################################
# class BaseStackedSlicer
################################################################################

class BaseStackedSlicer(BaseSlicer):
    """ A class to create linked axes for plotting stacked
        cuts of 3D maps.

        Attributes
        ----------

        axes: dictionnary of axes
            The axes used to plot each view.
        frame_axes: axes
            The axes framing the whole set of views.

        Notes
        -----

        The extent of the different axes are adjusted to fit the data
        best in the viewing area.
    """
    @classmethod
    def find_cut_coords(cls, img=None, threshold=None, cut_coords=None):
        if cut_coords is None:
            cut_coords = 7

        if img is None or img is False:
            bounds = ((-40, 40), (-30, 30), (-30, 75))
            lower, upper = bounds['xyz'.index(cls._direction)]
            cut_coords = np.linspace(lower, upper, cut_coords).tolist()
        else:
            if (not operator.isSequenceType(cut_coords) and
                    operator.isNumberType(cut_coords)):
                cut_coords = find_cut_slices(img,
                                             direction=cls._direction,
                                             n_cuts=cut_coords)

        return cut_coords

    def _init_axes(self, **kwargs):
        x0, y0, x1, y1 = self.rect
        # Create our axes:
        self.axes = dict()
        fraction = 1./len(self.cut_coords)
        for index, coord in enumerate(self.cut_coords):
            coord = float(coord)
            ax = pl.axes([fraction*index*(x1-x0) + x0, y0,
                          fraction*(x1-x0), y1-y0])
            ax.axis('off')
            display_ax = self._axes_class(ax, self._direction,
                                         coord, **kwargs)
            self.axes[coord] = display_ax
            ax.set_axes_locator(self._locator)

        if self._black_bg:
            for ax in self.axes.values():
                ax.ax.imshow(np.zeros((2, 2, 3)),
                            extent=[-5000, 5000, -5000, 5000],
                            zorder=-500, aspect='auto')

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
            adjusted_width = self._colorbar_width/len(self.axes)
            ticks_margin = adjusted_width*self._colorbar_labels_margin
            x1 = x1 - (adjusted_width+ticks_margin)

        for display_ax in display_ax_dict.itervalues():
            bounds = display_ax.get_object_bounds()
            if not bounds:
                # This happens if the call to _map_show was not
                # succesful. As it happens asyncroniously (during a
                # refresh of the figure) we capture the problem and
                # ignore it: it only adds a non informative traceback
                bounds = [0, 1, 0, 1]
            xmin, xmax, ymin, ymax = bounds
            width_dict[display_ax.ax] = (xmax - xmin)
        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.iteritems():
            width_dict[ax] = width/total_width*(x1 -x0)
        left_dict = dict()
        left = float(x0)
        for coord, display_ax in sorted(display_ax_dict.items()):
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
            cut_coords: 3-tuple of floats, optional
                The position of the cross to draw. If none is passed, the
                ortho_slicer's cut coordinates are used.
            kwargs:
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


SLICERS = dict(ortho=OrthoSlicer,
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


PROJECTORS = dict(ortho=OrthoProjector,
                  xz=XZProjector,
                  yz=YZProjector,
                  yx=YXProjector,
                  x=XProjector,
                  y=YProjector,
                  z=ZProjector)


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
