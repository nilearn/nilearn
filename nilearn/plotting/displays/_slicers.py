import collections
import contextlib
import numbers
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.transforms import Bbox

from nilearn._utils import check_niimg_3d
from nilearn._utils.docs import fill_doc
from nilearn._utils.niimg import is_binary_niimg, safe_get_data
from nilearn.image import get_data, new_img_like, reorder_img
from nilearn.image.resampling import get_bounds, get_mask_bounds
from nilearn.plotting.displays import CutAxes
from nilearn.plotting.displays._axes import coords_3d_to_2d
from nilearn.plotting.edge_detect import edge_map
from nilearn.plotting.find_cuts import find_cut_slices, find_xyz_cut_coords


@fill_doc
class BaseSlicer:
    """BaseSlicer implementation which main purpose is to auto adjust \
    the axes size to the data with different layout of cuts.

    It creates 3 linked axes for plotting orthogonal cuts.

    Attributes
    ----------
    cut_coords : 3 :obj:`tuple` of :obj:`int`
        The cut position, in world space.

    frame_axes : :class:`matplotlib.axes.Axes`, optional
        The matplotlib axes that will be subdivided in 3.

    black_bg : :obj:`bool`, default=False
        If ``True``, the background of the figure will be put to
        black. If you wish to save figures with a black background,
        you will need to pass ``facecolor='k', edgecolor='k'``
        to :func:`~matplotlib.pyplot.savefig`.

    brain_color : :obj:`tuple`, default=(0.5, 0.5, 0.5)
        The brain color to use as the background color (e.g., for
        transparent colorbars).
    """

    # This actually encodes the figsize for only one axe
    _default_figsize: ClassVar[list[float, float]] = [2.2, 2.6]
    _axes_class = CutAxes

    def __init__(
        self,
        cut_coords,
        axes=None,
        black_bg=False,
        brain_color=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        self.cut_coords = cut_coords
        if axes is None:
            axes = plt.axes((0.0, 0.0, 1.0, 1.0))
            axes.axis("off")
        self.frame_axes = axes
        axes.set_zorder(1)
        bb = axes.get_position()
        self.rect = (bb.x0, bb.y0, bb.x1, bb.y1)
        self._black_bg = black_bg
        self._brain_color = brain_color
        self._colorbar = False
        self._colorbar_width = 0.05 * bb.width
        self._cbar_tick_format = "%.2g"
        self._colorbar_margin = {
            "left": 0.25 * bb.width,
            "right": 0.02 * bb.width,
            "top": 0.05 * bb.height,
            "bottom": 0.05 * bb.height,
        }
        self._init_axes(**kwargs)

    @property
    def brain_color(self):
        """Return brain color."""
        return self._brain_color

    @property
    def black_bg(self):
        """Return black background."""
        return self._black_bg

    @staticmethod
    def find_cut_coords(img=None, threshold=None, cut_coords=None):
        """Act as placeholder and is not implemented in the base class \
        and has to be implemented in derived classes.
        """
        # Implement this as a staticmethod or a classmethod when
        # subclassing
        raise NotImplementedError

    @classmethod
    @fill_doc  # the fill_doc decorator must be last applied
    def init_with_figure(
        cls,
        img,
        threshold=None,
        cut_coords=None,
        figure=None,
        axes=None,
        black_bg=False,
        leave_space=False,
        colorbar=False,
        brain_color=(0.5, 0.5, 0.5),
        **kwargs,
    ):
        """Initialize the slicer with an image.

        Parameters
        ----------
        %(img)s
        cut_coords : 3 :obj:`tuple` of :obj:`int`
            The cut position, in world space.

        axes : :class:`matplotlib.axes.Axes`, optional
            The axes that will be subdivided in 3.

        black_bg : :obj:`bool`, default=False
            If ``True``, the background of the figure will be put to
            black. If you wish to save figures with a black background,
            you will need to pass ``facecolor='k', edgecolor='k'``
            to :func:`matplotlib.pyplot.savefig`.


        brain_color : :obj:`tuple`, default=(0.5, 0.5, 0.5)
            The brain color to use as the background color (e.g., for
            transparent colorbars).

        """
        # deal with "fake" 4D images
        if img is not None and img is not False:
            img = check_niimg_3d(img)

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
                figsize[0] += 0.7

            facecolor = "k" if black_bg else "w"

            if leave_space:
                figsize[0] += 3.4
            figure = plt.figure(figure, figsize=figsize, facecolor=facecolor)
        if isinstance(axes, plt.Axes):
            assert (
                axes.figure is figure
            ), "The axes passed are not in the figure"

        if axes is None:
            axes = [0.0, 0.0, 1.0, 1.0]
            if leave_space:
                axes = [0.3, 0, 0.7, 1.0]
        if isinstance(axes, collections.abc.Sequence):
            axes = figure.add_axes(axes)
        # People forget to turn their axis off, or to set the zorder, and
        # then they cannot see their slicer
        axes.axis("off")
        return cls(cut_coords, axes, black_bg, brain_color, **kwargs)

    def title(
        self,
        text,
        x=0.01,
        y=0.99,
        size=15,
        color=None,
        bgcolor=None,
        alpha=1,
        **kwargs,
    ):
        """Write a title to the view.

        Parameters
        ----------
        text : :obj:`str`
            The text of the title.

        x : :obj:`float`, default=0.01
            The horizontal position of the title on the frame in
            fraction of the frame width.

        y : :obj:`float`, default=0.99
            The vertical position of the title on the frame in
            fraction of the frame height.

        size : :obj:`int`, default=15
            The size of the title text.

        color : matplotlib color specifier, optional
            The color of the font of the title.

        bgcolor : matplotlib color specifier, optional
            The color of the background of the title.

        alpha : :obj:`float`, default=1
            The alpha value for the background.

        kwargs :
            Extra keyword arguments are passed to matplotlib's text
            function.
        """
        if color is None:
            color = "k" if self._black_bg else "w"
        if bgcolor is None:
            bgcolor = "w" if self._black_bg else "k"
        if hasattr(self, "_cut_displayed"):
            # Adapt to the case of mosaic plotting
            if isinstance(self.cut_coords, dict):
                first_axe = self._cut_displayed[-1]
                first_axe = (first_axe, self.cut_coords[first_axe][0])
            else:
                first_axe = self._cut_displayed[0]
        else:
            first_axe = self.cut_coords[0]
        ax = self.axes[first_axe].ax
        ax.text(
            x,
            y,
            text,
            transform=self.frame_axes.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            size=size,
            color=color,
            bbox={
                "boxstyle": "square,pad=.3",
                "ec": bgcolor,
                "fc": bgcolor,
                "alpha": alpha,
            },
            zorder=1000,
            **kwargs,
        )
        ax.set_zorder(1000)

    @fill_doc
    def add_overlay(
        self,
        img,
        threshold=1e-6,
        colorbar=False,
        cbar_tick_format="%.2g",
        cbar_vmin=None,
        cbar_vmax=None,
        **kwargs,
    ):
        """Plot a 3D map in all the views.

        Parameters
        ----------
        %(img)s
            If it is a masked array, only the non-masked part will be plotted.

        threshold : :obj:`int` or :obj:`float` or ``None``, default=1e-6
            Threshold to apply:

                - If ``None`` is given, the maps are not thresholded.
                - If a number is given, it is used to threshold the maps:
                  values below the threshold (in absolute value) are
                  plotted as transparent.

        cbar_tick_format : str, default="%%.2g" (scientific notation)
            Controls how to format the tick labels of the colorbar.
            Ex: use "%%i" to display as integers.

        colorbar : :obj:`bool`, default=False
            If ``True``, display a colorbar on the right of the plots.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`~matplotlib.pyplot.imshow`.

        cbar_vmin : :obj:`float`, optional
            Minimal value for the colorbar. If None, the minimal value
            is computed based on the data.

        cbar_vmax : :obj:`float`, optional
            Maximal value for the colorbar. If None, the maximal value
            is computed based on the data.
        """
        if colorbar and self._colorbar:
            raise ValueError(
                "This figure already has an overlay with a colorbar."
            )
        self._colorbar = colorbar
        self._cbar_tick_format = cbar_tick_format

        img = check_niimg_3d(img)

        # Make sure that add_overlay shows consistent default behavior
        # with plot_stat_map
        kwargs.setdefault("interpolation", "nearest")
        ims = self._map_show(img, type="imshow", threshold=threshold, **kwargs)

        # `ims` can be empty in some corner cases,
        # look at test_img_plotting.test_outlier_cut_coords.
        if colorbar and ims:
            self._show_colorbar(
                ims[0].cmap, ims[0].norm, cbar_vmin, cbar_vmax, threshold
            )

        plt.draw_if_interactive()

    @fill_doc
    def add_contours(self, img, threshold=1e-6, filled=False, **kwargs):
        """Contour a 3D map in all the views.

        Parameters
        ----------
        %(img)s
            Provides image to plot.

        threshold : :obj:`int` or :obj:`float` or ``None``, default=1e-6
            Threshold to apply:

                - If ``None`` is given, the maps are not thresholded.
                - If a number is given, it is used to threshold the maps,
                  values below the threshold (in absolute value) are plotted
                  as transparent.



        filled : :obj:`bool`, default=False
            If ``filled=True``, contours are displayed with color fillings.


        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`~matplotlib.pyplot.contour`, or function
            :func:`~matplotlib.pyplot.contourf`.
            Useful, arguments are typical "levels", which is a
            list of values to use for plotting a contour or contour
            fillings (if ``filled=True``), and
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
        self._map_show(img, type="contour", threshold=threshold, **kwargs)
        if filled:
            if "levels" in kwargs:
                levels = kwargs["levels"]
                if len(levels) <= 1:
                    # contour fillings levels
                    # should be given as (lower, upper).
                    levels.append(np.inf)

            self._map_show(img, type="contourf", threshold=threshold, **kwargs)

        plt.draw_if_interactive()

    def _map_show(
        self,
        img,
        type="imshow",
        resampling_interpolation="continuous",
        threshold=None,
        **kwargs,
    ):
        # In the special case where the affine of img is not diagonal,
        # the function `reorder_img` will trigger a resampling
        # of the provided image with a continuous interpolation
        # since this is the default value here. In the special
        # case where this image is binary, such as when this function
        # is called from `add_contours`, continuous interpolation
        # does not make sense and we turn to nearest interpolation instead.
        if is_binary_niimg(img):
            img = reorder_img(img, resample="nearest", copy_header=True)
        else:
            img = reorder_img(
                img, resample=resampling_interpolation, copy_header=True
            )
        threshold = float(threshold) if threshold is not None else None

        affine = img.affine
        if threshold is not None:
            data = safe_get_data(img, ensure_finite=True)
            if threshold == 0:
                data = np.ma.masked_equal(data, 0, copy=False)
            else:
                data = np.ma.masked_inside(
                    data, -threshold, threshold, copy=False
                )
            img = new_img_like(img, data, affine)

        data = safe_get_data(img, ensure_finite=True)
        data_bounds = get_bounds(data.shape, affine)
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = data_bounds

        xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = (
            xmin,
            xmax,
            ymin,
            ymax,
            zmin,
            zmax,
        )

        # Compute tight bounds
        if type in ("contour", "contourf"):
            # Define a pseudo threshold to have a tight bounding box
            thr = (
                0.9 * np.min(np.abs(kwargs["levels"]))
                if "levels" in kwargs
                else 1e-6
            )
            not_mask = np.logical_or(data > thr, data < -thr)
            xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = get_mask_bounds(
                new_img_like(img, not_mask, affine)
            )
        elif hasattr(data, "mask") and isinstance(data.mask, np.ndarray):
            not_mask = np.logical_not(data.mask)
            xmin_, xmax_, ymin_, ymax_, zmin_, zmax_ = get_mask_bounds(
                new_img_like(img, not_mask, affine)
            )

        data_2d_list = []
        for display_ax in self.axes.values():
            try:
                data_2d = display_ax.transform_to_2d(data, affine)
            except IndexError:
                # We are cutting outside the indices of the data
                data_2d = None

            data_2d_list.append(data_2d)

        if kwargs.get("vmin") is None:
            kwargs["vmin"] = np.ma.min(
                [d.min() for d in data_2d_list if d is not None]
            )
        if kwargs.get("vmax") is None:
            kwargs["vmax"] = np.ma.max(
                [d.max() for d in data_2d_list if d is not None]
            )

        bounding_box = (xmin_, xmax_), (ymin_, ymax_), (zmin_, zmax_)
        ims = []
        to_iterate_over = zip(self.axes.values(), data_2d_list)
        for display_ax, data_2d in to_iterate_over:
            # If data_2d is completely masked, then there is nothing to
            # plot. Hence, no point to do imshow().
            if data_2d is not None:
                data_2d = self._threshold(
                    data_2d,
                    threshold,
                    vmin=kwargs.get("vmin"),
                    vmax=kwargs.get("vmax"),
                )

                im = display_ax.draw_2d(
                    data_2d, data_bounds, bounding_box, type=type, **kwargs
                )
                ims.append(im)
        return ims

    @classmethod
    def _threshold(cls, data, threshold=None, vmin=None, vmax=None):
        """Threshold the data."""
        if threshold is not None:
            data = np.ma.masked_where(
                np.abs(data) <= threshold,
                data,
                copy=False,
            )
            if (vmin is not None) and (vmin >= -threshold):
                data = np.ma.masked_where(data < vmin, data, copy=False)
            if (vmax is not None) and (vmax <= threshold):
                data = np.ma.masked_where(data > vmax, data, copy=False)
        return data

    @fill_doc
    def _show_colorbar(
        self, cmap, norm, cbar_vmin=None, cbar_vmax=None, threshold=None
    ):
        """Display the colorbar.

        Parameters
        ----------
        %(cmap)s
        norm : :class:`~matplotlib.colors.Normalize`
            This object is typically found as the ``norm`` attribute of
            :class:`~matplotlib.image.AxesImage`.

        threshold : :obj:`float` or ``None``, optional
            The absolute value at which the colorbar is thresholded.

        cbar_vmin : :obj:`float`, optional
            Minimal value for the colorbar. If None, the minimal value
            is computed based on the data.

        cbar_vmax : :obj:`float`, optional
            Maximal value for the colorbar. If None, the maximal value
            is computed based on the data.
        """
        offset = 0 if threshold is None else threshold
        offset = min(offset, norm.vmax)

        cbar_vmin = cbar_vmin if cbar_vmin is not None else norm.vmin
        cbar_vmax = cbar_vmax if cbar_vmax is not None else norm.vmax

        # create new  axis for the colorbar
        figure = self.frame_axes.figure
        _, y0, x1, y1 = self.rect
        height = y1 - y0
        x_adjusted_width = self._colorbar_width / len(self.axes)
        x_adjusted_margin = self._colorbar_margin["right"] / len(self.axes)
        lt_wid_top_ht = [
            x1 - (x_adjusted_width + x_adjusted_margin),
            y0 + self._colorbar_margin["top"],
            x_adjusted_width,
            height
            - (self._colorbar_margin["top"] + self._colorbar_margin["bottom"]),
        ]
        self._colorbar_ax = figure.add_axes(lt_wid_top_ht)
        self._colorbar_ax.set_facecolor("w")

        our_cmap = plt.get_cmap(cmap)
        # edge case where the data has a single value
        # yields a cryptic matplotlib error message
        # when trying to plot the color bar
        n_ticks = 5 if cbar_vmin != cbar_vmax else 1
        ticks = _get_cbar_ticks(cbar_vmin, cbar_vmax, offset, n_ticks)
        bounds = np.linspace(cbar_vmin, cbar_vmax, our_cmap.N)

        # some colormap hacking
        cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
        transparent_start = int(norm(-offset, clip=True) * (our_cmap.N - 1))
        transparent_stop = int(norm(offset, clip=True) * (our_cmap.N - 1))
        for i in range(transparent_start, transparent_stop):
            cmaplist[i] = (*self._brain_color, 0.0)  # transparent
        if cbar_vmin == cbar_vmax:  # len(np.unique(data)) == 1 ?
            return
        else:
            our_cmap = LinearSegmentedColormap.from_list(
                "Custom cmap", cmaplist, our_cmap.N
            )
        self._cbar = ColorbarBase(
            self._colorbar_ax,
            ticks=ticks,
            norm=norm,
            orientation="vertical",
            cmap=our_cmap,
            boundaries=bounds,
            spacing="proportional",
            format=self._cbar_tick_format,
        )
        self._cbar.ax.set_facecolor(self._brain_color)

        self._colorbar_ax.yaxis.tick_left()
        tick_color = "w" if self._black_bg else "k"
        outline_color = "w" if self._black_bg else "k"

        for tick in self._colorbar_ax.yaxis.get_ticklabels():
            tick.set_color(tick_color)
        self._colorbar_ax.yaxis.set_tick_params(width=0)
        self._cbar.outline.set_edgecolor(outline_color)

    @fill_doc
    def add_edges(self, img, color="r"):
        """Plot the edges of a 3D map in all the views.

        Parameters
        ----------
        %(img)s
            The 3D map to be plotted.
            If it is a masked array, only the non-masked part will be plotted.

        color : matplotlib color: :obj:`str` or (r, g, b) value, default='r'
            The color used to display the edge map.

        """
        img = reorder_img(img, resample="continuous", copy_header=True)
        data = get_data(img)
        affine = img.affine
        single_color_cmap = ListedColormap([color])
        data_bounds = get_bounds(data.shape, img.affine)

        # For each ax, cut the data and plot it
        for display_ax in self.axes.values():
            try:
                data_2d = display_ax.transform_to_2d(data, affine)
                edge_mask = edge_map(data_2d)
            except IndexError:
                # We are cutting outside the indices of the data
                continue
            display_ax.draw_2d(
                edge_mask,
                data_bounds,
                data_bounds,
                type="imshow",
                cmap=single_color_cmap,
            )

        plt.draw_if_interactive()

    def add_markers(
        self, marker_coords, marker_color="r", marker_size=30, **kwargs
    ):
        """Add markers to the plot.

        Parameters
        ----------
        marker_coords : :class:`~numpy.ndarray` of shape ``(n_markers, 3)``
            Coordinates of the markers to plot. For each slice, only markers
            that are 2 millimeters away from the slice are plotted.

        marker_color : pyplot compatible color or \
                     :obj:`list` of shape ``(n_markers,)``, default='r'
            List of colors for each marker
            that can be string or matplotlib colors.


        marker_size : :obj:`float` or \
                    :obj:`list` of :obj:`float` of shape ``(n_markers,)``, \
                    default=30
            Size in pixel for each marker.
        """
        defaults = {"marker": "o", "zorder": 1000}
        marker_coords = np.asanyarray(marker_coords)
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        for display_ax in self.axes.values():
            direction = display_ax.direction
            coord = display_ax.coord
            marker_coords_2d, third_d = coords_3d_to_2d(
                marker_coords, direction, return_direction=True
            )
            xdata, ydata = marker_coords_2d.T
            # Allow markers only in their respective hemisphere
            # when appropriate
            marker_color_ = marker_color
            marker_size_ = marker_size
            if direction in ("lr"):
                if not isinstance(marker_color, str) and not isinstance(
                    marker_color, np.ndarray
                ):
                    marker_color_ = np.asarray(marker_color)
                xcoords, *_ = marker_coords.T
                if direction == "r":
                    relevant_coords = xcoords >= 0
                elif direction == "l":
                    relevant_coords = xcoords <= 0
                xdata = xdata[relevant_coords]
                ydata = ydata[relevant_coords]
                if (
                    not isinstance(marker_color, str)
                    and len(marker_color) != 1
                ):
                    marker_color_ = marker_color_[relevant_coords]
                if not isinstance(marker_size, numbers.Number):
                    marker_size_ = np.asarray(marker_size_)[relevant_coords]

            # Check if coord has integer represents a cut in direction
            # to follow the heuristic. If no foreground image is given
            # coordinate is empty or None. This case is valid for plotting
            # markers on glass brain without any foreground image.
            if isinstance(coord, numbers.Number):
                # Heuristic that plots only markers that are 2mm away
                # from the current slice.
                # XXX: should we keep this heuristic?
                mask = np.abs(third_d - coord) <= 2.0
                xdata = xdata[mask]
                ydata = ydata[mask]
            display_ax.ax.scatter(
                xdata, ydata, s=marker_size_, c=marker_color_, **kwargs
            )

    def annotate(
        self,
        left_right=True,
        positions=True,
        scalebar=False,
        size=12,
        scale_size=5.0,
        scale_units="cm",
        scale_loc=4,
        decimals=0,
        **kwargs,
    ):
        """Add annotations to the plot.

        Parameters
        ----------
        left_right : :obj:`bool`, default=True
            If ``True``, annotations indicating which side
            is left and which side is right are drawn.


        positions : :obj:`bool`, default=True
            If ``True``, annotations indicating the
            positions of the cuts are drawn.


        scalebar : :obj:`bool`, default=False
            If ``True``, cuts are annotated with a reference scale bar.
            For finer control of the scale bar, please check out
            the ``draw_scale_bar`` method on the axes in "axes" attribute
            of this object.


        size : :obj:`int`, default=12
            The size of the text used.

        scale_size : :obj:`int` or :obj:`float`, default=5.0
            The length of the scalebar, in units of ``scale_units``.


        scale_units : {'cm', 'mm'}, default='cm'
            The units for the ``scalebar``.

        scale_loc : :obj:`int`, default=4
            The positioning for the scalebar.
            Valid location codes are:

                - 1: "upper right"
                - 2: "upper left"
                - 3: "lower left"
                - 4: "lower right"
                - 5: "right"
                - 6: "center left"
                - 7: "center right"
                - 8: "lower center"
                - 9: "upper center"
                - 10: "center"

        decimals : :obj:`int`, default=0
            Number of decimal places on slice position annotation. If zero,
            the slice position is integer without decimal point.


        kwargs : :obj:`dict`
            Extra keyword arguments are passed to matplotlib's text
            function.
        """
        kwargs = kwargs.copy()
        if "color" not in kwargs:
            kwargs["color"] = "w" if self._black_bg else "k"
        bg_color = "k" if self._black_bg else "w"

        if left_right:
            for display_axis in self.axes.values():
                display_axis.draw_left_right(
                    size=size, bg_color=bg_color, **kwargs
                )

        if positions:
            for display_axis in self.axes.values():
                display_axis.draw_position(
                    size=size, bg_color=bg_color, decimals=decimals, **kwargs
                )

        if scalebar:
            axes = self.axes.values()
            for display_axis in axes:
                display_axis.draw_scale_bar(
                    bg_color=bg_color,
                    fontsize=size,
                    size=scale_size,
                    units=scale_units,
                    loc=scale_loc,
                    **kwargs,
                )

    def close(self):
        """Close the figure.

        This is necessary to avoid leaking memory.
        """
        plt.close(self.frame_axes.figure.number)

    def savefig(self, filename, dpi=None):
        """Save the figure to a file.

        Parameters
        ----------
        filename : :obj:`str`
            The file name to save to. Its extension determines the
            file type, typically '.png', '.svg' or '.pdf'.

        dpi : ``None`` or scalar, default=None
            The resolution in dots per inch.

        """
        facecolor = edgecolor = "k" if self._black_bg else "w"
        self.frame_axes.figure.savefig(
            filename, dpi=dpi, facecolor=facecolor, edgecolor=edgecolor
        )


def _get_cbar_ticks(vmin, vmax, offset, n_ticks=5):
    """Help for BaseSlicer."""
    # edge case where the data has a single value yields
    # a cryptic matplotlib error message when trying to plot the color bar
    if vmin == vmax:
        return np.linspace(vmin, vmax, 1)

    # edge case where the data has all negative values but vmax is exactly 0
    if vmax == 0:
        vmax += np.finfo(np.float32).eps

    # If a threshold is specified, we want two of the tick
    # to correspond to -thresold and +threshold on the colorbar.
    # If the threshold is very small compared to vmax,
    # we use a simple linspace as the result would be very difficult to see.
    ticks = np.linspace(vmin, vmax, n_ticks)
    if offset is not None and offset / vmax > 0.12:
        diff = [abs(abs(tick) - offset) for tick in ticks]
        # Edge case where the thresholds are exactly
        # at the same distance to 4 ticks
        if diff.count(min(diff)) == 4:
            idx_closest = np.sort(np.argpartition(diff, 4)[:4])
            idx_closest = np.isin(ticks, np.sort(ticks[idx_closest])[1:3])
        else:
            # Find the closest 2 ticks
            idx_closest = np.sort(np.argpartition(diff, 2)[:2])
            if 0 in ticks[idx_closest]:
                idx_closest = np.sort(np.argpartition(diff, 3)[:3])
                idx_closest = idx_closest[[0, 2]]
        ticks[idx_closest] = [-offset, offset]
    if len(ticks) > 0 and ticks[0] < vmin:
        ticks[0] = vmin

    return ticks


@fill_doc
class OrthoSlicer(BaseSlicer):
    """Class to create 3 linked axes for plotting orthogonal \
    cuts of 3D maps.

    This visualization mode can be activated
    from Nilearn plotting functions, like
    :func:`~nilearn.plotting.plot_img`, by setting
    ``display_mode='ortho'``:

     .. code-block:: python

         from nilearn.datasets import load_mni152_template
         from nilearn.plotting import plot_img

         img = load_mni152_template()
         # display is an instance of the OrthoSlicer class
         display = plot_img(img, display_mode="ortho")


    Attributes
    ----------
    cut_coords : :obj:`list`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~matplotlib.axes.Axes`
        The 3 axes used to plot each view.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    Notes
    -----
    The extent of the different axes are adjusted to fit the data
    best in the viewing area.

    See Also
    --------
    nilearn.plotting.displays.MosaicSlicer : Three cuts are performed \
    along multiple rows and columns.
    nilearn.plotting.displays.TiledSlicer : Three cuts are performed \
    and arranged in a 2x2 grid.

    """

    _cut_displayed: ClassVar[str] = "yxz"
    _axes_class = CutAxes
    _default_figsize: ClassVar[list[float, float]] = [2.2, 3.5]

    @classmethod
    @fill_doc  # the fill_doc decorator must be last applied
    def find_cut_coords(cls, img=None, threshold=None, cut_coords=None):
        """Instantiate the slicer and find cut coordinates.

        Parameters
        ----------
        %(img)s
        threshold : :obj:`int` or :obj:`float` or ``None``, default=None
            Threshold to apply:

                - If ``None`` is given, the maps are not thresholded.
                - If a number is given, it is used to threshold the maps,
                  values below the threshold (in absolute value) are plotted
                  as transparent.



        cut_coords : 3 :obj:`tuple` of :obj:`int`
            The cut position, in world space.
        """
        if cut_coords is None:
            if img is None or img is False:
                cut_coords = (0, 0, 0)
            else:
                cut_coords = find_xyz_cut_coords(
                    img, activation_threshold=threshold
                )
            cut_coords = [
                cut_coords["xyz".find(c)] for c in sorted(cls._cut_displayed)
            ]
        return cut_coords

    def _init_axes(self, **kwargs):
        cut_coords = self.cut_coords
        if len(cut_coords) != len(self._cut_displayed):
            raise ValueError(
                "The number cut_coords passed does not"
                " match the display_mode"
            )
        x0, y0, x1, y1 = self.rect
        facecolor = "k" if self._black_bg else "w"
        # Create our axes:
        self.axes = {}
        for index, direction in enumerate(self._cut_displayed):
            fh = self.frame_axes.get_figure()
            ax = fh.add_axes(
                [0.3 * index * (x1 - x0) + x0, y0, 0.3 * (x1 - x0), y1 - y0],
                aspect="equal",
            )
            ax.set_facecolor(facecolor)

            ax.axis("off")
            coord = self.cut_coords[
                sorted(self._cut_displayed).index(direction)
            ]
            display_ax = self._axes_class(ax, direction, coord, **kwargs)
            self.axes[direction] = display_ax
            ax.set_axes_locator(self._locator)

        if self._black_bg:
            for ax in self.axes.values():
                ax.ax.imshow(
                    np.zeros((2, 2, 3)),
                    extent=[-5000, 5000, -5000, 5000],
                    zorder=-500,
                    aspect="equal",
                )

            # To have a black background in PDF, we need to create a
            # patch in black for the background
            self.frame_axes.imshow(
                np.zeros((2, 2, 3)),
                extent=[-5000, 5000, -5000, 5000],
                zorder=-500,
                aspect="auto",
            )
            self.frame_axes.set_zorder(-1000)

    def _locator(
        self,
        axes,
        renderer,  # noqa: ARG002
    ):
        """Adjust the size of the axes.

        The locator function used by matplotlib to position axes.

        Here we put the logic used to adjust the size of the axes.

        ``renderer`` is required to match the matplotlib API.
        """
        x0, y0, x1, y1 = self.rect
        # A dummy axes, for the situation in which we are not plotting
        # all three (x, y, z) cuts
        dummy_ax = self._axes_class(None, None, None)
        width_dict = {dummy_ax.ax: 0}
        display_ax_dict = self.axes

        if self._colorbar:
            adjusted_width = self._colorbar_width / len(self.axes)
            right_margin = self._colorbar_margin["right"] / len(self.axes)
            ticks_margin = self._colorbar_margin["left"] / len(self.axes)
            x1 = x1 - (adjusted_width + ticks_margin + right_margin)

        for display_ax in display_ax_dict.values():
            bounds = display_ax.get_object_bounds()
            if not bounds:
                # This happens if the call to _map_show was not
                # successful. As it happens asynchronously (during a
                # refresh of the figure) we capture the problem and
                # ignore it: it only adds a non informative traceback
                bounds = [0, 1, 0, 1]
            xmin, xmax, _, _ = bounds
            width_dict[display_ax.ax] = xmax - xmin

        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.items():
            width_dict[ax] = width / total_width * (x1 - x0)

        direction_ax = [
            display_ax_dict.get(d, dummy_ax).ax for d in self._cut_displayed
        ]
        left_dict = {}
        for idx, ax in enumerate(direction_ax):
            left_dict[ax] = x0
            for prev_ax in direction_ax[:idx]:
                left_dict[ax] += width_dict[prev_ax]

        return Bbox(
            [[left_dict[axes], y0], [left_dict[axes] + width_dict[axes], y1]]
        )

    def draw_cross(self, cut_coords=None, **kwargs):
        """Draw a crossbar on the plot to show where the cut is performed.

        Parameters
        ----------
        cut_coords : 3-:obj:`tuple` of :obj:`float`, optional
            The position of the cross to draw. If ``None`` is passed, the
            ``OrthoSlicer``'s cut coordinates are used.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`~matplotlib.pyplot.axhline`.
        """
        if cut_coords is None:
            cut_coords = self.cut_coords
        coords = {}
        for direction in "xyz":
            coord = None
            if direction in self._cut_displayed:
                coord = cut_coords[
                    sorted(self._cut_displayed).index(direction)
                ]
            coords[direction] = coord
        x, y, z = coords["x"], coords["y"], coords["z"]

        kwargs = kwargs.copy()
        if "color" not in kwargs:
            kwargs["color"] = ".8" if self._black_bg else "k"
        if "y" in self.axes:
            ax = self.axes["y"].ax
            if x is not None:
                ax.axvline(x, ymin=0.05, ymax=0.95, **kwargs)
            if z is not None:
                ax.axhline(z, **kwargs)

        if "x" in self.axes:
            ax = self.axes["x"].ax
            if y is not None:
                ax.axvline(y, ymin=0.05, ymax=0.95, **kwargs)
            if z is not None:
                ax.axhline(z, xmax=0.95, **kwargs)

        if "z" in self.axes:
            ax = self.axes["z"].ax
            if x is not None:
                ax.axvline(x, ymin=0.05, ymax=0.95, **kwargs)
            if y is not None:
                ax.axhline(y, **kwargs)


class TiledSlicer(BaseSlicer):
    """A class to create 3 axes for plotting orthogonal \
    cuts of 3D maps, organized in a 2x2 grid.

    This visualization mode can be activated from Nilearn plotting functions,
    like :func:`~nilearn.plotting.plot_img`, by setting
    ``display_mode='tiled'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the TiledSlicer class
        display = plot_img(img, display_mode="tiled")

    Attributes
    ----------
    cut_coords : :obj:`list`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~matplotlib.axes.Axes`
        The 3 axes used to plot each view.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    Notes
    -----
    The extent of the different axes are adjusted to fit the data
    best in the viewing area.

    See Also
    --------
    nilearn.plotting.displays.MosaicSlicer : Three cuts are performed \
    along multiple rows and columns.
    nilearn.plotting.displays.OrthoSlicer : Three cuts are performed \
       and arranged in a 2x2 grid.

    """

    _cut_displayed: ClassVar[str] = "yxz"
    _axes_class = CutAxes
    _default_figsize: ClassVar[list[float, float]] = [2.0, 7.6]

    @classmethod
    def find_cut_coords(cls, img=None, threshold=None, cut_coords=None):
        """Instantiate the slicer and find cut coordinates.

        Parameters
        ----------
        img : 3D :class:`~nibabel.nifti1.Nifti1Image`
            The brain map.

        threshold : :obj:`float`, optional
            The lower threshold to the positive activation.
            If ``None``, the activation threshold is computed using the
            80% percentile of the absolute value of the map.

        cut_coords : :obj:`list` of :obj:`float`, optional
            xyz world coordinates of cuts.

        Returns
        -------
        cut_coords : :obj:`list` of :obj:`float`
            xyz world coordinates of cuts.
        """
        if cut_coords is None:
            if img is None or img is False:
                cut_coords = (0, 0, 0)
            else:
                cut_coords = find_xyz_cut_coords(
                    img, activation_threshold=threshold
                )
            cut_coords = [
                cut_coords["xyz".find(c)] for c in sorted(cls._cut_displayed)
            ]

        return cut_coords

    def _find_initial_axes_coord(self, index):
        """Find coordinates for initial axes placement for xyz cuts.

        Parameters
        ----------
        index : :obj:`int`
            Index corresponding to current cut 'x', 'y' or 'z'.

        Returns
        -------
        [coord1, coord2, coord3, coord4] : :obj:`list` of :obj:`int`
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
        """Initialize and place axes for display of 'xyz' cuts.

        Parameters
        ----------
        kwargs : :obj:`dict`
            Additional arguments to pass to ``self._axes_class``.
        """
        cut_coords = self.cut_coords
        if len(cut_coords) != len(self._cut_displayed):
            raise ValueError(
                "The number cut_coords passed does not"
                " match the display_mode"
            )

        facecolor = "k" if self._black_bg else "w"

        self.axes = {}
        for index, direction in enumerate(self._cut_displayed):
            fh = self.frame_axes.get_figure()
            axes_coords = self._find_initial_axes_coord(index)
            ax = fh.add_axes(axes_coords, aspect="equal")

            ax.set_facecolor(facecolor)

            ax.axis("off")
            coord = self.cut_coords[
                sorted(self._cut_displayed).index(direction)
            ]
            display_ax = self._axes_class(ax, direction, coord, **kwargs)
            self.axes[direction] = display_ax
            ax.set_axes_locator(self._locator)

    def _adjust_width_height(
        self, width_dict, height_dict, rect_x0, rect_y0, rect_x1, rect_y1
    ):
        """Adjust absolute image width and height to ratios.

        Parameters
        ----------
        width_dict : :obj:`dict`
            Width of image cuts displayed in axes.

        height_dict : :obj:`dict`
            Height of image cuts displayed in axes.

        rect_x0, rect_y0, rect_x1, rect_y1 : :obj:`float`
            Matplotlib figure boundaries.

        Returns
        -------
        width_dict : :obj:`dict`
            Width ratios of image cuts for optimal positioning of axes.

        height_dict : :obj:`dict`
            Height ratios of image cuts for optimal positioning of axes.
        """
        total_height = 0
        total_width = 0

        if "y" in self.axes:
            ax = self.axes["y"].ax
            total_height += height_dict[ax]
            total_width += width_dict[ax]

        if "x" in self.axes:
            ax = self.axes["x"].ax
            total_width = total_width + width_dict[ax]

        if "z" in self.axes:
            ax = self.axes["z"].ax
            total_height = total_height + height_dict[ax]

        for ax, width in width_dict.items():
            width_dict[ax] = width / total_width * (rect_x1 - rect_x0)

        for ax, height in height_dict.items():
            height_dict[ax] = height / total_height * (rect_y1 - rect_y0)

        return (width_dict, height_dict)

    def _find_axes_coord(
        self,
        rel_width_dict,
        rel_height_dict,
        rect_x0,
        rect_y0,
        rect_x1,
        rect_y1,
    ):
        """Find coordinates for initial axes placement for xyz cuts.

        Parameters
        ----------
        rel_width_dict : :obj:`dict`
            Width ratios of image cuts for optimal positioning of axes.

        rel_height_dict : :obj:`dict`
            Height ratios of image cuts for optimal positioning of axes.

        rect_x0, rect_y0, rect_x1, rect_y1 : :obj:`float`
            Matplotlib figure boundaries.

        Returns
        -------
        coord1, coord2, coord3, coord4 : :obj:`dict`
            x0, y0, x1, y1 coordinates per axes used by matplotlib
            to position axes in figure.
        """
        coord1 = {}
        coord2 = {}
        coord3 = {}
        coord4 = {}

        if "y" in self.axes:
            ax = self.axes["y"].ax
            coord1[ax] = rect_x0
            coord2[ax] = (rect_y1) - rel_height_dict[ax]
            coord3[ax] = rect_x0 + rel_width_dict[ax]
            coord4[ax] = rect_y1

        if "x" in self.axes:
            ax = self.axes["x"].ax
            coord1[ax] = (rect_x1) - rel_width_dict[ax]
            coord2[ax] = (rect_y1) - rel_height_dict[ax]
            coord3[ax] = rect_x1
            coord4[ax] = rect_y1

        if "z" in self.axes:
            ax = self.axes["z"].ax
            coord1[ax] = rect_x0
            coord2[ax] = rect_y0
            coord3[ax] = rect_x0 + rel_width_dict[ax]
            coord4[ax] = rect_y0 + rel_height_dict[ax]

        return (coord1, coord2, coord3, coord4)

    def _locator(
        self,
        axes,
        renderer,  # noqa: ARG002
    ):
        """Adjust the size of the axes.

        The locator function used by matplotlib to position axes.

        Here we put the logic used to adjust the size of the axes.

        ``renderer`` is required to match the matplotlib API.
        """
        rect_x0, rect_y0, rect_x1, rect_y1 = self.rect

        # A dummy axes, for the situation in which we are not plotting
        # all three (x, y, z) cuts
        dummy_ax = self._axes_class(None, None, None)
        width_dict = {dummy_ax.ax: 0}
        height_dict = {dummy_ax.ax: 0}
        display_ax_dict = self.axes

        if self._colorbar:
            adjusted_width = self._colorbar_width / len(self.axes)
            right_margin = self._colorbar_margin["right"] / len(self.axes)
            ticks_margin = self._colorbar_margin["left"] / len(self.axes)
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
            width_dict[display_ax.ax] = xmax - xmin
            height_dict[display_ax.ax] = ymax - ymin

        # relative image height and width
        rel_width_dict, rel_height_dict = self._adjust_width_height(
            width_dict, height_dict, rect_x0, rect_y0, rect_x1, rect_y1
        )

        coord1, coord2, coord3, coord4 = self._find_axes_coord(
            rel_width_dict, rel_height_dict, rect_x0, rect_y0, rect_x1, rect_y1
        )

        return Bbox(
            [[coord1[axes], coord2[axes]], [coord3[axes], coord4[axes]]]
        )

    def draw_cross(self, cut_coords=None, **kwargs):
        """Draw a crossbar on the plot to show where the cut is performed.

        Parameters
        ----------
        cut_coords : 3-:obj:`tuple` of :obj:`float`, optional
            The position of the cross to draw. If ``None`` is passed, the
            ``OrthoSlicer``'s cut coordinates are used.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`~matplotlib.pyplot.axhline`.
        """
        if cut_coords is None:
            cut_coords = self.cut_coords
        coords = {}
        for direction in "xyz":
            coord_ = None
            if direction in self._cut_displayed:
                sorted_cuts = sorted(self._cut_displayed)
                index = sorted_cuts.index(direction)
                coord_ = cut_coords[index]
            coords[direction] = coord_
        x, y, z = coords["x"], coords["y"], coords["z"]

        kwargs = kwargs.copy()
        if "color" not in kwargs:
            with contextlib.suppress(KeyError):
                kwargs["color"] = ".8" if self._black_bg else "k"

        if "y" in self.axes:
            ax = self.axes["y"].ax
            if x is not None:
                ax.axvline(x, **kwargs)
            if z is not None:
                ax.axhline(z, **kwargs)

        if "x" in self.axes:
            ax = self.axes["x"].ax
            if y is not None:
                ax.axvline(y, **kwargs)
            if z is not None:
                ax.axhline(z, **kwargs)

        if "z" in self.axes:
            ax = self.axes["z"].ax
            if x is not None:
                ax.axvline(x, **kwargs)
            if y is not None:
                ax.axhline(y, **kwargs)


class BaseStackedSlicer(BaseSlicer):
    """A class to create linked axes for plotting stacked cuts of 2D maps.

    Attributes
    ----------
    axes : :obj:`dict` of :class:`~matplotlib.axes.Axes`
        The axes used to plot each view.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    Notes
    -----
    The extent of the different axes are adjusted to fit the data
    best in the viewing area.
    """

    @classmethod
    def find_cut_coords(
        cls,
        img=None,
        threshold=None,  # noqa: ARG003
        cut_coords=None,
    ):
        """Instantiate the slicer and find cut coordinates.

        Parameters
        ----------
        img : 3D :class:`~nibabel.nifti1.Nifti1Image`
            The brain map.

        threshold : :obj:`float`, optional
            The lower threshold to the positive activation.
            If ``None``, the activation threshold is computed using the
            80% percentile of the absolute value of the map.

        cut_coords : :obj:`list` of :obj:`float`, optional
            xyz world coordinates of cuts.

        Returns
        -------
        cut_coords : :obj:`list` of :obj:`float`
            xyz world coordinates of cuts.
        """
        if cut_coords is None:
            cut_coords = 7

        if img is None or img is False:
            bounds = ((-40, 40), (-30, 30), (-30, 75))
            lower, upper = bounds["xyz".index(cls._direction)]
            if isinstance(cut_coords, numbers.Number):
                cut_coords = np.linspace(lower, upper, cut_coords).tolist()
        elif not isinstance(
            cut_coords, collections.abc.Sequence
        ) and isinstance(cut_coords, numbers.Number):
            cut_coords = find_cut_slices(
                img, direction=cls._direction, n_cuts=cut_coords
            )

        return cut_coords

    def _init_axes(self, **kwargs):
        x0, y0, x1, y1 = self.rect
        # Create our axes:
        self.axes = {}
        fraction = 1.0 / len(self.cut_coords)
        for index, coord in enumerate(self.cut_coords):
            coord = float(coord)
            fh = self.frame_axes.get_figure()
            ax = fh.add_axes(
                [
                    fraction * index * (x1 - x0) + x0,
                    y0,
                    fraction * (x1 - x0),
                    y1 - y0,
                ]
            )
            ax.axis("off")
            display_ax = self._axes_class(ax, self._direction, coord, **kwargs)
            self.axes[coord] = display_ax
            ax.set_axes_locator(self._locator)

        if self._black_bg:
            for ax in self.axes.values():
                ax.ax.imshow(
                    np.zeros((2, 2, 3)),
                    extent=[-5000, 5000, -5000, 5000],
                    zorder=-500,
                    aspect="equal",
                )

            # To have a black background in PDF, we need to create a
            # patch in black for the background
            self.frame_axes.imshow(
                np.zeros((2, 2, 3)),
                extent=[-5000, 5000, -5000, 5000],
                zorder=-500,
                aspect="auto",
            )
            self.frame_axes.set_zorder(-1000)

    def _locator(
        self,
        axes,
        renderer,  # noqa: ARG002
    ):
        """Adjust the size of the axes.

        The locator function used by matplotlib to position axes.

        Here we put the logic used to adjust the size of the axes.

        ``renderer`` is required to match the matplotlib API.
        """
        x0, y0, x1, y1 = self.rect
        width_dict = {}
        display_ax_dict = self.axes

        if self._colorbar:
            adjusted_width = self._colorbar_width / len(self.axes)
            right_margin = self._colorbar_margin["right"] / len(self.axes)
            ticks_margin = self._colorbar_margin["left"] / len(self.axes)
            x1 = x1 - (adjusted_width + right_margin + ticks_margin)

        for display_ax in display_ax_dict.values():
            bounds = display_ax.get_object_bounds()
            if not bounds:
                # This happens if the call to _map_show was not
                # successful. As it happens asynchronously (during a
                # refresh of the figure) we capture the problem and
                # ignore it: it only adds a non informative traceback
                bounds = [0, 1, 0, 1]
            xmin, xmax, _, _ = bounds
            width_dict[display_ax.ax] = xmax - xmin
        total_width = float(sum(width_dict.values()))
        for ax, width in width_dict.items():
            width_dict[ax] = width / total_width * (x1 - x0)
        left_dict = {}
        left = float(x0)
        for display_ax in display_ax_dict.values():
            left_dict[display_ax.ax] = left
            this_width = width_dict[display_ax.ax]
            left += this_width
        return Bbox(
            [[left_dict[axes], y0], [left_dict[axes] + width_dict[axes], y1]]
        )

    def draw_cross(self, cut_coords=None, **kwargs):
        """Draw a crossbar on the plot to show where the cut is performed.

        Parameters
        ----------
        cut_coords : 3-:obj:`tuple` of :obj:`float`, optional
            The position of the cross to draw. If ``None`` is passed, the
            ``OrthoSlicer``'s cut coordinates are used.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`matplotlib.pyplot.axhline`.
        """
        pass


class XSlicer(BaseStackedSlicer):
    """The ``XSlicer`` class enables sagittal visualization with \
    plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='x'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the XSlicer class
        display = plot_img(img, display_mode="x")

    Attributes
    ----------
    cut_coords : 1D :class:`~numpy.ndarray`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.YSlicer : Coronal view
    nilearn.plotting.displays.ZSlicer : Axial view

    """

    _direction: ClassVar[str] = "x"
    _default_figsize: ClassVar[list[float, float]] = [2.6, 2.3]


class YSlicer(BaseStackedSlicer):
    """The ``YSlicer`` class enables coronal visualization with \
    plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='y'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the YSlicer class
        display = plot_img(img, display_mode="y")

    Attributes
    ----------
    cut_coords : 1D :class:`~numpy.ndarray`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XSlicer : Sagittal view
    nilearn.plotting.displays.ZSlicer : Axial view

    """

    _direction: ClassVar[str] = "y"
    _default_figsize: ClassVar[list[float, float]] = [2.2, 3.0]


class ZSlicer(BaseStackedSlicer):
    """The ``ZSlicer`` class enables axial visualization with \
    plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='z'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the ZSlicer class
        display = plot_img(img, display_mode="z")

    Attributes
    ----------
    cut_coords : 1D :class:`~numpy.ndarray`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XSlicer : Sagittal view
    nilearn.plotting.displays.YSlicer : Coronal view

    """

    _direction: ClassVar[str] = "z"
    _default_figsize: ClassVar[list[float, float]] = [2.2, 3.2]


class XZSlicer(OrthoSlicer):
    """The ``XZSlicer`` class enables to combine sagittal and axial views \
    on the same figure with plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='xz'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the XZSlicer class
        display = plot_img(img, display_mode="xz")

    Attributes
    ----------
    cut_coords : :obj:`list` of :obj:`float`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting in each direction ('x' and 'z' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.YXSlicer : Coronal + Sagittal views
    nilearn.plotting.displays.YZSlicer : Coronal + Axial views

    """

    _cut_displayed = "xz"


class YXSlicer(OrthoSlicer):
    """The ``YXSlicer`` class enables to combine coronal and sagittal views \
    on the same figure with plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='yx'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the YXSlicer class
        display = plot_img(img, display_mode="yx")

    Attributes
    ----------
    cut_coords : :obj:`list` of :obj:`float`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting in each direction ('x' and 'y' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XZSlicer : Sagittal + Axial views
    nilearn.plotting.displays.YZSlicer : Coronal + Axial views

    """

    _cut_displayed = "yx"


class YZSlicer(OrthoSlicer):
    """The ``YZSlicer`` class enables to combine coronal and axial views \
    on the same figure with plotting functions of Nilearn like \
    :func:`nilearn.plotting.plot_img`.

    This visualization mode
    can be activated by setting ``display_mode='yz'``:

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the YZSlicer class
        display = plot_img(img, display_mode="yz")

    Attributes
    ----------
    cut_coords : :obj:`list` of :obj:`float`
        The cut coordinates.

    axes : :obj:`dict` of :class:`~nilearn.plotting.displays.CutAxes`
        The axes used for plotting in each direction ('y' and 'z' here).

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.XZSlicer : Sagittal + Axial views
    nilearn.plotting.displays.YXSlicer : Coronal + Sagittal views

    """

    _cut_displayed: ClassVar[str] = "yz"
    _default_figsize: ClassVar[list[float, float]] = [2.2, 3.0]


class MosaicSlicer(BaseSlicer):
    """A class to create 3 :class:`~matplotlib.axes.Axes` for \
    plotting cuts of 3D maps, in multiple rows and columns.

    This visualization mode can be activated from Nilearn plotting
    functions, like :func:`~nilearn.plotting.plot_img`, by setting
    ``display_mode='mosaic'``.

    .. code-block:: python

        from nilearn.datasets import load_mni152_template
        from nilearn.plotting import plot_img

        img = load_mni152_template()
        # display is an instance of the MosaicSlicer class
        display = plot_img(img, display_mode="mosaic")

    Attributes
    ----------
    cut_coords : :obj:`dict` <:obj:`str`: 1D :class:`~numpy.ndarray`>
        The cut coordinates in a dictionary. The keys are the directions
        ('x', 'y', 'z'), and the values are arrays holding the cut
        coordinates.

    axes : :obj:`dict` of :class:`~matplotlib.axes.Axes`
        The 3 axes used to plot multiple views.

    frame_axes : :class:`~matplotlib.axes.Axes`
        The axes framing the whole set of views.

    See Also
    --------
    nilearn.plotting.displays.TiledSlicer : Three cuts are performed \
    in orthogonal directions.
    nilearn.plotting.displays.OrthoSlicer : Three cuts are performed \
    and arranged in a 2x2 grid.

    """

    _cut_displayed: ClassVar[str] = "yxz"
    _axes_class: ClassVar[CutAxes] = CutAxes
    _default_figsize: ClassVar[list[float, float]] = [4.0, 5.0]

    @classmethod
    def find_cut_coords(
        cls,
        img=None,
        threshold=None,  # noqa: ARG003
        cut_coords=None,
    ):
        """Instantiate the slicer and find cut coordinates for mosaic plotting.

        Parameters
        ----------
        img : 3D :class:`~nibabel.nifti1.Nifti1Image`, optional
            The brain image.

        threshold : :obj:`float`, optional
            The lower threshold to the positive activation. If ``None``,
            the activation threshold is computed using the 80% percentile of
            the absolute value of the map.

        cut_coords : :obj:`list` / :obj:`tuple` of 3 :obj:`float`,\
        :obj:`int`, optional
            xyz world coordinates of cuts. If ``cut_coords``
            are not provided, 7 coordinates of cuts are automatically
            calculated.

        Returns
        -------
        cut_coords : :obj:`dict`
            xyz world coordinates of cuts in a direction.
            Each key denotes the direction.
        """
        if cut_coords is None:
            cut_coords = 7

        if not isinstance(cut_coords, collections.abc.Sequence) and isinstance(
            cut_coords, numbers.Number
        ):
            cut_coords = [cut_coords] * 3
        else:
            if len(cut_coords) != len(cls._cut_displayed):
                raise ValueError(
                    "The number cut_coords passed does not"
                    " match the display_mode. Mosaic plotting "
                    "expects tuple of length 3."
                )
            cut_coords = [
                cut_coords["xyz".find(c)] for c in sorted(cls._cut_displayed)
            ]
        cut_coords = cls._find_cut_coords(img, cut_coords, cls._cut_displayed)
        return cut_coords

    @staticmethod
    def _find_cut_coords(img, cut_coords, cut_displayed):
        """Find slicing positions along a given axis.

        Help to :func:`~nilearn.plotting.find_cut_coords`.

        Parameters
        ----------
        img : 3D :class:`~nibabel.nifti1.Nifti1Image`
            The brain image.

        cut_coords : :obj:`list` / :obj:`tuple` of 3 :obj:`float`,\
        :obj:`int`, optional
            xyz world coordinates of cuts.

        cut_displayed : :obj:`str`
            Sectional directions 'yxz'.

        Returns
        -------
        cut_coords : 1D :class:`~numpy.ndarray` of length specified\
        in ``n_cuts``
            The computed ``cut_coords``.
        """
        coords = {}
        if img is None or img is False:
            bounds = ((-40, 40), (-30, 30), (-30, 75))
            for direction, n_cuts in zip(sorted(cut_displayed), cut_coords):
                lower, upper = bounds["xyz".index(direction)]
                coords[direction] = np.linspace(lower, upper, n_cuts).tolist()
        else:
            for direction, n_cuts in zip(sorted(cut_displayed), cut_coords):
                coords[direction] = find_cut_slices(
                    img, direction=direction, n_cuts=n_cuts
                )
        return coords

    def _init_axes(self, **kwargs):
        """Initialize and place axes for display of 'xyz' multiple cuts.

        Also adapts the width of the color bar relative to the axes.

        Parameters
        ----------
        kwargs : :obj:`dict`
            Additional arguments to pass to ``self._axes_class``.
        """
        if not isinstance(self.cut_coords, dict):
            self.cut_coords = self.find_cut_coords(cut_coords=self.cut_coords)

        if len(self.cut_coords) != len(self._cut_displayed):
            raise ValueError(
                "The number cut_coords passed does not"
                " match the mosaic mode"
            )
        x0, y0, x1, y1 = self.rect

        # Create our axes:
        self.axes = {}
        # portions for main axes
        fraction = y1 / len(self.cut_coords)
        height = fraction
        for index, direction in enumerate(self._cut_displayed):
            coords = self.cut_coords[direction]
            # portions allotment for each of 'x', 'y', 'z' coordinate
            fraction_c = 1.0 / len(coords)
            fh = self.frame_axes.get_figure()
            indices = [
                x0,
                fraction * index * (y1 - y0) + y0,
                x1,
                fraction * (y1 - y0),
            ]
            ax = fh.add_axes(indices)
            ax.axis("off")
            this_x0, this_y0, this_x1, _ = indices
            for index_c, coord in enumerate(coords):
                coord = float(coord)
                fh_c = ax.get_figure()
                # indices for each sub axes within main axes
                indices = [
                    fraction_c * index_c * (this_x1 - this_x0) + this_x0,
                    this_y0,
                    fraction_c * (this_x1 - this_x0),
                    height,
                ]
                ax = fh_c.add_axes(indices)
                ax.axis("off")
                display_ax = self._axes_class(ax, direction, coord, **kwargs)
                self.axes[(direction, coord)] = display_ax
                ax.set_axes_locator(self._locator)

        # increase color bar width to adapt to the number of cuts
        #  see issue https://github.com/nilearn/nilearn/pull/4284
        self._colorbar_width *= len(coords) ** 1.1

    def _locator(
        self,
        axes,
        renderer,  # noqa: ARG002
    ):
        """Adjust the size of the axes.

        Locator function used by matplotlib to position axes.

        Here we put the logic used to adjust the size of the axes.

        ``renderer`` is required to match the matplotlib API.
        """
        x0, y0, x1, y1 = self.rect
        display_ax_dict = self.axes

        if self._colorbar:
            adjusted_width = self._colorbar_width / len(self.axes)
            right_margin = self._colorbar_margin["right"] / len(self.axes)
            ticks_margin = self._colorbar_margin["left"] / len(self.axes)
            x1 = x1 - (adjusted_width + right_margin + ticks_margin)

        # capture widths for each axes for anchoring Bbox
        width_dict = {}
        for direction in self._cut_displayed:
            this_width = {}
            for display_ax in display_ax_dict.values():
                if direction == display_ax.direction:
                    bounds = display_ax.get_object_bounds()
                    if not bounds:
                        # This happens if the call to _map_show was not
                        # successful. As it happens asynchronously (during a
                        # refresh of the figure) we capture the problem and
                        # ignore it: it only adds a non informative traceback
                        bounds = [0, 1, 0, 1]
                    xmin, xmax, _, _ = bounds
                    this_width[display_ax.ax] = xmax - xmin
            total_width = float(sum(this_width.values()))
            for ax, w in this_width.items():
                width_dict[ax] = w / total_width * (x1 - x0)

        left_dict = {}
        # bottom positions in Bbox according to cuts
        bottom_dict = {}
        # fraction is divided by the cut directions 'y', 'x', 'z'
        fraction = y1 / len(self._cut_displayed)
        height_dict = {}
        for index, direction in enumerate(self._cut_displayed):
            left = float(x0)
            this_height = fraction + fraction * index
            for display_ax in display_ax_dict.values():
                if direction == display_ax.direction:
                    left_dict[display_ax.ax] = left
                    this_width = width_dict[display_ax.ax]
                    left += this_width
                    bottom_dict[display_ax.ax] = fraction * index * (y1 - y0)
                    height_dict[display_ax.ax] = this_height
        return Bbox(
            [
                [left_dict[axes], bottom_dict[axes]],
                [left_dict[axes] + width_dict[axes], height_dict[axes]],
            ]
        )

    def draw_cross(self, cut_coords=None, **kwargs):
        """Draw a crossbar on the plot to show where the cut is performed.

        Parameters
        ----------
        cut_coords : 3-:obj:`tuple` of :obj:`float`, optional
            The position of the cross to draw. If ``None`` is passed, the
            ``OrthoSlicer``'s cut coordinates are used.

        kwargs : :obj:`dict`
            Extra keyword arguments are passed to function
            :func:`matplotlib.pyplot.axhline`.
        """
        pass


SLICERS = {
    "ortho": OrthoSlicer,
    "tiled": TiledSlicer,
    "mosaic": MosaicSlicer,
    "xz": XZSlicer,
    "yz": YZSlicer,
    "yx": YXSlicer,
    "x": XSlicer,
    "y": YSlicer,
    "z": ZSlicer,
}


def get_slicer(display_mode):
    """Retrieve a slicer from a given display mode.

    Parameters
    ----------
    display_mode : :obj:`str`
        The desired display mode.
        Possible options are:

            - "ortho": Three cuts are performed in orthogonal directions.
            - "tiled": Three cuts are performed and arranged in a 2x2 grid.
            - "mosaic": Three cuts are performed along multiple rows and
              columns.
            - "x": Sagittal
            - "y": Coronal
            - "z": Axial
            - "xz": Sagittal + Axial
            - "yz": Coronal + Axial
            - "yx": Coronal + Sagittal

    Returns
    -------
    slicer : An instance of one of the subclasses of\
    :class:`~nilearn.plotting.displays.BaseSlicer`

        The slicer corresponding to the requested display mode:

            - "ortho": Returns an
              :class:`~nilearn.plotting.displays.OrthoSlicer`.
            - "tiled": Returns a
              :class:`~nilearn.plotting.displays.TiledSlicer`.
            - "mosaic": Returns a
              :class:`~nilearn.plotting.displays.MosaicSlicer`.
            - "xz": Returns a
              :class:`~nilearn.plotting.displays.XZSlicer`.
            - "yz": Returns a
              :class:`~nilearn.plotting.displays.YZSlicer`.
            - "yx": Returns a
              :class:`~nilearn.plotting.displays.YZSlicer`.
            - "x": Returns a
              :class:`~nilearn.plotting.displays.XSlicer`.
            - "y": Returns a
              :class:`~nilearn.plotting.displays.YSlicer`.
            - "z": Returns a
              :class:`~nilearn.plotting.displays.ZSlicer`.

    """
    return _get_create_display_fun(display_mode, SLICERS)


def _get_create_display_fun(display_mode, class_dict):
    """Help for functions \
    :func:`~nilearn.plotting.displays.get_slicer` and \
    :func:`~nilearn.plotting.displays.get_projector`.
    """
    try:
        return class_dict[display_mode].init_with_figure
    except KeyError:
        message = (
            f"{display_mode} is not a valid display_mode. "
            f"Valid options are {sorted(class_dict.keys())}"
        )
        raise ValueError(message)
