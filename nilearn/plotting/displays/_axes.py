import numbers

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrow
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

from nilearn._utils.docs import fill_doc
from nilearn.image import coord_transform
from nilearn.plotting.glass_brain import plot_brain_schematics


@fill_doc
class BaseAxes:
    """An MPL axis-like object that displays a 2D view of 3D volumes.

    Parameters
    ----------
    %(ax)s
    direction : {'x', 'y', 'z'}
        The directions of the view.

    coord : :obj:`float`
        The coordinate along the direction of the cut.
    %(radiological)s
    """

    def __init__(self, ax, direction, coord, radiological=False):
        self.ax = ax
        self.direction = direction
        self.coord = coord
        self._object_bounds = []
        self.shape = None
        self.radiological = radiological

    def transform_to_2d(self, data, affine):
        """Transform to a 2D."""
        raise NotImplementedError(
            "'transform_to_2d' needs to be implemented in derived classes'"
        )

    def add_object_bounds(self, bounds):
        """Ensure that axes get rescaled when adding object bounds."""
        old_object_bounds = self.get_object_bounds()
        self._object_bounds.append(bounds)
        new_object_bounds = self.get_object_bounds()

        if new_object_bounds != old_object_bounds:
            self.ax.axis(self.get_object_bounds())

    def draw_2d(
        self, data_2d, data_bounds, bounding_box, type="imshow", **kwargs
    ):
        """Draw 2D."""
        kwargs["origin"] = "upper"

        if self.direction == "y":
            (xmin, xmax), (_, _), (zmin, zmax) = data_bounds
            (xmin_, xmax_), (_, _), (zmin_, zmax_) = bounding_box
        elif self.direction in "xlr":
            (_, _), (xmin, xmax), (zmin, zmax) = data_bounds
            (_, _), (xmin_, xmax_), (zmin_, zmax_) = bounding_box
        elif self.direction == "z":
            (xmin, xmax), (zmin, zmax), (_, _) = data_bounds
            (xmin_, xmax_), (zmin_, zmax_), (_, _) = bounding_box
        else:
            raise ValueError(f"Invalid value for direction {self.direction}")
        ax = self.ax
        # Here we need to do a copy to avoid having the image changing as
        # we change the data
        im = getattr(ax, type)(
            data_2d.copy(), extent=(xmin, xmax, zmin, zmax), **kwargs
        )

        self.add_object_bounds((xmin_, xmax_, zmin_, zmax_))
        self.shape = data_2d.T.shape
        # The bounds of the object do not take into account a possible
        # inversion of the axis. As such, we check that the axis is properly
        # inverted when direction is left
        if self.direction == "l" and not (ax.get_xlim()[0] > ax.get_xlim()[1]):
            ax.invert_xaxis()
        return im

    def get_object_bounds(self):
        """Return the bounds of the objects on this axes."""
        if len(self._object_bounds) == 0:
            # Nothing plotted yet
            return -0.01, 0.01, -0.01, 0.01
        xmins, xmaxs, ymins, ymaxs = np.array(self._object_bounds).T
        xmax = max(xmaxs.max(), xmins.max())
        xmin = min(xmins.min(), xmaxs.min())
        ymax = max(ymaxs.max(), ymins.max())
        ymin = min(ymins.min(), ymaxs.min())

        return xmin, xmax, ymin, ymax

    def draw_left_right(self, size, bg_color, **kwargs):
        """Draw the annotation "L" for left, and "R" for right.

        Parameters
        ----------
        size : :obj:`float`, optional
            Size of the text areas.

        bg_color : matplotlib color: :obj:`str` or (r, g, b) value
            The background color for both text areas.

        """
        if self.direction in "xlr":
            return
        ax = self.ax
        annotation_on_left = "L"
        annotation_on_right = "R"
        if self.radiological:
            annotation_on_left = "R"
            annotation_on_right = "L"
        ax.text(
            0.1,
            0.95,
            annotation_on_left,
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            size=size,
            bbox={
                "boxstyle": "square,pad=0",
                "ec": bg_color,
                "fc": bg_color,
                "alpha": 1,
            },
            **kwargs,
        )

        ax.text(
            0.9,
            0.95,
            annotation_on_right,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            size=size,
            bbox={"boxstyle": "square,pad=0", "ec": bg_color, "fc": bg_color},
            **kwargs,
        )

    def draw_scale_bar(
        self,
        bg_color,
        size=5.0,
        units="cm",
        fontproperties=None,
        frameon=False,
        loc=4,
        pad=0.1,
        borderpad=0.5,
        sep=5,
        size_vertical=0,
        label_top=False,
        color="black",
        fontsize=None,
        **kwargs,
    ):
        """Add a scale bar annotation to the display.

        Parameters
        ----------
        bg_color : matplotlib color: :obj:`str` or (r, g, b) value
            The background color of the scale bar annotation.

        size : :obj:`float`, default=5.0
            Horizontal length of the scale bar, given in `units`.


        units : :obj:`str`, default='cm'
            Physical units of the scale bar (`'cm'` or `'mm'`).


        fontproperties : :class:`~matplotlib.font_manager.FontProperties`\
        or :obj:`dict`, optional
            Font properties for the label text.

        frameon : :obj:`bool`, default=False
            Whether the scale bar is plotted with a border.

        loc : :obj:`int`, default=4
            Location of this scale bar.
            Valid location codes are documented in
            :class:`~mpl_toolkits.axes_grid1.anchored_artists.AnchoredSizeBar`

        pad : :obj:`int` or :obj:`float`, default=0.1
            Padding around the label and scale bar, in fraction of the font
            size.

        borderpad : :obj:`int` or :obj:`float`, default=0.5
            Border padding, in fraction of the font size.

        sep : :obj:`int` or :obj:`float`, default=5
            Separation between the label and the scale bar, in points.


        size_vertical : :obj:`int` or :obj:`float`, default=0
            Vertical length of the size bar, given in `units`.


        label_top : :obj:`bool`, default=False
            If ``True``, the label will be over the scale bar.


        color : :obj:`str`, default='black'
            Color for the scale bar and label.

        fontsize : :obj:`int`, optional
            Label font size (overwrites the size passed in through the
            ``fontproperties`` argument).

        **kwargs :
            Keyworded arguments to pass to
            :class:`~matplotlib.offsetbox.AnchoredOffsetbox`.

        """
        axis = self.ax
        fontproperties = fontproperties or FontProperties()
        if fontsize:
            fontproperties.set_size(fontsize)
        width_mm = size
        if units == "cm":
            width_mm *= 10

        anchor_size_bar = AnchoredSizeBar(
            axis.transData,
            width_mm,
            f"{size:g}{units}",
            fontproperties=fontproperties,
            frameon=frameon,
            loc=loc,
            pad=pad,
            borderpad=borderpad,
            sep=sep,
            size_vertical=size_vertical,
            label_top=label_top,
            color=color,
            **kwargs,
        )

        if frameon:
            anchor_size_bar.patch.set_facecolor(bg_color)
            anchor_size_bar.patch.set_edgecolor("none")
        axis.add_artist(anchor_size_bar)

    def draw_position(self, size, bg_color, **kwargs):
        """``draw_position`` is not implemented in base class and \
        should be implemented in derived classes.
        """
        raise NotImplementedError(
            "'draw_position' should be implemented in derived classes"
        )


@fill_doc
class CutAxes(BaseAxes):
    """An MPL axis-like object that displays a cut of 3D volumes.

    Parameters
    ----------
    %(ax)s
    direction : {'x', 'y', 'z'}
        The directions of the view.

    coord : :obj:`float`
        The coordinate along the direction of the cut.
    """

    def transform_to_2d(self, data, affine):
        """Cut the 3D volume into a 2D slice.

        Parameters
        ----------
        data : 3D :class:`~numpy.ndarray`
            The 3D volume to cut.

        affine : 4x4 :class:`~numpy.ndarray`
            The affine of the volume.

        """
        coords = [0, 0, 0]
        if self.direction not in ["x", "y", "z"]:
            raise ValueError(f"Invalid value for direction {self.direction}")
        coords["xyz".index(self.direction)] = self.coord
        x_map, y_map, z_map = (
            int(np.round(c))
            for c in coord_transform(
                coords[0], coords[1], coords[2], np.linalg.inv(affine)
            )
        )
        if self.direction == "y":
            cut = np.rot90(data[:, y_map, :])
        elif self.direction == "x":
            cut = np.rot90(data[x_map, :, :])
        elif self.direction == "z":
            cut = np.rot90(data[:, :, z_map])
        return cut

    def draw_position(self, size, bg_color, decimals=False, **kwargs):
        """Draw coordinates.

        Parameters
        ----------
        size : :obj:`float`, optional
            Size of the text area.

        bg_color : matplotlib color: :obj:`str` or (r, g, b) value
            The background color for text area.

        decimals : :obj:`bool` or :obj:`str`, default=False
            Formatting string for the coordinates.
            If set to ``False``, integer formatting will be used.


        """
        if decimals:
            text = f"%s=%.{decimals}f"
            coord = float(self.coord)
        else:
            text = "%s=%i"
            coord = self.coord
        ax = self.ax
        ax.text(
            0,
            0,
            text % (self.direction, coord),
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="bottom",
            size=size,
            bbox={
                "boxstyle": "square,pad=0",
                "ec": bg_color,
                "fc": bg_color,
                "alpha": 1,
            },
            **kwargs,
        )


def _get_index_from_direction(direction):
    """Return numerical index from direction."""
    directions = ["x", "y", "z"]
    try:
        # l and r are subcases of x
        index = 0 if direction in "lr" else directions.index(direction)
    except ValueError:
        message = (
            f"{direction} is not a valid direction. "
            "Allowed values are 'l', 'r', 'x', 'y' and 'z'"
        )
        raise ValueError(message)
    return index


def coords_3d_to_2d(coords_3d, direction, return_direction=False):
    """Project 3d coordinates into 2d ones given the direction of a cut."""
    index = _get_index_from_direction(direction)
    dimensions = [0, 1, 2]
    dimensions.pop(index)
    if return_direction:
        return coords_3d[:, dimensions], coords_3d[:, index]
    return coords_3d[:, dimensions]


@fill_doc
class GlassBrainAxes(BaseAxes):
    """An MPL axis-like object that displays a 2D projection of 3D \
    volumes with a schematic view of the brain.

    Parameters
    ----------
    %(ax)s
    direction : {'x', 'y', 'z'}
        The directions of the view.

    coord : :obj:`float`
        The coordinate along the direction of the cut.

    plot_abs : :obj:`bool`, default=True
        If set to ``True`` the absolute value of the data will be considered.

    """

    def __init__(
        self, ax, direction, coord, plot_abs=True, radiological=False, **kwargs
    ):
        super().__init__(ax, direction, coord, radiological=radiological)
        self._plot_abs = plot_abs
        if ax is not None:
            object_bounds = plot_brain_schematics(ax, direction, **kwargs)
            self.add_object_bounds(object_bounds)

    def transform_to_2d(self, data, affine):
        """Return the maximum of the absolute value of the 3D volume \
        along an axis.

        Parameters
        ----------
        data : 3D :class:`numpy.ndarray`
            The 3D volume.

        affine : 4x4 :class:`numpy.ndarray`
            The affine of the volume.

        """
        max_axis = (
            0 if self.direction in "xlr" else ".yz".index(self.direction)
        )
        # set unselected brain hemisphere activations to 0
        if self.direction == "l":
            x_center, _, _, _ = np.dot(
                np.linalg.inv(affine), np.array([0, 0, 0, 1])
            )
            data_selection = data[: int(x_center), :, :]
        elif self.direction == "r":
            x_center, _, _, _ = np.dot(
                np.linalg.inv(affine), np.array([0, 0, 0, 1])
            )
            data_selection = data[int(x_center) :, :, :]
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
        if (
            self.direction == "l"
            and data_selection.min() is np.ma.masked
            and not (self.ax.get_xlim()[0] > self.ax.get_xlim()[1])
        ):
            self.ax.invert_xaxis()

        return np.rot90(maximum_intensity_data)

    def draw_position(self, size, bg_color, **kwargs):
        """Not implemented as it does not make sense to draw crosses for \
        the position of the cuts \
        since we are taking the max along one axis.
        """
        pass

    def _add_markers(self, marker_coords, marker_color, marker_size, **kwargs):
        """Plot markers.

        In the case of 'l' and 'r' directions (for hemispheric projections),
        markers in the coordinate x == 0 are included in both hemispheres.
        """
        marker_coords_2d = coords_3d_to_2d(marker_coords, self.direction)
        xdata, ydata = marker_coords_2d.T

        # Allow markers only in their respective hemisphere when appropriate
        if self.direction in "lr":
            if not isinstance(marker_color, str) and not isinstance(
                marker_color, np.ndarray
            ):
                marker_color = np.asarray(marker_color)
            relevant_coords = []
            xcoords, ycoords, zcoords = marker_coords.T
            relevant_coords.extend(
                cidx
                for cidx, xc in enumerate(xcoords)
                if (self.direction == "r" and xc >= 0)
                or (self.direction == "l" and xc <= 0)
            )
            xdata = xdata[relevant_coords]
            ydata = ydata[relevant_coords]
            # if marker_color is string for example 'red' or 'blue', then
            # we pass marker_color as it is to matplotlib scatter without
            # making any selection in 'l' or 'r' color.
            # More likely that user wants to display all nodes to be in
            # same color.
            if not isinstance(marker_color, str) and len(marker_color) != 1:
                marker_color = marker_color[relevant_coords]

            if not isinstance(marker_size, numbers.Number):
                marker_size = np.asarray(marker_size)[relevant_coords]

        defaults = {"marker": "o", "zorder": 1000}
        for k, v in defaults.items():
            kwargs.setdefault(k, v)

        self.ax.scatter(xdata, ydata, s=marker_size, c=marker_color, **kwargs)

    def _add_lines(
        self,
        line_coords,
        line_values,
        cmap,
        vmin=None,
        vmax=None,
        directed=False,
        **kwargs,
    ):
        """Plot lines.

        Parameters
        ----------
        line_coords : :obj:`list` of :class:`numpy.ndarray` of shape (2, 3)
            3D coordinates of lines start points and end points.

        line_values : array_like
            Values of the lines.

        cmap : :class:`~matplotlib.colors.Colormap`
            Colormap used to map ``line_values`` to a color.

        vmin, vmax : :obj:`float`, optional
            If not ``None``, either or both of these values will be used to
            as the minimum and maximum values to color lines. If ``None`` are
            supplied the maximum absolute value within the given threshold
            will be used as minimum (multiplied by -1) and maximum
            coloring levels.

        directed : :obj:`bool`, default=False
            Add arrows instead of lines if set to ``True``.
            Use this when plotting directed graphs for example.


        kwargs : :obj:`dict`
            Additional arguments to pass to :class:`~matplotlib.lines.Line2D`.

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
        norm = Normalize(vmin=vmin, vmax=vmax)
        # normalization useful for colorbar
        self.norm = norm
        abs_norm = Normalize(vmin=0, vmax=max(abs(vmax), abs(vmin)))
        value_to_color = plt.cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba

        # Allow lines only in their respective hemisphere when appropriate
        if self.direction in "lr":
            relevant_lines = [
                lidx
                for lidx, line in enumerate(line_coords)
                if (
                    self.direction == "r"
                    and line[0, 0] >= 0
                    and line[1, 0] >= 0
                )
                or (
                    self.direction == "l" and line[0, 0] < 0 and line[1, 0] < 0
                )
            ]
            line_coords = np.array(line_coords)[relevant_lines]
            line_values = line_values[relevant_lines]

        for start_end_point_3d, line_value in zip(line_coords, line_values):
            start_end_point_2d = coords_3d_to_2d(
                start_end_point_3d, self.direction
            )

            color = value_to_color(line_value)
            abs_line_value = abs(line_value)
            linewidth = 1 + 2 * abs_norm(abs_line_value)
            # Hacky way to put the strongest connections on top of the weakest
            # note sign does not matter hence using 'abs'
            zorder = 10 + 10 * abs_norm(abs_line_value)
            this_kwargs = {
                "color": color,
                "linewidth": linewidth,
                "zorder": zorder,
            }
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
                if dx == dy == 0:
                    arrow = FancyArrow(xdata[0], ydata[0], dx, dy)
                else:
                    arrow = FancyArrow(
                        xdata[0],
                        ydata[0],
                        dx,
                        dy,
                        length_includes_head=True,
                        width=linewidth,
                        head_width=3 * linewidth,
                        **this_kwargs,
                    )
                self.ax.add_patch(arrow)
            # Otherwise a line
            else:
                line = Line2D(xdata, ydata, **this_kwargs)
                self.ax.add_line(line)
