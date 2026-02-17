"""Utilities for viewing images

Includes version of OrthoSlicer3D code originally written by our own
Paul Ivanov.
"""

import weakref

import numpy as np

from .affines import voxel_sizes
from .optpkg import optional_package
from .orientations import aff2axcodes, axcodes2ornt


class OrthoSlicer3D:
    """Orthogonal-plane slice viewer

    OrthoSlicer3d expects 3- or 4-dimensional array data.  It treats
    4D data as a sequence of 3D spatial volumes, where a slice over the final
    array axis gives a single 3D spatial volume.

    For 3D data, the default behavior is to create a figure with 3 axes, one
    for each slice orientation of the spatial volume.

    Clicking and dragging the mouse in any one axis will select out the
    corresponding slices in the other two. Scrolling up and
    down moves the slice up and down in the current axis.

    For 4D data, the fourth figure axis can be used to control which
    3D volume is displayed.  Alternatively, the ``-`` key can be used to
    decrement the displayed volume and the ``+`` or ``=`` keys can be used to
    increment it.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.sin(np.linspace(0, np.pi, 20))
    >>> b = np.sin(np.linspace(0, np.pi*5, 20))
    >>> data = np.outer(a, b)[..., np.newaxis] * a
    >>> OrthoSlicer3D(data).show()  # doctest: +SKIP
    """

    # Skip doctest above b/c not all systems have mpl installed

    def __init__(self, data, affine=None, axes=None, title=None):
        """
        Parameters
        ----------
        data : array-like
            The data that will be displayed by the slicer. Should have 3+
            dimensions.
        affine : array-like or None, optional
            Affine transform for the data. This is used to determine
            how the data should be sliced for plotting into the sagittal,
            coronal, and axial view axes. If None, identity is assumed.
            The aspect ratio of the data are inferred from the affine
            transform.
        axes : tuple of mpl.Axes or None, optional
            3 or 4 axes instances for the 3 slices plus volumes,
            or None (default).
        title : str or None, optional
            The title to display. Can be None (default) to display no
            title.
        """
        # Use these late imports of matplotlib so that we have some hope that
        # the test functions are the first to set the matplotlib backend. The
        # tests set the backend to something that doesn't require a display.
        self._plt = plt = optional_package('matplotlib.pyplot')[0]
        mpl_patch = optional_package('matplotlib.patches')[0]
        self._title = title
        self._closed = False
        self._cross = True

        data = np.asanyarray(data)
        if data.ndim < 3:
            raise ValueError('data must have at least 3 dimensions')
        if np.iscomplexobj(data):
            raise TypeError('Complex data not supported')
        affine = np.array(affine, float) if affine is not None else np.eye(4)
        if affine.shape != (4, 4):
            raise ValueError('affine must be a 4x4 matrix')
        # determine our orientation
        self._affine = affine
        codes = axcodes2ornt(aff2axcodes(self._affine))
        self._order = np.argsort([c[0] for c in codes])
        self._flips = np.array([c[1] < 0 for c in codes])[self._order]
        self._flips = list(self._flips) + [False]  # add volume dim
        self._scalers = voxel_sizes(self._affine)
        self._inv_affine = np.linalg.inv(affine)
        # current volume info
        self._volume_dims = data.shape[3:]
        self._current_vol_data = data[:, :, :, 0] if data.ndim > 3 else data
        self._data = data
        self._clim = np.percentile(data, (1.0, 99.0))
        del data

        if axes is None:  # make the axes
            # ^ +---------+   ^ +---------+
            # | |         |   | |         |
            #   |   Sag   |     |   Cor   |
            # S |    0    |   S |    1    |
            #   |         |     |         |
            #   |         |     |         |
            #   +---------+     +---------+
            #        A  -->          R  -->
            # ^ +---------+     +---------+
            # | |         |     |         |
            #   |  Axial  |     |   Vol   |
            # A |    2    |     |    3    |
            #   |         |     |         |
            #   |         |     |         |
            #   +---------+     +---------+
            #        R  -->     <--  t  -->

            fig, axes = plt.subplots(2, 2)
            fig.set_size_inches((8, 8), forward=True)
            self._axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
            plt.tight_layout(pad=0.1)
            if self.n_volumes <= 1:
                fig.delaxes(self._axes[3])
                self._axes.pop(-1)
            if self._title is not None:
                fig.canvas.manager.set_window_title(str(title))
        else:
            self._axes = [axes[0], axes[1], axes[2]]
            if len(axes) > 3:
                self._axes.append(axes[3])

        # Start midway through each axis, idx is current slice number
        self._ims, self._data_idx = list(), list()

        # set up axis crosshairs
        self._crosshairs = [None] * 3
        r = [
            self._scalers[self._order[2]] / self._scalers[self._order[1]],
            self._scalers[self._order[2]] / self._scalers[self._order[0]],
            self._scalers[self._order[1]] / self._scalers[self._order[0]],
        ]
        self._sizes = [self._data.shape[order] for order in self._order]
        for ii, xax, yax, ratio, label in zip(
            [0, 1, 2], [1, 0, 0], [2, 2, 1], r, ('SAIP', 'SRIL', 'ARPL')
        ):
            ax = self._axes[ii]
            d = np.zeros((self._sizes[yax], self._sizes[xax]))
            im = self._axes[ii].imshow(
                d,
                vmin=self._clim[0],
                vmax=self._clim[1],
                aspect=1,
                cmap='gray',
                interpolation='nearest',
                origin='lower',
            )
            self._ims.append(im)
            vert = ax.plot(
                [0] * 2, [-0.5, self._sizes[yax] - 0.5], color=(0, 1, 0), linestyle='-'
            )[0]
            horiz = ax.plot(
                [-0.5, self._sizes[xax] - 0.5], [0] * 2, color=(0, 1, 0), linestyle='-'
            )[0]
            self._crosshairs[ii] = dict(vert=vert, horiz=horiz)
            # add text labels (top, right, bottom, left)
            lims = [0, self._sizes[xax], 0, self._sizes[yax]]
            bump = 0.01
            poss = [
                [lims[1] / 2.0, lims[3]],
                [(1 + bump) * lims[1], lims[3] / 2.0],
                [lims[1] / 2.0, 0],
                [lims[0] - bump * lims[1], lims[3] / 2.0],
            ]
            anchors = [
                ['center', 'bottom'],
                ['left', 'center'],
                ['center', 'top'],
                ['right', 'center'],
            ]
            for pos, anchor, lab in zip(poss, anchors, label):
                ax.text(
                    pos[0], pos[1], lab, horizontalalignment=anchor[0], verticalalignment=anchor[1]
                )
            ax.axis(lims)
            ax.set_aspect(ratio)
            ax.patch.set_visible(False)
            ax.set_frame_on(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.axes.get_xaxis().set_visible(False)
            self._data_idx.append(0)
        self._data_idx.append(-1)  # volume

        # Set up volumes axis
        if self.n_volumes > 1 and len(self._axes) > 3:
            ax = self._axes[3]
            try:
                ax.set_facecolor('k')
            except AttributeError:  # old mpl
                ax.set_axis_bgcolor('k')
            ax.set_title('Volumes')
            y = np.zeros(self.n_volumes + 1)
            x = np.arange(self.n_volumes + 1) - 0.5
            step = ax.step(x, y, where='post', color='y')[0]
            ax.set_xticks(np.unique(np.linspace(0, self.n_volumes - 1, 5).astype(int)))
            ax.set_xlim(x[0], x[-1])
            yl = [self._data.min(), self._data.max()]
            yl = [lim + s * np.diff(lims)[0] for lim, s in zip(yl, [-1.01, 1.01])]
            patch = mpl_patch.Rectangle(
                [-0.5, yl[0]],
                1.0,
                np.diff(yl)[0],
                fill=True,
                facecolor=(0, 1, 0),
                edgecolor=(0, 1, 0),
                alpha=0.25,
            )
            ax.add_patch(patch)
            ax.set_ylim(yl)
            self._volume_ax_objs = dict(step=step, patch=patch)

        self._figs = {a.figure for a in self._axes}
        for fig in self._figs:
            fig.canvas.mpl_connect('scroll_event', self._on_scroll)
            fig.canvas.mpl_connect('motion_notify_event', self._on_mouse)
            fig.canvas.mpl_connect('button_press_event', self._on_mouse)
            fig.canvas.mpl_connect('key_press_event', self._on_keypress)
            fig.canvas.mpl_connect('close_event', self._cleanup)

        # actually set data meaningfully
        self._position = np.zeros(4)
        self._position[3] = 1.0  # convenience for affine multiplication
        self._changing = False  # keep track of status to avoid loops
        self._links = []  # other viewers this one is linked to
        self._plt.draw()
        for fig in self._figs:
            fig.canvas.draw()
        self._set_volume_index(0, update_slices=False)
        self._set_position(0.0, 0.0, 0.0)
        self._draw()

    def __repr__(self):
        title = '' if self._title is None else f'{self._title} '
        vol = '' if self.n_volumes <= 1 else f', {self.n_volumes}'
        r = (
            f'<{self.__class__.__name__}: {title}({self._sizes[0]}, '
            f'{self._sizes[1]}, {self._sizes[2]}{vol})>'
        )
        return r

    # User-level functions ###################################################
    def show(self):
        """Show the slicer in blocking mode; convenience for ``plt.show()``"""
        self._plt.show()

    def close(self):
        """Close the viewer figures"""
        self._cleanup()
        for f in self._figs:
            self._plt.close(f)

    def _cleanup(self):
        """Clean up before closing"""
        self._closed = True
        for link in list(self._links):  # make a copy before iterating
            self._unlink(link())

    def draw(self):
        """Redraw the current image"""
        for fig in self._figs:
            fig.canvas.draw()

    @property
    def n_volumes(self):
        """Number of volumes in the data"""
        return int(np.prod(self._volume_dims))

    @property
    def position(self):
        """The current coordinates"""
        return self._position[:3].copy()

    @property
    def figs(self):
        """A tuple of the figure(s) containing the axes"""
        return tuple(self._figs)

    @property
    def cmap(self):
        """The current colormap"""
        return self._cmap

    @cmap.setter
    def cmap(self, cmap):
        for im in self._ims:
            im.set_cmap(cmap)
        self._cmap = cmap
        self.draw()

    @property
    def clim(self):
        """The current color limits"""
        return self._clim

    @clim.setter
    def clim(self, clim):
        clim = np.array(clim, float)
        if clim.shape != (2,):
            raise ValueError('clim must be a 2-element array-like')
        for im in self._ims:
            im.set_clim(clim)
        self._clim = tuple(clim)
        self.draw()

    def link_to(self, other):
        """Link positional changes between two canvases

        Parameters
        ----------
        other : instance of OrthoSlicer3D
            Other viewer to use to link movements.
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f'other must be an instance of {self.__class__.__name__}, not {type(other)}'
            )
        self._link(other, is_primary=True)

    def _link(self, other, is_primary):
        """Link a viewer"""
        ref = weakref.ref(other)
        if ref in self._links:
            return
        self._links.append(ref)
        if is_primary:
            other._link(self, is_primary=False)
            other.set_position(*self.position)

    def _unlink(self, other):
        """Unlink a viewer"""
        ref = weakref.ref(other)
        if ref in self._links:
            self._links.pop(self._links.index(ref))
            ref()._unlink(self)

    def _notify_links(self):
        """Notify linked canvases of a position change"""
        for link in self._links:
            link().set_position(*self.position[:3])

    def set_position(self, x=None, y=None, z=None):
        """Set current displayed slice indices

        Parameters
        ----------
        x : float | None
            X coordinate to use. If None, do not change.
        y : float | None
            Y coordinate to use. If None, do not change.
        z : float | None
            Z coordinate to use. If None, do not change.
        """
        self._set_position(x, y, z)
        self._draw()

    def set_volume_idx(self, v):
        """Set current displayed volume index

        Parameters
        ----------
        v : int
            Volume index.
        """
        self._set_volume_index(v)
        self._draw()

    def _set_volume_index(self, v, update_slices=True):
        """Set the plot data using a volume index"""
        v = self._data_idx[3] if v is None else int(round(v))
        if v == self._data_idx[3]:
            return
        max_ = np.prod(self._volume_dims)
        self._data_idx[3] = max(min(int(round(v)), max_ - 1), 0)
        idx = (slice(None), slice(None), slice(None))
        if self._data.ndim > 3:
            idx = idx + tuple(np.unravel_index(self._data_idx[3], self._volume_dims))
        self._current_vol_data = self._data[idx]
        # update all of our slice plots
        if update_slices:
            self._set_position(None, None, None, notify=False)

    def _set_position(self, x, y, z, notify=True):
        """Set the plot data using a physical position"""
        # deal with volume first
        if self._changing:
            return
        self._changing = True
        x = self._position[0] if x is None else float(x)
        y = self._position[1] if y is None else float(y)
        z = self._position[2] if z is None else float(z)

        # deal with slicing appropriately
        self._position[:3] = [x, y, z]
        idxs = np.dot(self._inv_affine, self._position)[:3]
        idxs_new_order = idxs[self._order]
        for ii, (size, idx) in enumerate(zip(self._sizes, idxs_new_order)):
            self._data_idx[ii] = max(min(int(round(idx)), size - 1), 0)
        for ii in range(3):
            # sagittal: get to S/A
            # coronal: get to S/L
            # axial: get to A/L
            data = np.rollaxis(self._current_vol_data, axis=self._order[ii])[self._data_idx[ii]]
            xax = [1, 0, 0][ii]
            yax = [2, 2, 1][ii]
            if self._order[xax] < self._order[yax]:
                data = data.T
            if self._flips[xax]:
                data = data[:, ::-1]
            if self._flips[yax]:
                data = data[::-1]
            self._ims[ii].set_data(data)
            # deal with crosshairs
            loc = self._data_idx[ii]
            if self._flips[ii]:
                loc = self._sizes[ii] - 1 - loc
            loc = [loc] * 2
            if ii == 0:
                self._crosshairs[2]['vert'].set_xdata(loc)
                self._crosshairs[1]['vert'].set_xdata(loc)
            elif ii == 1:
                self._crosshairs[2]['horiz'].set_ydata(loc)
                self._crosshairs[0]['vert'].set_xdata(loc)
            else:  # ii == 2
                self._crosshairs[1]['horiz'].set_ydata(loc)
                self._crosshairs[0]['horiz'].set_ydata(loc)

        # Update volume trace
        if self.n_volumes > 1 and len(self._axes) > 3:
            idx = [slice(None)] * len(self._axes)
            for ii in range(3):
                idx[self._order[ii]] = self._data_idx[ii]
            vdata = self._data[tuple(idx)].ravel()
            vdata = np.concatenate((vdata, [vdata[-1]]))
            self._volume_ax_objs['patch'].set_x(self._data_idx[3] - 0.5)
            self._volume_ax_objs['step'].set_ydata(vdata)
        if notify:
            self._notify_links()
        self._changing = False

    # Matplotlib handlers ####################################################
    def _in_axis(self, event):
        """Return axis index if within one of our axes, else None"""
        if event.inaxes is None:
            return None
        for ii, ax in enumerate(self._axes):
            if event.inaxes is ax:
                return ii

    def _on_scroll(self, event):
        """Handle mpl scroll wheel event"""
        assert event.button in ('up', 'down')
        ii = self._in_axis(event)
        if ii is None:
            return
        if event.key is not None and 'shift' in event.key:
            if self.n_volumes <= 1:
                return
            ii = 3  # shift: change volume in any axis
        assert ii in range(4)
        dv = 10.0 if event.key is not None and 'control' in event.key else 1.0
        dv *= 1.0 if event.button == 'up' else -1.0
        dv *= -1 if self._flips[ii] else 1
        val = self._data_idx[ii] + dv

        if ii == 3:
            self._set_volume_index(val)
        else:
            coords = [self._data_idx[k] for k in range(3)]
            coords[ii] = val
            coords_ordered = [0, 0, 0, 1]
            for k in range(3):
                coords_ordered[self._order[k]] = coords[k]
            position = np.dot(self._affine, coords_ordered)[:3]
            self._set_position(*position)
        self._draw()

    def _on_mouse(self, event):
        """Handle mpl mouse move and button press events"""
        if event.button != 1:  # only enabled while dragging
            return
        ii = self._in_axis(event)
        if ii is None:
            return
        if ii == 3:
            # volume plot directly translates
            self._set_volume_index(event.xdata)
        else:
            # translate click xdata/ydata to physical position
            xax, yax = [
                [self._order[1], self._order[2]],
                [self._order[0], self._order[2]],
                [self._order[0], self._order[1]],
            ][ii]
            x, y = event.xdata, event.ydata
            x = self._sizes[xax] - x - 1 if self._flips[xax] else x
            y = self._sizes[yax] - y - 1 if self._flips[yax] else y
            idxs = np.ones(4)
            idxs[xax] = x
            idxs[yax] = y
            idxs[self._order[ii]] = self._data_idx[ii]
            self._set_position(*np.dot(self._affine, idxs)[:3])
        self._draw()

    def _on_keypress(self, event):
        """Handle mpl keypress events"""
        if event.key is not None and 'escape' in event.key:
            self.close()
        elif event.key in ('=', '+'):
            # increment volume index
            new_idx = min(self._data_idx[3] + 1, self.n_volumes)
            self._set_volume_index(new_idx, update_slices=True)
            self._draw()
        elif event.key == '-':
            # decrement volume index
            new_idx = max(self._data_idx[3] - 1, 0)
            self._set_volume_index(new_idx, update_slices=True)
            self._draw()
        elif event.key == 'ctrl+x':
            self._cross = not self._cross
            self._draw()

    def _draw(self):
        """Update all four (or three) plots"""
        if self._closed:  # make sure we don't draw when we shouldn't
            return
        for ii in range(3):
            ax = self._axes[ii]
            ax.draw_artist(self._ims[ii])
            if self._cross:
                for line in self._crosshairs[ii].values():
                    ax.draw_artist(line)
            ax.figure.canvas.blit(ax.bbox)
        if self.n_volumes > 1 and len(self._axes) > 3:
            ax = self._axes[3]
            ax.draw_artist(ax.patch)  # axis bgcolor to erase old lines
            for key in ('step', 'patch'):
                ax.draw_artist(self._volume_ax_objs[key])
            ax.figure.canvas.blit(ax.bbox)
