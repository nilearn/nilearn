"""
Functions for surface visualization.
Only matplotlib is required.
"""
import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

from matplotlib.colorbar import make_axes
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap

from ..datasets import fetch_surf_fsaverage
from ..surface import load_surf_data, load_surf_mesh, vol_to_surf
from .._utils.compat import _basestring
from .img_plotting import _get_colorbar_and_data_ranges, _crop_colorbar


def plot_surf(surf_mesh, surf_map=None, bg_map=None,
              hemi='left', view='lateral', cmap=None, colorbar=False,
              avg_method='mean', threshold=None, alpha='auto',
              bg_on_data=False, darkness=1, vmin=None, vmax=None,
              cbar_vmin=None, cbar_vmax=None,
              title=None, output_file=None, axes=None, figure=None, **kwargs):
    """ Plotting of surfaces with optional background and data

    .. versionadded:: 0.3

    Parameters
    ----------
    surf_mesh: str or list of two numpy.ndarray
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces.

    surf_map: str or numpy.ndarray, optional.
        Data to be displayed on the surface mesh. Can be a file (valid formats
        are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .curv, .sulc, .annot, .label) or
        a Numpy array

    bg_map: Surface data object (to be defined), optional,
        Background image to be plotted on the mesh underneath the
        surf_data in greyscale, most likely a sulcal depth map for
        realistic shading.

    hemi : {'left', 'right'}, default is 'left'
        Hemisphere to display.

    view: {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'}, default is 'lateral'
        View of the surface that is rendered.

    cmap: matplotlib colormap, str or colormap object, default is None
        To use for plotting of the stat_map. Either a string
        which is a name of a matplotlib colormap, or a matplotlib
        colormap object. If None, matplolib default will be chosen

    colorbar : bool, optional, default is False
        If True, a colorbar of surf_map is displayed.

    avg_method: {'mean', 'median'}, default is 'mean'
        How to average vertex values to derive the face value, mean results
        in smooth, median in sharp boundaries.

    threshold : a number, None, or 'auto', default is None.
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image, values
        below the threshold (in absolute value) are plotted as transparent.

    alpha: float, alpha level of the mesh (not surf_data), default 'auto'
        If 'auto' is chosen, alpha will default to .5 when no bg_map
        is passed and to 1 if a bg_map is passed.

    bg_on_stat: bool, default is False
        If True, and a bg_map is specified, the surf_data data is multiplied
        by the background image, so that e.g. sulcal depth is visible beneath
        the surf_data.
        NOTE: that this non-uniformly changes the surf_data values according
        to e.g the sulcal depth.

    darkness: float, between 0 and 1, default is 1
        Specifying the darkness of the background image.
        1 indicates that the original values of the background are used.
        .5 indicates the background values are reduced by half before being
        applied.

    vmin, vmax: lower / upper bound to plot surf_data values
        If None , the values will be set to min/max of the data

    title : str, optional
        Figure title.

    output_file: str, or None, optional
        The name of an image file to export plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.

    axes: instance of matplotlib axes, None, optional
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, a new axes is created.

    figure: instance of matplotlib figure, None, optional
        The figure instance to plot to. If None, a new figure is created.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf_roi : For plotting statistical maps on brain
        surfaces.

    nilearn.plotting.plot_surf_stat_map for plotting statistical maps on
        brain surfaces.
    """

    # load mesh and derive axes limits
    mesh = load_surf_mesh(surf_mesh)
    coords, faces = mesh[0], mesh[1]
    limits = [coords.min(), coords.max()]

    # set view
    if hemi == 'right':
        if view == 'lateral':
            elev, azim = 0, 0
        elif view == 'medial':
            elev, azim = 0, 180
        elif view == 'dorsal':
            elev, azim = 90, 0
        elif view == 'ventral':
            elev, azim = 270, 0
        elif view == 'anterior':
            elev, azim = 0, 90
        elif view == 'posterior':
            elev, azim = 0, 270
        else:
            raise ValueError('view must be one of lateral, medial, '
                             'dorsal, ventral, anterior, or posterior')
    elif hemi == 'left':
        if view == 'medial':
            elev, azim = 0, 0
        elif view == 'lateral':
            elev, azim = 0, 180
        elif view == 'dorsal':
            elev, azim = 90, 0
        elif view == 'ventral':
            elev, azim = 270, 0
        elif view == 'anterior':
            elev, azim = 0, 90
        elif view == 'posterior':
            elev, azim = 0, 270
        else:
            raise ValueError('view must be one of lateral, medial, '
                             'dorsal, ventral, anterior, or posterior')
    else:
        raise ValueError('hemi must be one of right or left')

    # set alpha if in auto mode
    if alpha == 'auto':
        if bg_map is None:
            alpha = .5
        else:
            alpha = 1

    # if no cmap is given, set to matplotlib default
    if cmap is None:
        cmap = plt.cm.get_cmap(plt.rcParamsDefault['image.cmap'])
    else:
        # if cmap is given as string, translate to matplotlib cmap
        if isinstance(cmap, _basestring):
            cmap = plt.cm.get_cmap(cmap)

    # initiate figure and 3d axes
    if axes is None:
        if figure is None:
            figure = plt.figure()
        axes = Axes3D(figure, rect=[0, 0, 1, 1],
                      xlim=limits, ylim=limits)
    else:
        if figure is None:
            figure = axes.get_figure()
        axes.set_xlim(*limits)
        axes.set_ylim(*limits)
    axes.set_aspect(.74)
    axes.view_init(elev=elev, azim=azim)
    axes.set_axis_off()

    # plot mesh without data
    p3dcollec = axes.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                  triangles=faces, linewidth=0.,
                                  antialiased=False,
                                  color='white')

    # reduce viewing distance to remove space around mesh
    axes.dist = 8

    # If depth_map and/or surf_map are provided, map these onto the surface
    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    if bg_map is not None or surf_map is not None:

        face_colors = np.ones((faces.shape[0], 4))
        # face_colors[:, :3] = .5*face_colors[:, :3]  # why this?

        if bg_map is not None:
            bg_data = load_surf_data(bg_map)
            if bg_data.shape[0] != coords.shape[0]:
                raise ValueError('The bg_map does not have the same number '
                                 'of vertices as the mesh.')
            bg_faces = np.mean(bg_data[faces], axis=1)
            bg_faces = bg_faces - bg_faces.min()
            bg_faces = bg_faces / bg_faces.max()
            # control background darkness
            bg_faces *= darkness
            face_colors = plt.cm.gray_r(bg_faces)

        # modify alpha values of background
        face_colors[:, 3] = alpha * face_colors[:, 3]
        # should it be possible to modify alpha of surf data as well?

        if surf_map is not None:
            surf_map_data = load_surf_data(surf_map)
            if len(surf_map_data.shape) is not 1:
                raise ValueError('surf_map can only have one dimension but has'
                                 '%i dimensions' % len(surf_map_data.shape))
            if surf_map_data.shape[0] != coords.shape[0]:
                raise ValueError('The surf_map does not have the same number '
                                 'of vertices as the mesh.')

            # create face values from vertex values by selected avg methods
            if avg_method == 'mean':
                surf_map_faces = np.mean(surf_map_data[faces], axis=1)
            elif avg_method == 'median':
                surf_map_faces = np.median(surf_map_data[faces], axis=1)

            # if no vmin/vmax are passed figure them out from data
            if vmin is None:
                vmin = np.nanmin(surf_map_faces)
            if vmax is None:
                vmax = np.nanmax(surf_map_faces)

            # treshold if inidcated
            if threshold is None:
                kept_indices = np.where(surf_map_faces)[0]
            else:
                kept_indices = np.where(np.abs(surf_map_faces) >= threshold)[0]

            surf_map_faces = surf_map_faces - vmin
            surf_map_faces = surf_map_faces / (vmax - vmin)

            # multiply data with background if indicated
            if bg_on_data:
                face_colors[kept_indices] = cmap(surf_map_faces[kept_indices])\
                    * face_colors[kept_indices]
            else:
                face_colors[kept_indices] = cmap(surf_map_faces[kept_indices])

            if colorbar:
                our_cmap = get_cmap(cmap)
                norm = Normalize(vmin=vmin, vmax=vmax)

                nb_ticks = 5
                ticks = np.linspace(vmin, vmax, nb_ticks)
                bounds = np.linspace(vmin, vmax, our_cmap.N)

                if threshold is not None:
                    cmaplist = [our_cmap(i) for i in range(our_cmap.N)]
                    # set colors to grey for absolute values < threshold
                    istart = int(norm(-threshold, clip=True) *
                                 (our_cmap.N - 1))
                    istop = int(norm(threshold, clip=True) *
                                (our_cmap.N - 1))
                    for i in range(istart, istop):
                        cmaplist[i] = (0.5, 0.5, 0.5, 1.)
                    our_cmap = LinearSegmentedColormap.from_list(
                        'Custom cmap', cmaplist, our_cmap.N)

                # we need to create a proxy mappable
                proxy_mappable = ScalarMappable(cmap=our_cmap, norm=norm)
                proxy_mappable.set_array(surf_map_faces)
                cax, kw = make_axes(axes, location='right', fraction=.1,
                                    shrink=.6, pad=.0)
                cbar = figure.colorbar(
                    proxy_mappable, cax=cax, ticks=ticks,
                    boundaries=bounds, spacing='proportional',
                    format='%.2g', orientation='vertical')
                _crop_colorbar(cbar, cbar_vmin, cbar_vmax)

        p3dcollec.set_facecolors(face_colors)

    if title is not None:
        axes.set_title(title, position=(.5, .95))

    # save figure if output file is given
    if output_file is not None:
        figure.savefig(output_file)
        plt.close(figure)
    else:
        return figure


def plot_surf_stat_map(surf_mesh, stat_map, bg_map=None,
                       hemi='left', view='lateral', threshold=None,
                       alpha='auto', vmax=None, cmap='cold_hot',
                       colorbar=True, symmetric_cbar="auto", bg_on_data=False,
                       darkness=1, title=None, output_file=None, axes=None,
                       figure=None, **kwargs):
    """ Plotting a stats map on a surface mesh with optional background

    .. versionadded:: 0.3

    Parameters
    ----------
    surf_mesh : str or list of two numpy.ndarray
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z
        coordinates of the mesh vertices, the second containing the
        indices (into coords) of the mesh faces

    stat_map : str or numpy.ndarray
        Statistical map to be displayed on the surface mesh, can
        be a file (valid formats are .gii, .mgz, .nii, .nii.gz, or
        Freesurfer specific files such as .thickness, .curv, .sulc, .annot,
        .label) or
        a Numpy array

    bg_map : Surface data object (to be defined), optional,
        Background image to be plotted on the mesh underneath the
        stat_map in greyscale, most likely a sulcal depth map for
        realistic shading.

    hemi : {'left', 'right'}, default is 'left'
        Hemispere to display.

    view: {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'}, default is 'lateral'
        View of the surface that is rendered.

    threshold : a number, None, or 'auto', default is None
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image,
        values below the threshold (in absolute value) are plotted
        as transparent.

    cmap : matplotlib colormap in str or colormap object, default 'coolwarm'
        To use for plotting of the stat_map. Either a string
        which is a name of a matplotlib colormap, or a matplotlib
        colormap object.

    colorbar : bool, optional, default is False
        If True, a symmetric colorbar of the statistical map is displayed.

    alpha : float, alpha level of the mesh (not the stat_map), default 'auto'
        If 'auto' is chosen, alpha will default to .5 when no bg_map is
        passed and to 1 if a bg_map is passed.

    vmax : upper bound for plotting of stat_map values.

    symmetric_cbar : bool or 'auto', optional, default 'auto'
        Specifies whether the colorbar should range from -vmax to vmax
        or from vmin to vmax. Setting to 'auto' will select the latter
        if the range of the whole image is either positive or negative.
        Note: The colormap will always range from -vmax to vmax.

    bg_on_data : bool, default is False
        If True, and a bg_map is specified, the stat_map data is multiplied
        by the background image, so that e.g. sulcal depth is visible beneath
        the stat_map.
        NOTE: that this non-uniformly changes the stat_map values according
        to e.g the sulcal depth.

    darkness: float, between 0 and 1, default 1
        Specifying the darkness of the background image. 1 indicates that the
        original values of the background are used. .5 indicates the
        background values are reduced by half before being applied.

    title : str, optional
        Figure title.

    output_file: str, or None, optional
        The name of an image file to export plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.

    axes: instance of matplotlib axes, None, optional
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, a new axes is created.

    figure: instance of matplotlib figure, None, optional
        The figure instance to plot to. If None, a new figure is created.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf : For brain surface visualization.
    """

    # Call _get_colorbar_and_data_ranges to derive symmetric vmin, vmax
    # And colorbar limits depending on symmetric_cbar settings
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        stat_map, vmax, symmetric_cbar, kwargs)

    display = plot_surf(
        surf_mesh, surf_map=stat_map, bg_map=bg_map, hemi=hemi, view=view,
        avg_method='mean', threshold=threshold, cmap=cmap, colorbar=colorbar,
        alpha=alpha, bg_on_data=bg_on_data, darkness=1, vmax=vmax, vmin=vmin,
        title=title, output_file=output_file, axes=axes, figure=figure,
        cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax, **kwargs)

    return display


def _check_display_mode(display_mode):
    """Checks whether the display_mode string passed to plot_surf_montage
    is meaningful.

    display_mode: str
        Any combination of 'lateral', 'medial', 'dorsal', 'ventral',
        'anterior', 'posterior', united by a '+'. Ex.: 'dorsal+lateral'
    """

    available_modes = {'lateral', 'medial',
                       'dorsal', 'ventral',
                       'anterior', 'posterior'}

    desired_modes = display_mode.split('+')

    wrong_modes = set(desired_modes).difference(available_modes)
    if wrong_modes:
        msg = 'display_mode given: {}, available modes: {}'
        msg = msg.format(wrong_modes, available_modes)
        raise ValueError(msg)

    return desired_modes


def _check_hemisphere(hemisphere):
    """Checks whether the hemisphere in plot_surf_montage is correct.

    display_mode: str
        Any combination of 'left', 'right'
    """
    desired_hemispheres = hemisphere.split('+')
    wrong_hemisphere = set(desired_hemispheres).difference({'left', 'right'})
    if wrong_hemisphere:
        raise ValueError('hemisphere only accepts left and right.')
    if len(desired_hemispheres) > 2:
        raise ValueError('Currently only supports plotting 2 hemispheres.')

    return desired_hemispheres


def plot_surf_montage(stat_map, surf_mesh=None, mask_img=None,
                      hemisphere='left+right',
                      inflate=False, display_mode='lateral+medial',
                      **kwargs):
    """Convenience function to plot multiple views of plot_surf_stat_map
    in a single figure. It projects stat_map into meshes and plots views of
    left and right hemispheres. The display_mode argument defines the views
    that are shown. This function returns the fig, axes elements from
    matplotlib unless kwargs sets and output_file, in which case nothing
    is returned.

    stat_map : str or Niimg-like object, 3d.
        See http://nilearn.github.io/manipulating_images/input_output.html

    surf_mesh: dictionary, or None, optional (default=None)
        A dictionary with keys ['infl_left', 'infl_right',
                                'pial_left', 'pial_right',
                                'sulc_left', 'sulc_right'], where
       values are surface mesh geometries as accepted by plot_surf_stat_map.
       If None, plot_surf_montage will use Freesurfer's fsaverage.

    mask_img : Niimg-like object or None, optional (default=None)
        The mask is passed to vol_to_surf.
        Samples falling out of this mask or out of the image are ignored
        during projection of the volume to the surface.
        If ``None``, don't apply any mask.

    inflate: bool, optional (default=False)
        If True, display images in inflated brain.
        If False, display images in pial surface.

    display_mode: str, optional (default='lateral+medial')
        A string containing all views to display, separated by '+'.
        Available views: {'lateral', 'medial',
                           'dorsal', 'ventral',
                           'anterior', 'posterior'}
        The montage will contain as many rows as views specified by
        display mode. Order is preserved, and left and right hemispheres
        are shown on the left and right sides of the figure.

    kwargs: keyword arguments passed to plot_surf_stat_map

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as the default background map for this plotting function.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.

    nilearn.plotting.plot_surf_stat_map : For info on kwargs options
    accepted by plot_surf_montage.
    """
    for arg in ('figure', 'axes'):
        if arg in kwargs:
            raise ValueError(('plot_surf_montage does not'
                              ' accept %s as an argument' % arg))

    modes = _check_display_mode(display_mode)
    hemis = _check_hemisphere(hemisphere)

    # Get kwargs that behave differently in the montage
    # out of the dict and into the local scope. Do not pass them downstream.
    output_file = kwargs.pop('output_file', None)
    title = kwargs.pop('title', None)
    colorbar = kwargs.pop('colorbar', False)
    vmax = kwargs.pop('vmax', None)
    threshold = kwargs.pop('threshold', 0.)
    symmetric_cbar = kwargs.pop('symmetric_cbar', 'auto')

    if surf_mesh is None:
        surf_mesh = fetch_surf_fsaverage()
    else:
        req_keys = {'infl_left', 'infl_right',
                    'pial_left', 'pial_right',
                    'sulc_left', 'sulc_right'}
        assert all(key in surf_mesh.keys() for key in req_keys), \
            'MESHES argument requires keys %s.' % req_keys

    surf = {
        'left': surf_mesh['infl_left'] if inflate else surf_mesh['pial_left'],
        'right': surf_mesh['infl_right'] if inflate else surf_mesh['pial_right']
    }

    texture = {
        'left': vol_to_surf(stat_map, surf_mesh['pial_left'],
                            mask_img=mask_img),
        'right': vol_to_surf(stat_map, surf_mesh['pial_right'],
                             mask_img=mask_img)
    }

    fig, axes = plt.subplots(nrows=len(modes), ncols=len(hemis),
                             figsize=plt.figaspect(len(modes) / 2),
                             subplot_kw={'projection': '3d'})

    axes = np.atleast_2d(axes)

    if len(hemis) == 1:
        axes = axes.T

    for index_mode, mode in enumerate(modes):
        for index_hemi, hemi in enumerate(hemis):
            bg_map = surf_mesh['sulc_%s' % hemi]
            plot_surf_stat_map(surf[hemi], texture[hemi],
                               view=mode, hemi=hemi,
                               bg_map=bg_map,
                               axes=axes[index_mode, index_hemi],
                               colorbar=False,
                               vmax=vmax,
                               threshold=threshold,
                               symmetric_cbar=symmetric_cbar,
                               **kwargs)

    for ax in axes.flatten():
        # We increase this value to better position the camera of the
        # 3D projection plot. The default value makes meshes look too small.
        ax.dist = 6

    # Add colorbar
    if colorbar:

        cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
            stat_map.get_data(), vmax, symmetric_cbar, kwargs)

        if 'cmap' in kwargs:
            cmap = get_cmap([kwargs['cmap']])
        else:
            cmap = get_cmap('cold_hot')

        norm = Normalize(vmin=vmin, vmax=vmax)
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # set colors to grey for absolute values < threshold
        istart = int(norm(-threshold, clip=True) * (cmap.N - 1))
        istop = int(norm(threshold, clip=True) * (cmap.N - 1))
        for i in range(istart, istop):
            cmaplist[i] = (0.5, 0.5, 0.5, 1.)
        our_cmap = LinearSegmentedColormap.from_list('Custom cmap',
                                                     cmaplist, cmap.N)
        sm = plt.cm.ScalarMappable(cmap=our_cmap,
                                   norm=plt.Normalize(vmin=vmin, vmax=vmax))
        # fake up the array of the scalar mappable.
        sm._A = []
        # fig.subplots_adjust(bottom=0.05)
        cbar_ax = fig.add_subplot(32, 1, 32)
        fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')

    fig.subplots_adjust(wspace=-0.02, hspace=0.0)

    if title is not None:
        fig.suptitle(title)

    if output_file is not None:
        fig.savefig(output_file)
        plt.close(fig)
    else:
        return fig, axes


def plot_surf_roi(surf_mesh, roi_map, bg_map=None,
                  hemi='left', view='lateral', alpha='auto',
                  vmin=None, vmax=None, cmap='gist_ncar',
                  bg_on_data=False, darkness=1, title=None,
                  output_file=None, axes=None, figure=None, **kwargs):
    """ Plotting ROI on a surface mesh with optional background

    .. versionadded:: 0.3

    Parameters
    ----------
    surf_mesh : str or list of two numpy.ndarray
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z
        coordinates of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces

    roi_map : str or numpy.ndarray or list of numpy.ndarray
        ROI map to be displayed on the surface mesh, can be a file
        (valid formats are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific
        files such as .annot or .label), or
        a Numpy array containing a value for each vertex, or
        a list of Numpy arrays, one array per ROI which contains indices
        of all vertices included in that ROI.

    hemi : {'left', 'right'}, default is 'left'
        Hemisphere to display.

    bg_map : Surface data object (to be defined), optional,
        Background image to be plotted on the mesh underneath the
        stat_map in greyscale, most likely a sulcal depth map for
        realistic shading.

    view: {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'}, default is 'lateral'
        View of the surface that is rendered.

    cmap : matplotlib colormap str or colormap object, default 'coolwarm'
        To use for plotting of the rois. Either a string which is a name
        of a matplotlib colormap, or a matplotlib colormap object.

    alpha : float, default is 'auto'
        Alpha level of the mesh (not the stat_map). If default,
        alpha will default to .5 when no bg_map is passed
        and to 1 if a bg_map is passed.

    bg_on_data : bool, default is False
        If True, and a bg_map is specified, the stat_map data is multiplied
        by the background image, so that e.g. sulcal depth is visible beneath
        the stat_map. Beware that this non-uniformly changes the stat_map
        values according to e.g the sulcal depth.

    darkness : float, between 0 and 1, default is 1
        Specifying the darkness of the background image. 1 indicates that the
        original values of the background are used. .5 indicates the background
        values are reduced by half before being applied.

    title : str, optional
        Figure title.

    output_file: str, or None, optional
        The name of an image file to export plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.

    axes: Axes instance | None
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `plt.subplots(subplot_kw={'projection': '3d'})`).
        If None, a new axes is created.

    figure: Figure instance | None
        The figure to plot to. If None, a new figure is created.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage: For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf: For brain surface visualization.
    """

    v, _ = load_surf_mesh(surf_mesh)

    # if roi_map is a list of arrays with indices for different rois
    if isinstance(roi_map, list):
        roi_list = roi_map[:]
        roi_map = np.zeros(v.shape[0])
        idx = 1
        for arr in roi_list:
            roi_map[arr] = idx
            idx += 1

    elif isinstance(roi_map, np.ndarray):
        # if roi_map is an array with values for all surface nodes
        roi_data = load_surf_data(roi_map)
        # or a single array with indices for a single roi
        if roi_data.shape[0] != v.shape[0]:
            roi_map = np.zeros(v.shape[0], dtype=int)
            roi_map[roi_data] = 1

    else:
        raise ValueError('Invalid input for roi_map. Input can be a file '
                         '(valid formats are .gii, .mgz, .nii, '
                         '.nii.gz, or Freesurfer specific files such as '
                         '.annot or .label), or a Numpy array containing a '
                         'value for each vertex, or a list of Numpy arrays, '
                         'one array per ROI which contains indices of all '
                         'vertices included in that ROI')
    vmin, vmax = np.min(roi_map), 1 + np.max(roi_map)
    display = plot_surf(surf_mesh, surf_map=roi_map, bg_map=bg_map,
                        hemi=hemi, view=view, avg_method='median',
                        cmap=cmap, alpha=alpha, bg_on_data=bg_on_data,
                        darkness=darkness, vmin=vmin, vmax=vmax,
                        title=title, output_file=output_file,
                        axes=axes, figure=figure, **kwargs)

    return display
