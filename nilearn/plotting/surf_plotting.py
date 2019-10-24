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

from ..surface import load_surf_data, load_surf_mesh
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
        a Numpy array with a value for each vertex of the surf_mesh.

    bg_map: Surface data object (to be defined), optional,
        Background image to be plotted on the mesh underneath the
        surf_data in greyscale, most likely a sulcal depth map for
        realistic shading.

    hemi : {'left', 'right'}, default is 'left'
        Hemisphere to display.

    view: {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'},
        default is 'lateral'
        View of the surface that is rendered.

    cmap: matplotlib colormap, str or colormap object, default is None
        To use for plotting of the stat_map. Either a string
        which is a name of a matplotlib colormap, or a matplotlib
        colormap object. If None, matplotlib default will be chosen

    colorbar : bool, optional, default is False
        If True, a colorbar of surf_map is displayed.

    avg_method: {'mean', 'median'}, default is 'mean'
        How to average vertex values to derive the face value, mean results
        in smooth, median in sharp boundaries.

    threshold : a number or None, default is None.
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image, values
        below the threshold (in absolute value) are plotted as transparent.

    alpha: float, alpha level of the mesh (not surf_data), default 'auto'
        If 'auto' is chosen, alpha will default to .5 when no bg_map
        is passed and to 1 if a bg_map is passed.

    bg_on_data: bool, default is False
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

    nilearn.plotting.plot_surf_stat_map : for plotting statistical maps on
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
    axes.view_init(elev=elev, azim=azim)
    axes.set_axis_off()

    # plot mesh without data
    p3dcollec = axes.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                  triangles=faces, linewidth=0.,
                                  antialiased=False,
                                  color='white')

    # reduce viewing distance to remove space around mesh
    axes.dist = 8

    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    face_colors = np.ones((faces.shape[0], 4))

    if bg_map is None:
        bg_data = np.ones(coords.shape[0]) * 0.5

    else:
        bg_data = load_surf_data(bg_map)
        if bg_data.shape[0] != coords.shape[0]:
            raise ValueError('The bg_map does not have the same number '
                             'of vertices as the mesh.')

    bg_faces = np.mean(bg_data[faces], axis=1)
    if bg_faces.min() != bg_faces.max():
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
            kept_indices = np.arange(surf_map_faces.shape[0])
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
        a Numpy array with a value for each vertex of the surf_mesh.

    bg_map : Surface data object (to be defined), optional,
        Background image to be plotted on the mesh underneath the
        stat_map in greyscale, most likely a sulcal depth map for
        realistic shading.

    hemi : {'left', 'right'}, default is 'left'
        Hemispere to display.

    view: {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'},
        default is 'lateral'
        View of the surface that is rendered.

    threshold : a number or None, default is None
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image,
        values below the threshold (in absolute value) are plotted
        as transparent.

    cmap : matplotlib colormap in str or colormap object, default 'cold_hot'
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
    nilearn.datasets.fetch_surf_fsaverage: For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf: For brain surface visualization.
    """

    # Call _get_colorbar_and_data_ranges to derive symmetric vmin, vmax
    # And colorbar limits depending on symmetric_cbar settings
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        stat_map, vmax, symmetric_cbar, kwargs)

    display = plot_surf(
        surf_mesh, surf_map=stat_map, bg_map=bg_map, hemi=hemi, view=view,
        avg_method='mean', threshold=threshold, cmap=cmap, colorbar=colorbar,
        alpha=alpha, bg_on_data=bg_on_data, darkness=darkness, vmax=vmax,
        vmin=vmin, title=title, output_file=output_file, axes=axes,
        figure=figure, cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax, **kwargs)

    return display


def plot_surf_roi(surf_mesh, roi_map, bg_map=None,
                  hemi='left', view='lateral', threshold=1e-14,
                  alpha='auto', vmin=None, vmax=None, cmap='gist_ncar',
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
        a Numpy array with a value for each vertex of the surf_mesh.
        The value at each vertex one inside the ROI and zero inside ROI, or an
        integer giving the label number for atlases.

    hemi : {'left', 'right'}, default is 'left'
        Hemisphere to display.

    bg_map : Surface data object (to be defined), optional,
        Background image to be plotted on the mesh underneath the
        stat_map in greyscale, most likely a sulcal depth map for
        realistic shading.

    view: {'lateral', 'medial', 'dorsal', 'ventral', 'anterior', 'posterior'},
        default is 'lateral'
        View of the surface that is rendered.

    threshold: a number or None
        default is 1e-14 to threshold regions that are labelled 0. If you want
        to use 0 as a label, set threshold to None.

    cmap : matplotlib colormap str or colormap object, default 'gist_ncar'
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

    # preload roi and mesh to determine vmin, vmax and give more useful error
    # messages in case of wrong inputs

    roi = load_surf_data(roi_map)
    if vmin is None:
        vmin = np.min(roi)
    if vmax is None:
        vmax = 1 + np.max(roi)

    mesh = load_surf_mesh(surf_mesh)

    if len(roi.shape) is not 1:
        raise ValueError('roi_map can only have one dimension but has '
                         '%i dimensions' % len(roi.shape))
    if roi.shape[0] != mesh[0].shape[0]:
        raise ValueError('roi_map does not have the same number of vertices '
                         'as the mesh. If you have a list of indices for the '
                         'ROI you can convert them into a ROI map like this:\n'
                         'roi_map = np.zeros(n_vertices)\n'
                         'roi_map[roi_idx] = 1')

    display = plot_surf(mesh, surf_map=roi, bg_map=bg_map,
                        hemi=hemi, view=view, avg_method='median',
                        threshold=threshold, cmap=cmap, alpha=alpha,
                        bg_on_data=bg_on_data, darkness=darkness,
                        vmin=vmin, vmax=vmax, title=title,
                        output_file=output_file, axes=axes,
                        figure=figure, **kwargs)

    return display
