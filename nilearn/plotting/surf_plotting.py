"""
Functions for surface visualization.
Only matplotlib is required.
"""
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colorbar import make_axes
from matplotlib.cm import ScalarMappable, get_cmap
from matplotlib.colors import Normalize, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from nilearn import image
from nilearn.plotting.img_plotting import (_get_colorbar_and_data_ranges,
                                           _crop_colorbar)
from nilearn.surface import (load_surf_data,
                             load_surf_mesh,
                             vol_to_surf)
from nilearn.surface.surface import _check_mesh
from nilearn._utils import check_niimg_3d

from matplotlib.colors import to_rgba
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

VALID_VIEWS = "anterior", "posterior", "medial", "lateral", "dorsal", "ventral"
VALID_HEMISPHERES = "left", "right"


def plot_surf(surf_mesh, surf_map=None, bg_map=None,
              hemi='left', view='lateral', cmap=None, colorbar=False,
              avg_method='mean', threshold=None, alpha='auto',
              bg_on_data=False, darkness=1, vmin=None, vmax=None,
              cbar_vmin=None, cbar_vmax=None, cbar_tick_format='%.2g',
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
        If None, the values will be set to min/max of the data

    cbar_vmin, cbar_vmax: lower / upper bounds for the colorbar, optional.
        If None, the values will be set from the data.
        Default values are None.

    cbar_tick_format: str, optional, default is '%.2g' for scientific notation.
        Controls how to format the tick labels of the colorbar.
        Ex: use "%i" to display as integers.

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
        if isinstance(cmap, str):
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
        if surf_map_data.ndim != 1:
            raise ValueError('surf_map can only have one dimension but has'
                             '%i dimensions' % surf_map_data.ndim)
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
                istart = int(norm(-threshold, clip=True) * (our_cmap.N - 1))
                istop = int(norm(threshold, clip=True) * (our_cmap.N - 1))
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
                format=cbar_tick_format, orientation='vertical')
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


def _get_faces_on_edge(faces, parc_idx):
    '''
    Internal function for identifying which faces lie on the outer
    edge of the parcellation defined by the indices in parc_idx.

    Parameters
    ----------
    faces: numpy.ndarray of shape (n, 3), indices of the mesh faces

    parc_idx: numpy.ndarray, indices of the vertices
        of the region to be plotted
    '''
    # count how many vertices belong to the given parcellation in each face
    verts_per_face = np.isin(faces, parc_idx).sum(axis=1)

    # test if parcellation forms regions
    if np.all(verts_per_face < 2):
        raise ValueError('Vertices in parcellation do not form region.')

    vertices_on_edge = np.intersect1d(np.unique(faces[verts_per_face == 2]),
                                      parc_idx)
    faces_outside_edge = np.isin(faces, vertices_on_edge).sum(axis=1)

    return np.logical_and(faces_outside_edge > 0, verts_per_face < 3)


def plot_surf_contours(surf_mesh, roi_map, axes=None, figure=None, levels=None,
                       labels=None, colors=None, legend=False, cmap='tab20',
                       title=None, output_file=None, **kwargs):
    """
    Plotting contours of ROIs on a surface, optionally over a statistical map.

    Parameters
    ----------
    surf_mesh: str or list of two numpy.ndarray
        Surface mesh geometry, can be a file (valid formats are
        .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or
        a list of two Numpy arrays, the first containing the x-y-z coordinates
        of the mesh vertices, the second containing the indices
        (into coords) of the mesh faces.

    roi_map: str or numpy.ndarray or list of numpy.ndarray
        ROI map to be displayed on the surface mesh, can be a file
        (valid formats are .gii, .mgz, .nii, .nii.gz, or Freesurfer specific
        files such as .annot or .label), or
        a Numpy array with a value for each vertex of the surf_mesh.
        The value at each vertex one inside the ROI and zero inside ROI, or an
        integer giving the label number for atlases.

    axes: instance of matplotlib axes, None, optional
        The axes instance to plot to. The projection must be '3d' (e.g.,
        `figure, axes = plt.subplots(subplot_kw={'projection': '3d'})`,
        where axes should be passed.).
        If None, uses axes from figure if available, else creates new axes.

    figure: instance of matplotlib figure, None, optional
        The figure instance to plot to.
        If None, uses figure of axes if available, else creates a new figure.

    levels: list of integers, or None, optional
        A list of indices of the regions that are to be outlined.
        Every index needs to correspond to one index in roi_map.
        If None, all regions in roi_map are used.

    labels: list of strings or None, or None, optional
        A list of labels for the individual regions of interest.
        Provide None as list entry to skip showing the label of that region.
        If None no labels are used.

    colors: list of matplotlib color names or RGBA values, or None.

    legend: boolean,  optional
        Whether to plot a legend of region's labels.

    cmap: matplotlib colormap, str or colormap object, default is None
        To use for plotting of the contours. Either a string
        which is a name of a matplotlib colormap, or a matplotlib
        colormap object.

    title : str, optional
        Figure title.

    output_file: str, or None, optional
        The name of an image file to export plot to. Valid extensions
        are .png, .pdf, .svg. If output_file is not None, the plot
        is saved to a file, and the display is closed.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf_stat_map : for plotting statistical maps on
        brain surfaces.
    """
    if figure is None and axes is None:
        figure = plot_surf(surf_mesh, **kwargs)
        axes = figure.axes[0]
    if figure is None:
        figure = axes.get_figure()
    if axes is None:
        axes = figure.axes[0]
    if axes.name != '3d':
        raise ValueError('Axes must be 3D.')
    # test if axes contains Poly3DCollection, if not initialize surface
    if not axes.collections or not isinstance(axes.collections[0],
                                              Poly3DCollection):
        _ = plot_surf(surf_mesh, axes=axes, **kwargs)

    coords, faces = load_surf_mesh(surf_mesh)
    roi = load_surf_data(roi_map)
    if levels is None:
        levels = np.unique(roi_map)
    if colors is None:
        n_levels = len(levels)
        vmax = n_levels
        cmap = get_cmap(cmap)
        norm = Normalize(vmin=0, vmax=vmax)
        colors = [cmap(norm(color_i)) for color_i in range(vmax)]
    else:
        try:
            colors = [to_rgba(color, alpha=1.) for color in colors]
        except ValueError:
            raise ValueError('All elements of colors need to be either a'
                             ' matplotlib color string or RGBA values.')

    if labels is None:
        labels = [None] * len(levels)
    if not (len(labels) == len(levels) and len(colors) == len(labels)):
        raise ValueError('Levels, labels, and colors '
                         'argument need to be either the same length or None.')

    patch_list = []
    for level, color, label in zip(levels, colors, labels):
        roi_indices = np.where(roi == level)[0]
        faces_outside = _get_faces_on_edge(faces, roi_indices)
        # Fix: Matplotlib version 3.3.2 to 3.3.3
        # Attribute _facecolors3d changed to _facecolor3d in
        # matplotlib version 3.3.3
        try:
            axes.collections[0]._facecolors3d[faces_outside] = color
        except AttributeError:
            axes.collections[0]._facecolor3d[faces_outside] = color
        if label and legend:
            patch_list.append(Patch(color=color, label=label))
    # plot legend only if indicated and labels provided
    if legend and np.any([lbl is not None for lbl in labels]):
        figure.legend(handles=patch_list)
    if title:
        figure.suptitle(title)
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

    loaded_stat_map = load_surf_data(stat_map)

    # Call _get_colorbar_and_data_ranges to derive symmetric vmin, vmax
    # And colorbar limits depending on symmetric_cbar settings
    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        loaded_stat_map, vmax, symmetric_cbar, kwargs)

    display = plot_surf(
        surf_mesh, surf_map=loaded_stat_map, bg_map=bg_map, hemi=hemi, view=view,
        avg_method='mean', threshold=threshold, cmap=cmap, colorbar=colorbar,
        alpha=alpha, bg_on_data=bg_on_data, darkness=darkness, vmax=vmax,
        vmin=vmin, title=title, output_file=output_file, axes=axes,
        figure=figure, cbar_vmin=cbar_vmin, cbar_vmax=cbar_vmax, **kwargs)

    return display


def _check_hemispheres(hemispheres):
    """Checks whether the hemispheres passed to in plot_img_on_surf are
    correct.

    hemispheres : list
        Any combination of 'left' and 'right'.
    """
    invalid_hemi = any([hemi not in VALID_HEMISPHERES for hemi in hemispheres])
    if invalid_hemi:
        supported = "Supported hemispheres:\n" + str(VALID_HEMISPHERES)
        raise ValueError("Invalid hemispheres definition!\n" + supported)
    return hemispheres


def _check_views(views) -> list:
    """Checks whether the views passed to in plot_img_on_surf are
    correct.

    views : list
        Any combination of "anterior", "posterior", "medial", "lateral",
        "dorsal", "ventral".
    """
    invalid_view = any([view not in VALID_VIEWS for view in views])
    if invalid_view:
        supported = "Supported views:\n" + str(VALID_VIEWS)
        raise ValueError("Invalid view definition!\n" + supported)
    return views


def _colorbar_from_array(array, vmax, threshold, kwargs,
                         cmap='cold_hot'):
    """Generate a custom colorbar for an array.

    Internal function used by plot_img_on_surf

    array : np.ndarray
        Any 3D array.

    vmax : float
        upper bound for plotting of stat_map values.

    threshold : float
        If None is given, the colorbar is not thresholded.
        If a number is given, it is used to threshold the colorbar.
        Absolute values lower than threshold are shown in gray.

    kwargs : dict
        Extra arguments passed to _get_colorbar_and_data_ranges.

    cmap : str, optional (default='cold_hot')
        The name of a matplotlib or nilearn colormap.
    """

    cbar_vmin, cbar_vmax, vmin, vmax = _get_colorbar_and_data_ranges(
        array, vmax, True, kwargs
    )
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmaplist = [cmap(i) for i in range(cmap.N)]

    if threshold is None:
        threshold = 0.

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

    return sm


def plot_img_on_surf(stat_map, surf_mesh='fsaverage5', mask_img=None,
                     hemispheres=['left', 'right'],
                     inflate=False,
                     views=['lateral', 'medial'],
                     output_file=None, title=None, colorbar=True,
                     vmax=None, threshold=None,
                     cmap='cold_hot', aspect_ratio=1.4, **kwargs):
    """Convenience function to plot multiple views of plot_surf_stat_map
    in a single figure. It projects stat_map into meshes and plots views of
    left and right hemispheres. The *views* argument defines the views
    that are shown. This function returns the fig, axes elements from
    matplotlib unless kwargs sets and output_file, in which case nothing
    is returned.

    stat_map : str or 3D Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html

    surf_mesh : str, dict, or None, default is 'fsaverage5'
        If str, either one of the two:
        'fsaverage5': the low-resolution fsaverage5 mesh (10242 nodes)
        'fsaverage': the high-resolution fsaverage mesh (163842 nodes)
        If dict, a dictionary with keys: ['infl_left', 'infl_right',
        'pial_left', 'pial_right', 'sulc_left', 'sulc_right'], where
        values are surface mesh geometries as accepted by plot_surf_stat_map.

    mask_img : Niimg-like object or None, optional (default=None)
        The mask is passed to vol_to_surf.
        Samples falling out of this mask or out of the image are ignored
        during projection of the volume to the surface.
        If ``None``, don't apply any mask.

    inflate : bool, optional (default=False)
        If True, display images in inflated brain.
        If False, display images in pial surface.

    views : list, optional (default=['lateral', 'medial'])
        A list containing all views to display.
        The montage will contain as many rows as views specified by
        display mode. Order is preserved, and left and right hemispheres
        are shown on the left and right sides of the figure.

    hemispheres : list, optional (default=['left', 'right'])
        Hemispheres to display

    output_file : str, optional (default=None)
        The name of an image file to export plot to. Valid extensions
        are: *.png*, *.pdf*, *.svg*. If output_file is not None,
        the plot is saved to a file, and the display is closed. Return
        value is None.

    title : str, optional (default=None)
        Place a title on the upper center of the figure.

    colorbar : bool, optional (default=True)
        If *True*, a symmetric colorbar of the statistical map is displayed.

    vmax : float, optional (default=None)
        Upper bound for plotting of stat_map values.

    threshold : float, optional (default=None)
        If None is given, the image is not thresholded.
        If a number is given, it is used to threshold the image,
        values below the threshold (in absolute value) are plotted
        as transparent.

    cmap : str, optional (default='cold_hot')
        The name of a matplotlib or nilearn colormap.

    kwargs : dict
        keyword arguments passed to plot_surf_stat_map.

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage : For surface data object to be
        used as the default background map for this plotting function.

    nilearn.surface.vol_to_surf : For info on the generation of surfaces.

    nilearn.plotting.plot_surf_stat_map : For info on kwargs options
        accepted by plot_img_on_surf.
    """
    for arg in ('figure', 'axes'):
        if arg in kwargs:
            raise ValueError(('plot_img_on_surf does not'
                              ' accept %s as an argument' % arg))

    stat_map = check_niimg_3d(stat_map, dtype='auto')
    modes = _check_views(views)
    hemis = _check_hemispheres(hemispheres)
    surf_mesh = _check_mesh(surf_mesh)

    mesh_prefix = "infl" if inflate else "pial"
    surf = {
        'left': surf_mesh[mesh_prefix + '_left'],
        'right': surf_mesh[mesh_prefix + '_right'],
    }

    texture = {
        'left': vol_to_surf(stat_map, surf_mesh['pial_left'],
                            mask_img=mask_img),
        'right': vol_to_surf(stat_map, surf_mesh['pial_right'],
                             mask_img=mask_img)
    }

    figsize = plt.figaspect(len(modes) / (aspect_ratio * len(hemispheres)))
    fig, axes = plt.subplots(nrows=len(modes),
                             ncols=len(hemis),
                             figsize=figsize,
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
                               colorbar=False,  # Colorbar created externally.
                               vmax=vmax,
                               threshold=threshold,
                               cmap=cmap,
                               **kwargs)

    for ax in axes.flatten():
        # We increase this value to better position the camera of the
        # 3D projection plot. The default value makes meshes look too small.
        ax.dist = 6

    if colorbar:
        sm = _colorbar_from_array(image.get_data(stat_map),
                                  vmax, threshold, kwargs,
                                  cmap=get_cmap(cmap))

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

    if roi.ndim != 1:
        raise ValueError('roi_map can only have one dimension but has '
                         '%i dimensions' % roi.ndim)
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
