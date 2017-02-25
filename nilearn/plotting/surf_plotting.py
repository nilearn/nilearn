"""
Functions for surface visualization.
Only matplotlib is required.
"""

# Import libraries
import nibabel
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from nibabel import gifti

from .._utils.compat import _basestring
from .img_plotting import _get_colorbar_and_data_ranges


# function to figure out datatype and load data
def load_surf_data(surf_data):
    """Loading data to be represented on a surface mesh.

    Parameters
    ----------
    surf_data : str or numpy.ndarray
        Either a file containing surface data (valid format are .gii,
        .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .curv, .sulc, .annot, .label) or
        a Numpy array containing surface data.
    Returns
    -------
    data : numpy.ndarray
        An array containing surface data
    """
    # if the input is a filename, load it
    if isinstance(surf_data, _basestring):
        if (surf_data.endswith('nii') or surf_data.endswith('nii.gz') or
                surf_data.endswith('mgz')):
            data = np.squeeze(nibabel.load(surf_data).get_data())
        elif (surf_data.endswith('curv') or surf_data.endswith('sulc') or
                surf_data.endswith('thickness')):
            data = nibabel.freesurfer.io.read_morph_data(surf_data)
        elif surf_data.endswith('annot'):
            data = nibabel.freesurfer.io.read_annot(surf_data)[0]
        elif surf_data.endswith('label'):
            data = nibabel.freesurfer.io.read_label(surf_data)
        elif surf_data.endswith('gii'):
            gii = gifti.read(surf_data)
            try:
                data = np.zeros((len(gii.darrays[0].data), len(gii.darrays)))
                for arr in range(len(gii.darrays)):
                    data[:, arr] = gii.darrays[arr].data
                data = np.squeeze(data)
            except IndexError:
                raise ValueError('Gifti must contain at least one data array')
        else:
            raise ValueError(('The input type is not recognized. %r was given '
                              'while valid inputs are a Numpy array or one of '
                              'the following file formats: .gii, .mgz, .nii, '
                              '.nii.gz, Freesurfer specific files such as '
                              '.curv, .sulc, .thickness, .annot, '
                              '.label') % surf_data)
    # if the input is a numpy array
    elif isinstance(surf_data, np.ndarray):
        data = np.squeeze(surf_data)
    else:
        raise ValueError('The input type is not recognized. '
                         'Valid inputs are a Numpy array or one of the '
                         'following file formats: .gii, .mgz, .nii, .nii.gz, '
                         'Freesurfer specific files such as .curv,  .sulc, '
                         '.thickness, .annot, .label')
    return data


# function to figure out datatype and load data
def load_surf_mesh(surf_mesh):
    """Loading a surface mesh geometry

    Parameters
    ----------
    surf_mesh : str or numpy.ndarray
        Either a file containing surface mesh geometry (valid formats
        are .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or a list of two Numpy arrays,
        the first containing the x-y-z coordinates of the mesh
        vertices, the second containing the indices (into coords)
        of the mesh faces.

    Returns
    --------
    [coords, faces] : List of two numpy.ndarray
        The first containing the x-y-z coordinates of the mesh vertices,
        the second containing the indices (into coords) of the mesh faces.
    """
    # if input is a filename, try to load it
    if isinstance(surf_mesh, _basestring):
        if (surf_mesh.endswith('orig') or surf_mesh.endswith('pial') or
                surf_mesh.endswith('white') or surf_mesh.endswith('sphere') or
                surf_mesh.endswith('inflated')):
            coords, faces = nibabel.freesurfer.io.read_geometry(surf_mesh)
        elif surf_mesh.endswith('gii'):
            try:
                coords = gifti.read(surf_mesh).getArraysFromIntent(
                    nibabel.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].data
            except IndexError:
                raise ValueError('Gifti file needs to contain a data array '
                                 'with intent NIFTI_INTENT_POINTSET')
            try:
                faces = gifti.read(surf_mesh).getArraysFromIntent(
                    nibabel.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].data
            except IndexError:
                raise ValueError('Gifti file needs to contain a data array '
                                 'with intent NIFTI_INTENT_TRIANGLE')
        else:
            raise ValueError(('The input type is not recognized. %r was given '
                              'while valid inputs are one of the following '
                              'file formats: .gii, Freesurfer specific files '
                              'such as .orig, .pial, .sphere, .white, '
                              '.inflated or a list containing two Numpy '
                              'arrays [vertex coordinates, face indices]'
                              ) % surf_mesh)
    elif isinstance(surf_mesh, list):
        if len(surf_mesh) == 2:
            coords, faces = surf_mesh[0], surf_mesh[1]
        else:
            raise ValueError(('If a list is given as input, it must have '
                              'two elements, the first is a Numpy array '
                              'containing the x-y-z coordinates of the mesh '
                              'vertices, the second is a Numpy array '
                              'containing  the indices (into coords) of the '
                              'mesh faces. The input was a list with '
                              '%r elements.') % len(surf_mesh))
    else:
        raise ValueError('The input type is not recognized. '
                         'Valid inputs are one of the following file '
                         'formats: .gii, Freesurfer specific files such as '
                         '.orig, .pial, .sphere, .white, .inflated '
                         'or a list containing two Numpy arrays '
                         '[vertex coordinates, face indices]')

    return [coords, faces]


def plot_surf(surf_mesh, surf_map=None, bg_map=None,
              hemi='left', view='lateral', cmap=None,
              avg_method='mean', threshold=None, alpha='auto',
              bg_on_data=False, darkness=1, vmin=None, vmax=None,
              title=None, output_file=None, **kwargs):

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

    view: {'lateral', 'medial', 'dorsal', 'ventral'}, default is 'lateral'
        View of the surface that is rendered.

    cmap: matplotlib colormap, str or colormap object, default is None
        To use for plotting of the stat_map. Either a string
        which is a name of a matplotlib colormap, or a matplotlib
        colormap object. If None, matplolib default will be chosen

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

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage5 : For surface data object to be
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
        else:
            raise ValueError('view must be one of lateral, medial, '
                             'dorsal or ventral')
    elif hemi == 'left':
        if view == 'medial':
            elev, azim = 0, 0
        elif view == 'lateral':
            elev, azim = 0, 180
        elif view == 'dorsal':
            elev, azim = 90, 0
        elif view == 'ventral':
            elev, azim = 270, 0
        else:
            raise ValueError('view must be one of lateral, medial, '
                             'dorsal or ventral')
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
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', xlim=limits, ylim=limits)
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    # plot mesh without data
    p3dcollec = ax.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2],
                                triangles=faces, linewidth=0.,
                                antialiased=False,
                                color='white')

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

        p3dcollec.set_facecolors(face_colors)

    if title is not None:
        plt.title(title)

    # save figure if output file is given
    if output_file is not None:
        fig.savefig(output_file)
        plt.close(fig)
    else:
        return fig


def plot_surf_stat_map(surf_mesh, stat_map, bg_map=None,
                       hemi='left', view='lateral', threshold=None,
                       alpha='auto', vmax=None, cmap='coolwarm',
                       symmetric_cbar="auto", bg_on_data=False, darkness=1,
                       title=None, output_file=None, **kwargs):
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

    view : {'lateral', 'medial', 'dorsal', 'ventral'}, default 'lateral'
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

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage5 : For surface data object to be
        used as background map for this plotting function.

    nilearn.plotting.plot_surf : For brain surface visualization.
    """

    # Call _get_colorbar_and_data_ranges to derive symmetric vmin, vmax
    # And colorbar limits depending on symmetric_cbar settings
    cbar_vmin, cbar_vmax, vmin, vmax = \
        _get_colorbar_and_data_ranges(stat_map, vmax,
                                      symmetric_cbar, kwargs)

    display = plot_surf(surf_mesh, surf_map=stat_map, bg_map=bg_map,
                        hemi=hemi, view=view, avg_method='mean',
                        threshold=threshold, cmap=cmap,
                        alpha=alpha, bg_on_data=bg_on_data, darkness=1,
                        vmax=vmax, title=title, output_file=output_file,
                        **kwargs)

    return display


def plot_surf_roi(surf_mesh, roi_map, bg_map=None,
                  hemi='left', view='lateral', alpha='auto',
                  vmin=None, vmax=None, cmap='coolwarm',
                  bg_on_data=False, darkness=1,
                  title=None, output_file=None, **kwargs):
    """ Plotting of surfaces with optional background and stats map

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

    view : {'lateral', 'medial', 'dorsal', 'ventral'}, default 'lateral'
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

    See Also
    --------
    nilearn.datasets.fetch_surf_fsaverage5: For surface data object to be
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
            roi_map = np.zeros(v.shape[0])
            roi_map[roi_data] = 1

    else:
        raise ValueError('Invalid input for roi_map. Input can be a file '
                         '(valid formats are .gii, .mgz, .nii, '
                         '.nii.gz, or Freesurfer specific files such as '
                         '.annot or .label), or a Numpy array containing a '
                         'value for each vertex, or a list of Numpy arrays, '
                         'one array per ROI which contains indices of all '
                         'vertices included in that ROI')

    display = plot_surf(surf_mesh, surf_map=roi_map, bg_map=bg_map,
                        hemi=hemi, view=view, avg_method='median',
                        cmap=cmap, alpha=alpha, bg_on_data=bg_on_data,
                        darkness=darkness, vmin=vmin, vmax=vmax,
                        title=title, output_file=output_file,
                        **kwargs)

    return display
