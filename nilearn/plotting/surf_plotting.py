"""
Functions for surface visualization.
Only matplotlib is required.
"""

from nilearn._utils.compat import _basestring
from .img_plotting import _get_colorbar_and_data_ranges

# Import libraries
import numpy as np
import nibabel
from nibabel import gifti
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D


# function to figure out datatype and load data
def check_surf_data(surf_data, gii_darray=0):
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
            data = gifti.read(surf_data).darrays[gii_darray].data
        else:
            raise ValueError('Format of data file not recognized.')
    # if the input is an array, it should have a single dimension
    elif isinstance(surf_data, np.ndarray):
        data = np.squeeze(surf_data)
        if len(data.shape) is not 1:
            raise ValueError('Data array cannot have more than one dimension.')
    return data


# function to figure out datatype and load data
def check_surf_mesh(surf_mesh):
    # if input is a filename, try to load it
    if isinstance(surf_mesh, _basestring):
        if (surf_mesh.endswith('orig') or surf_mesh.endswith('pial') or
                surf_mesh.endswith('white') or surf_mesh.endswith('sphere') or
                surf_mesh.endswith('inflated')):
            coords, faces = nibabel.freesurfer.io.read_geometry(surf_mesh)
        elif surf_mesh.endswith('gii'):
            coords, faces = gifti.read(surf_mesh).darrays[0].data, \
                            gifti.read(surf_mesh).darrays[1].data
        else:
            raise ValueError('Format of mesh file not recognized.')
    # if a dictionary is given, check it contains entries for coords and faces
    elif isinstance(surf_mesh, dict):
        if ('faces' in surf_mesh and 'coords' in surf_mesh):
            coords, faces = surf_mesh['coords'], surf_mesh['faces']
        else:
            raise ValueError('If surf_mesh is given as a dictionary it must '
                             'contain items with keys "coords" and "faces"')
    else:
        raise ValueError('surf_mesh must be a either filename or a dictionary '
                         'containing items with keys "coords" and "faces"')
    return coords, faces


def plot_surf_stat_map(surf_mesh, hemi, stat_map=None, bg_map=None,
                       view='lateral', threshold=None, cmap='coolwarm',
                       alpha='auto', vmax=None, symmetric_cbar="auto",
                       bg_on_stat=False, darkness=1, gii_darray=0,
                       output_file=None, **kwargs):

    """ Plotting of surfaces with optional background and stats map

            Parameters
            ----------
            surf_mesh: Surface object (to be defined)
            hemi: Hemisphere to display
            stat_map: Surface data (to be defined) to be displayed, optional
            bg_map: Surface data object (to be defined), optional,
                background image to be plotted on the mesh underneath the
                stat_map in greyscale, most likely a sulcal depth map for
                realistic shading.
            view: {'lateral', 'medial', 'dorsal', 'ventral'}, view of the
                surface that is rendered. Default is 'lateral'
            threshold : a number, None, or 'auto'
                If None is given, the image is not thresholded.
                If a number is given, it is used to threshold the image:
                values below the threshold (in absolute value) are plotted
                as transparent.
            cmap: colormap to use for plotting of the stat_map. Either a string
                which is a name of a matplotlib colormap, or a matplotlib
                colormap object.
            alpha: float, alpha level of the mesh (not the stat_map). If 'auto'
                is chosen, alpha will default to .5 when no bg_map ist passed
                and to 1 if a bg_map is passed.
            vmax: upper bound for plotting of stat_map values.
            symmetric_cbar: boolean or 'auto', optional, default 'auto'
                Specifies whether the colorbar should range from -vmax to vmax
                or from vmin to vmax. Setting to 'auto' will select the latter if
                the range of the whole image is either positive or negative.
                Note: The colormap will always be set to range from -vmax to vmax.
            bg_on_stat: boolean, if True, and a bg_map is specified, the
                stat_map data is multiplied by the background image, so that
                e.g. sulcal depth is visible beneath the stat_map. Beware
                that this non-uniformly changes the stat_map values according
                to e.g the sulcal depth.
            darkness: float, between 0 and 1, specifying the darkness of the
                background image. 1 indicates that the original values of the
                background are used. .5 indicates the background values are
                reduced by half before being applied.
            gii_darray: integer, only applies when stat_map is given as a
                gii_file, specifies the index of the gii array in which the data
                for the stat_map ist stored.
            output_file: string, or None, optional
                The name of an image file to export the plot to. Valid extensions
                are .png, .pdf, .svg. If output_file is not None, the plot
                is saved to a file, and the display is closed.
            kwargs: extra keyword arguments, optional
                Extra keyword arguments passed to matplotlib.pyplot.imshow
        """

    # load mesh and derive axes limits
    coords, faces = check_surf_mesh(surf_mesh)
    limits = [coords.min(), coords.max()]

    # set view
    if hemi == 'rh':
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
    elif hemi == 'lh':
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
        raise ValueError('hemi must be one of rh or lh')

    # set alpha if in auto mode
    if alpha == 'auto':
        if bg_map is None:
            alpha = .5
        else:
            alpha = 1

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

    # If depth_map and/or stat_map are provided, map these onto the surface
    # set_facecolors function of Poly3DCollection is used as passing the
    # facecolors argument to plot_trisurf does not seem to work
    if bg_map is not None or stat_map is not None:

        face_colors = np.ones((faces.shape[0], 4))
        face_colors[:, :3] = .5*face_colors[:, :3]

        if bg_map is not None:
            bg_data = check_surf_data(bg_map, gii_darray=gii_darray)
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
        face_colors[:, 3] = alpha*face_colors[:, 3]

        if stat_map is not None:
            stat_map_data = check_surf_data(stat_map, gii_darray=gii_darray)
            if stat_map_data.shape[0] != coords.shape[0]:
                raise ValueError('The stat_map does not have the same number '
                                 'of vertices as the mesh. For plotting of '
                                 'rois or labels use plot_roi_surf instead')
            stat_map_faces = np.mean(stat_map_data[faces], axis=1)

            # Call _get_colorbar_and_data_ranges to derive symmetric vmin and vmax
            # And colorbar limits depending on symmetric_cbar settings
            cbar_vmin, cbar_vmax, vmin, vmax = \
                _get_colorbar_and_data_ranges(stat_map_faces, vmax,
                                          symmetric_cbar, kwargs)

            if threshold is not None:
                kept_indices = np.where(abs(stat_map_faces) >= threshold)[0]
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax-vmin)
                if bg_on_stat:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices]) * face_colors[kept_indices]
                else:
                    face_colors[kept_indices] = cmap(stat_map_faces[kept_indices])
            else:
                stat_map_faces = stat_map_faces - vmin
                stat_map_faces = stat_map_faces / (vmax-vmin)
                if bg_on_stat:
                    face_colors = cmap(stat_map_faces) * face_colors
                else:
                    face_colors = cmap(stat_map_faces)

        p3dcollec.set_facecolors(face_colors)

    # save figure if output file is given
    if output_file is not None:
        fig.savefig(output_file)
        plt.close(fig)
    else:
        return fig
