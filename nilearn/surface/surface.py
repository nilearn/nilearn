"""
Functions for surface manipulation.
"""
import os
import warnings
import gzip
from distutils.version import LooseVersion

import numpy as np
from scipy import sparse, interpolate
import sklearn.preprocessing
import sklearn.cluster
try:
    from sklearn.exceptions import EfficiencyWarning
except ImportError:
    class EfficiencyWarning(UserWarning):
        """Warning used to notify the user of inefficient computation."""

import nibabel
from nibabel import gifti

from ..image import load_img
from ..image import resampling
from .._utils.compat import _basestring
from .. import _utils


def _uniform_ball_cloud(n_points=20, dim=3, n_monte_carlo=50000):
    """Get points uniformly spaced in the unit ball."""
    rng = np.random.RandomState(0)
    mc_cube = rng.uniform(-1, 1, size=(n_monte_carlo, dim))
    mc_ball = mc_cube[(mc_cube**2).sum(axis=1) <= 1.]
    centroids, assignments, _ = sklearn.cluster.k_means(
        mc_ball, n_clusters=n_points, random_state=0)
    return centroids


def _load_uniform_ball_cloud(n_points=20):
    stored_points = os.path.abspath(
        os.path.join(__file__, '..', 'data',
                     'ball_cloud_{}_samples.csv'.format(n_points)))
    if os.path.isfile(stored_points):
        points = np.loadtxt(stored_points)
        return points
    warnings.warn(
        'Cached sample positions are provided for '
        'n_samples = 10, 20, 40, 80, 160. Since the number of samples does '
        'have a big impact on the result, we strongly recommend using one '
        'of these values when using kind="ball" for much better performance.',
        EfficiencyWarning)
    return _uniform_ball_cloud(n_points=n_points)


def _face_outer_normals(mesh):
    """Get the normal to each triangle in a mesh.

    They are the outer normals if the mesh respects the convention that the
    direction given by the direct order of a triangle's vertices (right-hand
    rule) points outwards.
    """
    vertices, faces = load_surf_mesh(mesh)
    face_vertices = vertices[faces]
    # The right-hand rule gives the direction of the outer normal
    normals = np.cross(face_vertices[:, 1, :] - face_vertices[:, 0, :],
                       face_vertices[:, 2, :] - face_vertices[:, 0, :])
    normals = sklearn.preprocessing.normalize(normals)
    return normals


def _surrounding_faces(mesh):
    """Get matrix indicating which faces the nodes belong to.

    i, j is set if node i is a vertex of triangle j.
    """
    vertices, faces = load_surf_mesh(mesh)
    n_faces = faces.shape[0]
    return sparse.csr_matrix((np.ones(3 * n_faces), (faces.ravel(), np.tile(
        np.arange(n_faces), (3, 1)).T.ravel())), (vertices.shape[0], n_faces))


def _vertex_outer_normals(mesh):
    """Get the normal at each vertex in a triangular mesh.

    They are the outer normals if the mesh respects the convention that the
    direction given by the direct order of a triangle's vertices (right-hand
    rule) points outwards.
    """
    vertices, faces = load_surf_mesh(mesh)
    vertex_faces = _surrounding_faces(mesh)
    face_normals = _face_outer_normals(mesh)
    normals = vertex_faces.dot(face_normals)
    return sklearn.preprocessing.normalize(normals)


def _ball_sample_locations(mesh, affine, ball_radius=3., n_points=20):
    """Locations to draw samples from to project volume data onto a mesh.

    For each mesh vertex, the locations of `n_points` points evenly spread in a
    ball around the vertex are returned.

    Parameters
    ----------
    mesh : pair of np arrays.
        mesh[0] contains the 3d coordinates of the vertices
        (shape n_vertices, 3)
        mesh[1] contains, for each triangle, the indices into mesh[0] of its
        vertices (shape n_triangles, 3)

    affine : array of shape (4, 4)
        affine transformation from image voxels to the vertices' coordinate
        space.

    ball_radius : float, optional (default=3.)
        size in mm of the neighbourhood around each vertex in which to draw
        samples

    n_points : int, optional (default=20)
        number of samples to draw for each vertex.

    Returns
    -------
    numpy array, shape (n_vertices, n_points, 3)
        The locations, in voxel space, from which to draw samples.
        First dimension iterates over mesh vertices, second dimension iterates
        over the sample points associated to a vertex, third dimension is x, y,
        z in voxel space.

    """
    vertices, faces = mesh
    offsets_world_space = _load_uniform_ball_cloud(
        n_points=n_points) * ball_radius
    mesh_voxel_space = np.asarray(
        resampling.coord_transform(*vertices.T,
                                   affine=np.linalg.inv(affine))).T
    linear_map = np.eye(affine.shape[0])
    linear_map[:-1, :-1] = affine[:-1, :-1]
    offsets_voxel_space = np.asarray(
        resampling.coord_transform(*offsets_world_space.T,
                                   affine=np.linalg.inv(linear_map))).T
    sample_locations_voxel_space = (mesh_voxel_space[:, np.newaxis, :] +
                                    offsets_voxel_space[np.newaxis, :])
    return sample_locations_voxel_space


def _line_sample_locations(
        mesh, affine, segment_half_width=3., n_points=10):
    """Locations to draw samples from to project volume data onto a mesh.

    For each mesh vertex, the locations of `n_points` points evenly spread in a
    segment of the normal to the vertex are returned. The line segment has
    length 2 * `segment_half_width` and is centered at the vertex.

    Parameters
    ----------
    mesh : pair of numpy.ndarray.
        mesh[0] contains the 3d coordinates of the vertices
        (shape n_vertices, 3)
        mesh[1] contains, for each triangle, the indices into mesh[0] of its
        vertices (shape n_triangles, 3)

    affine : numpy.ndarray of shape (4, 4)
        affine transformation from image voxels to the vertices' coordinate
        space.

    segment_half_width : float, optional (default=3.)
        size in mm of the neighbourhood around each vertex in which to draw
        samples

    n_points : int, optional (default=10)
        number of samples to draw for each vertex.

    Returns
    -------
    numpy array, shape (n_vertices, n_points, 3)
        The locations, in voxel space, from which to draw samples.
        First dimension iterates over mesh vertices, second dimension iterates
        over the sample points associated to a vertex, third dimension is x, y,
        z in voxel space.

    """
    vertices, faces = mesh
    normals = _vertex_outer_normals(mesh)
    offsets = np.linspace(-segment_half_width, segment_half_width, n_points)
    sample_locations = vertices[
        np.newaxis, :, :] + normals * offsets[:, np.newaxis, np.newaxis]
    sample_locations = np.rollaxis(sample_locations, 1)
    sample_locations_voxel_space = np.asarray(
        resampling.coord_transform(
            *np.vstack(sample_locations).T,
            affine=np.linalg.inv(affine))).T.reshape(sample_locations.shape)
    return sample_locations_voxel_space


def _sample_locations(mesh, affine, radius, kind='line', n_points=None):
    """Get either ball or line sample locations."""
    projectors = {
        'line': _line_sample_locations,
        'ball': _ball_sample_locations
    }
    if kind not in projectors:
        raise ValueError(
            '"kind" must be one of {}'.format(tuple(projectors.keys())))
    projector = projectors[kind]
    # let the projector choose the default for n_points
    # (for example a ball probably needs more than a line)
    loc_kwargs = ({} if n_points is None else {'n_points': n_points})
    sample_locations = projector(
        mesh, affine, radius, **loc_kwargs)
    return sample_locations


def _masked_indices(sample_locations, img_shape, mask=None):
    """Get the indices of sample points which should be ignored.

    Parameters:
    -----------
    sample_locations : array, shape(n_sample_locations, 3)
        The coordinates of candidate interpolation points

    img_shape : tuple
        The dimensions of the image to be sampled

    mask : array of shape img_shape or None
        Part of the image to be masked. If None, don't apply any mask.

    Returns
    -------
    array of shape (n_sample_locations,)
        True if this particular location should be ignored (outside of image or
        masked).

    """
    kept = (sample_locations >= 0).all(axis=1)
    for dim, size in enumerate(img_shape):
        kept = np.logical_and(kept, sample_locations[:, dim] < size)
    if mask is not None:
        indices = np.asarray(np.round(sample_locations[kept]), dtype=int)
        kept[kept] = mask[
            indices[:, 0], indices[:, 1], indices[:, 2]] != 0
    return ~kept


def _projection_matrix(mesh, affine, img_shape,
                       kind='line', radius=3., n_points=None, mask=None):
    """Get a sparse matrix that projects volume data onto a mesh.

    Parameters
    ----------
    mesh : str or numpy.ndarray
        Either a file containing surface mesh geometry (valid formats
        are .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or a list of two Numpy arrays,
        the first containing the x-y-z coordinates of the mesh
        vertices, the second containing the indices (into coords)
        of the mesh faces.

    affine : array of shape (4, 4)
        affine transformation from image voxels to the vertices' coordinate
        space.

    img_shape : 3-tuple of integers
        The shape of the image to be projected.

    kind : {'line', 'ball'}
        The strategy used to sample image intensities around each vertex.

        - 'line' (the default):
            samples are regularly spaced along the normal to the mesh, over the
            interval [-radius, +radius].
        - 'ball':
            samples are regularly spaced inside a ball centered at the mesh
            vertex.

    radius : float, optional (default=3.).
        The size (in mm) of the neighbourhood from which samples are drawn
        around each node.

    n_points : int or None, optional (default=None)
        How many samples are drawn around each vertex and averaged. If None,
        use a reasonable default for the chosen sampling strategy (20 for
        'ball' or 10 for 'line').
        For performance reasons, if using kind="ball", choose `n_points` in
        [10, 20, 40, 80, 160] (default is 20), because cached positions are
        available.

    mask : array of shape img_shape or None
        Part of the image to be masked. If None, don't apply any mask.

    Returns
    -------
    scipy.sparse.csr_matrix
       Shape (n_voxels, n_mesh_vertices). The dot product of this matrix with
       an image (represented as a column vector) gives the projection onto mesh
       vertices.

    See Also
    --------
    nilearn.surface.vol_to_surf
        Compute the projection for one or several images.

    """
    # A user might want to call this function directly so check mask size.
    if mask is not None and tuple(mask.shape) != img_shape:
        raise ValueError('mask should have shape img_shape')
    mesh = load_surf_mesh(mesh)
    sample_locations = _sample_locations(
        mesh, affine, kind=kind, radius=radius, n_points=n_points)
    sample_locations = np.asarray(np.round(sample_locations), dtype=int)
    n_vertices, n_points, img_dim = sample_locations.shape
    masked = _masked_indices(np.vstack(sample_locations), img_shape, mask=mask)
    sample_locations = np.rollaxis(sample_locations, -1)
    sample_indices = np.ravel_multi_index(
        sample_locations, img_shape, mode='clip').ravel()
    row_indices, _ = np.mgrid[:n_vertices, :n_points]
    row_indices = row_indices.ravel()
    row_indices = row_indices[~masked]
    sample_indices = sample_indices[~masked]
    weights = np.ones(len(row_indices))
    proj = sparse.csr_matrix(
        (weights, (row_indices, sample_indices.ravel())),
        shape=(n_vertices, np.prod(img_shape)))
    proj = sklearn.preprocessing.normalize(proj, axis=1, norm='l1')
    return proj


def _nearest_voxel_sampling(images, mesh, affine, kind='ball', radius=3.,
                            n_points=None, mask=None):
    """In each image, measure the intensity at each node of the mesh.

    Image intensity at each sample point is that of the nearest voxel.
    A 2-d array is returned, where each row corresponds to an image and each
    column to a mesh vertex.
    See documentation of vol_to_surf for details.

    """
    proj = _projection_matrix(
        mesh, affine, images[0].shape, kind=kind, radius=radius,
        n_points=n_points, mask=mask)
    data = np.asarray(images).reshape(len(images), -1).T
    texture = proj.dot(data)
    # if all samples around a mesh vertex are outside the image,
    # there is no reasonable value to assign to this vertex.
    # in this case we return NaN for this vertex.
    texture[np.asarray(proj.sum(axis=1) == 0).ravel()] = np.nan
    return texture.T


def _interpolation_sampling(images, mesh, affine, kind='ball', radius=3,
                            n_points=None, mask=None):
    """In each image, measure the intensity at each node of the mesh.

    Image intensity at each sample point is computed with trilinear
    interpolation.
    A 2-d array is returned, where each row corresponds to an image and each
    column to a mesh vertex.
    See documentation of vol_to_surf for details.

    """
    sample_locations = _sample_locations(
        mesh, affine, kind=kind, radius=radius, n_points=n_points)
    n_vertices, n_points, img_dim = sample_locations.shape
    grid = [np.arange(size) for size in images[0].shape]
    interp_locations = np.vstack(sample_locations)
    masked = _masked_indices(interp_locations, images[0].shape, mask=mask)
    # loop over images rather than building a big array to use less memory
    all_samples = []
    for img in images:
        interpolator = interpolate.RegularGridInterpolator(
            grid, img,
            bounds_error=False, method='linear', fill_value=None)
        samples = interpolator(interp_locations)
        # if all samples around a mesh vertex are outside the image,
        # there is no reasonable value to assign to this vertex.
        # in this case we return NaN for this vertex.
        samples[masked] = np.nan
        all_samples.append(samples)
    all_samples = np.asarray(all_samples)
    all_samples = all_samples.reshape((len(images), n_vertices, n_points))
    texture = np.nanmean(all_samples, axis=2)
    return texture


def vol_to_surf(img, surf_mesh,
                radius=3., interpolation='linear', kind='line',
                n_samples=None, mask_img=None):
    """Extract surface data from a Nifti image.

    .. versionadded:: 0.4.0

    Parameters
    ----------

    img : Niimg-like object, 3d or 4d.
        See http://nilearn.github.io/manipulating_images/input_output.html

    surf_mesh : str or numpy.ndarray
        Either a file containing surface mesh geometry (valid formats
        are .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or a list of two Numpy arrays,
        the first containing the x-y-z coordinates of the mesh
        vertices, the second containing the indices (into coords)
        of the mesh faces.

    radius : float, optional (default=3.).
        The size (in mm) of the neighbourhood from which samples are drawn
        around each node.

    interpolation : {'linear', 'nearest'}
        How the image intensity is measured at a sample point.

        - 'linear' (the default):
            Use a trilinear interpolation of neighboring voxels.
        - 'nearest':
            Use the intensity of the nearest voxel.

        For one image, the speed difference is small, 'linear' takes about x1.5
        more time. For many images, 'nearest' scales much better, up to x20
        faster.

    kind : {'line', 'ball'}
        The strategy used to sample image intensities around each vertex.

        - 'line' (the default):
            samples are regularly spaced along the normal to the mesh, over the
            interval [- `radius`, + `radius`].
            (sometimes called thickness sampling)
        - 'ball':
            samples are regularly spaced inside a ball centered at the mesh
            vertex.

    n_samples : int or None, optional (default=None)
        How many samples are drawn around each vertex and averaged. If
        ``None``, use a reasonable default for the chosen sampling strategy
        (20 for 'ball' or 10 for 'line').
        For performance reasons, if using `kind` ="ball", choose `n_samples` in
        [10, 20, 40, 80, 160] (default is 20), because cached positions are
        available.

    mask_img : Niimg-like object or None, optional (default=None)
        Samples falling out of this mask or out of the image are ignored.
        If ``None``, don't apply any mask.

    Returns
    -------
    texture : numpy.ndarray, 1d or 2d.
        If 3D image is provided, a 1d vector is returned, containing one value
        for each mesh node.
        If 4D image is provided, a 2d array is returned, where each row
        corresponds to a mesh node.

    Notes
    -----
    This function computes a value for each vertex of the mesh. In order to do
    so, it selects a few points in the volume surrounding that vertex,
    interpolates the image intensities at these sampling positions, and
    averages the results.

    Two strategies are available to select these positions.
        - 'ball' uses points regularly spaced in a ball centered at the mesh
            vertex. The radius of the ball is controlled by the parameter
            `radius`.
        - 'line' starts by drawing the normal to the mesh passing through this
            vertex. It then selects a segment of this normal, centered at the
            vertex, of length 2 * `radius`. Image intensities are measured at
            points regularly spaced on this normal segment.

    You can control how many samples are drawn by setting `n_samples`.

    Once the sampling positions are chosen, those that fall outside of the 3d
    image (or ouside of the mask if you provided one) are discarded. If all
    sample positions are discarded (which can happen, for example, if the
    vertex itself is outside of the support of the image), the projection at
    this vertex will be ``numpy.nan``.

    The 3d image then needs to be interpolated at each of the remaining points.
    Two options are available: 'nearest' selects the value of the nearest
    voxel, and 'linear' performs trilinear interpolation of neighbouring
    voxels. 'linear' may give better results - for example, the projected
    values are more stable when resampling the 3d image or applying affine
    transformations to it. For one image, the speed difference is small,
    'linear' takes about x1.5 more time. For many images, 'nearest' scales much
    better, up to x20 faster.

    Once the 3d image has been interpolated at each sample point, the
    interpolated values are averaged to produce the value associated to this
    particular mesh vertex.

    WARNING: This function is experimental and details such as the
    interpolation method are subject to change.

    """
    sampling_schemes = {'linear': _interpolation_sampling,
                        'nearest': _nearest_voxel_sampling}
    if interpolation not in sampling_schemes:
        raise ValueError('"interpolation" should be one of {}'.format(
            tuple(sampling_schemes.keys())))
    img = load_img(img)
    if mask_img is not None:
        mask_img = _utils.check_niimg(mask_img)
        mask = resampling.resample_to_img(
            mask_img, img, interpolation='nearest', copy=False).get_data()
    else:
        mask = None
    original_dimension = len(img.shape)
    img = _utils.check_niimg(img, atleast_4d=True)
    frames = np.rollaxis(img.get_data(), -1)
    mesh = load_surf_mesh(surf_mesh)
    sampling = sampling_schemes[interpolation]
    texture = sampling(
        frames, mesh, img.affine, radius=radius, kind=kind,
        n_points=n_samples, mask=mask)
    if original_dimension == 3:
        texture = texture[0]
    return texture.T


def _load_surf_files_gifti_gzip(surf_file):
    """Load surface data Gifti files which are gzipped. This
    function is used by load_surf_mesh and load_surf_data for
    extracting gzipped files.

    Part of the code can be removed while bumping nibabel 2.0.2
    """
    with gzip.open(surf_file) as f:
        as_bytes = f.read()
    if LooseVersion(nibabel.__version__) >= LooseVersion('2.1.0'):
        parser = gifti.GiftiImage.parser()
        parser.parse(as_bytes)
        gifti_img = parser.img
    else:
        from nibabel.gifti.parse_gifti_fast import ParserCreate, Outputter
        parser = ParserCreate()
        parser.buffer_text = True
        out = Outputter()
        parser.StartElementHandler = out.StartElementHandler
        parser.EndElementHandler = out.EndElementHandler
        parser.CharacterDataHandler = out.CharacterDataHandler
        parser.Parse(as_bytes)
        gifti_img = out.img
    return gifti_img


def _gifti_img_to_data(gifti_img):
    """Load surface image e.g. sulcal depth or statistical map in
    nibabel.gifti.GiftiImage to data

    Used by load_surf_data function in common to surface sulcal data
    acceptable to .gii or .gii.gz
    """
    if not gifti_img.darrays:
        raise ValueError('Gifti must contain at least one data array')
    return np.asarray([arr.data for arr in gifti_img.darrays]).T.squeeze()


# function to figure out datatype and load data
def load_surf_data(surf_data):
    """Loading data to be represented on a surface mesh.

    Parameters
    ----------
    surf_data : str or numpy.ndarray
        Either a file containing surface data (valid format are .gii,
        .gii.gz, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
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
            if LooseVersion(nibabel.__version__) >= LooseVersion('2.1.0'):
                gii = nibabel.load(surf_data)
            else:
                gii = gifti.read(surf_data)
            data = _gifti_img_to_data(gii)
        elif surf_data.endswith('gii.gz'):
            gii = _load_surf_files_gifti_gzip(surf_data)
            data = _gifti_img_to_data(gii)
        else:
            raise ValueError(('The input type is not recognized. %r was given '
                              'while valid inputs are a Numpy array or one of '
                              'the following file formats: .gii, .gii.gz, '
                              '.mgz, .nii, .nii.gz, Freesurfer specific files '
                              'such as .curv, .sulc, .thickness, .annot, '
                              '.label') % surf_data)
    # if the input is a numpy array
    elif isinstance(surf_data, np.ndarray):
        data = np.squeeze(surf_data)
    else:
        raise ValueError('The input type is not recognized. '
                         'Valid inputs are a Numpy array or one of the '
                         'following file formats: .gii, .gii.gz, .mgz, .nii, '
                         '.nii.gz, Freesurfer specific files such as .curv, '
                         '.sulc, .thickness, .annot, .label')
    return data


def _gifti_img_to_mesh(gifti_img):
    """Load surface image in nibabel.gifti.GiftiImage to data

    Used by load_surf_mesh function in common to surface mesh
    acceptable to .gii or .gii.gz
    """
    error_message = ('The surf_mesh input is not recognized. Valid Freesurfer '
                     'surface mesh inputs are .pial, .inflated, .sphere, '
                     '.orig, .white. You provided input which have no '
                     '{0} or of empty value={1}') 
    if LooseVersion(nibabel.__version__) >= LooseVersion('2.1.0'):
        try:
            coords = gifti_img.get_arrays_from_intent(
                nibabel.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].data
        except IndexError:
            raise ValueError(error_message.format(
                     'NIFTI_INTENT_POINTSET', gifti_img.get_arrays_from_intent(
                        nibabel.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])))
        try:
            faces = gifti_img.get_arrays_from_intent(
                nibabel.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].data
        except IndexError:
            raise ValueError(error_message.format(
                     'NIFTI_INTENT_TRIANGLE', gifti_img.get_arrays_from_intent(
                        nibabel.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])))
    else:
        try:
            coords = gifti_img.getArraysFromIntent(
                nibabel.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])[0].data
        except IndexError:
            raise ValueError(error_message.format(
                        'NIFTI_INTENT_POINTSET', gifti_img.getArraysFromIntent(
                        nibabel.nifti1.intent_codes['NIFTI_INTENT_POINTSET'])))
        try:
            faces = gifti_img.getArraysFromIntent(
                nibabel.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])[0].data
        except IndexError:
            raise ValueError(error_message.format(
                        'NIFTI_INTENT_TRIANGLE', gifti_img.getArraysFromIntent(
                        nibabel.nifti1.intent_codes['NIFTI_INTENT_TRIANGLE'])))

    return coords, faces


# function to figure out datatype and load data
def load_surf_mesh(surf_mesh):
    """Loading a surface mesh geometry

    Parameters
    ----------
    surf_mesh : str or numpy.ndarray
        Either a file containing surface mesh geometry (valid formats
        are .gii .gii.gz or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or a list or tuple of two Numpy arrays,
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
            if LooseVersion(nibabel.__version__) >= LooseVersion('2.1.0'):
                gifti_img = nibabel.load(surf_mesh)
            else:
                gifti_img = gifti.read(surf_mesh)
            coords, faces = _gifti_img_to_mesh(gifti_img)
        elif surf_mesh.endswith('.gii.gz'):
            gifti_img = _load_surf_files_gifti_gzip(surf_mesh)
            coords, faces = _gifti_img_to_mesh(gifti_img)
        else:
            raise ValueError(('The input type is not recognized. %r was given '
                              'while valid inputs are one of the following '
                              'file formats: .gii, .gii.gz, Freesurfer specific'
                              ' files such as .orig, .pial, .sphere, .white, '
                              '.inflated or a list containing two Numpy '
                              'arrays [vertex coordinates, face indices]'
                              ) % surf_mesh)
    elif isinstance(surf_mesh, (list, tuple)):
        try:
            coords, faces = surf_mesh
        except Exception:
            raise ValueError(('If a list or tuple is given as input, '
                              'it must have two elements, the first is '
                              'a Numpy array containing the x-y-z coordinates '
                              'of the mesh vertices, the second is a Numpy '
                              'array containing  the indices (into coords) of '
                              'the mesh faces. The input was a list with '
                              '%r elements.') % len(surf_mesh))
    else:
        raise ValueError('The input type is not recognized. '
                         'Valid inputs are one of the following file '
                         'formats: .gii, .gii.gz, Freesurfer specific files '
                         'such as .orig, .pial, .sphere, .white, .inflated '
                         'or a list containing two Numpy arrays '
                         '[vertex coordinates, face indices]')

    return [coords, faces]


def check_mesh_and_data(mesh, data):
    """Load surface mesh and data, check that they have compatible shapes."""
    mesh = load_surf_mesh(mesh)
    nodes, faces = mesh
    data = load_surf_data(data)
    if len(data) != len(nodes):
        raise ValueError(
            'Mismatch between number of nodes in mesh ({}) and '
            'size of surface data ({})'.format(len(nodes), len(data)))
    return mesh, data
