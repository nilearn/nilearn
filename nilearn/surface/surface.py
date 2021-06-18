"""
Functions for surface manipulation.
"""
import os
import warnings
import collections
import gzip
from distutils.version import LooseVersion
from collections import namedtuple


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
from nibabel import freesurfer as fs

from nilearn import datasets
from nilearn.image import load_img
from nilearn.image import resampling
from nilearn._utils.path_finding import _resolve_globbing
from nilearn import _utils
from nilearn.image import get_data

# Create a namedtuple object for meshes
Mesh = namedtuple("mesh", ["coordinates", "faces"])

# Create a namedtuple object for surfaces
Surface = namedtuple("surface", ["mesh", "data"])

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


def _sample_locations_between_surfaces(
        mesh, inner_mesh, affine, n_points=10, depth=None):
    outer_vertices, _ = mesh
    inner_vertices, _ = inner_mesh
    # when we drop support for np 1.5 replace the next 2 lines with
    # sample_locations = np.linspace(inner_vertices, outer_vertices, n_points)
    if depth is None:
        steps = np.linspace(0, 1, n_points)[:, None, None]
    else:
        steps = np.asarray(depth)[:, None, None]
    sample_locations = outer_vertices + steps * (
        inner_vertices - outer_vertices)
    sample_locations = np.rollaxis(sample_locations, 1)
    sample_locations_voxel_space = np.asarray(
        resampling.coord_transform(
            *np.vstack(sample_locations).T,
            affine=np.linalg.inv(affine))).T.reshape(sample_locations.shape)
    return sample_locations_voxel_space


def _ball_sample_locations(
        mesh, affine, ball_radius=3., n_points=20, depth=None):
    """Locations to draw samples from to project volume data onto a mesh.

    For each mesh vertex, the locations of `n_points` points evenly spread in a
    ball around the vertex are returned.

    Parameters
    ----------
    mesh : pair of np arrays.
        `mesh[0]` contains the 3d coordinates of the vertices
        (shape n_vertices, 3)
        `mesh[1]` contains, for each triangle, the indices into `mesh[0]` of its
        vertices (shape n_triangles, 3)

    affine : array of shape (4, 4)
        Affine transformation from image voxels to the vertices' coordinate
        space.

    ball_radius : float, optional
        Size in mm of the neighbourhood around each vertex in which to draw
        samples. Default=3.0.

    n_points : int, optional
        Number of samples to draw for each vertex. Default=20.

    depth : None
        Raises a ValueError if not None because incompatible with this sampling
        strategy.

    Returns
    -------
    sample_location_voxel_space : numpy array, shape (n_vertices, n_points, 3)
        The locations, in voxel space, from which to draw samples.
        First dimension iterates over mesh vertices, second dimension iterates
        over the sample points associated to a vertex, third dimension is x, y,
        z in voxel space.

    """
    if depth is not None:
        raise ValueError("The 'ball' sampling strategy does not support "
                         "the 'depth' parameter")
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
        mesh, affine, segment_half_width=3., n_points=10, depth=None):
    """Locations to draw samples from to project volume data onto a mesh.

    For each mesh vertex, the locations of `n_points` points evenly spread in a
    segment of the normal to the vertex are returned. The line segment has
    length 2 * `segment_half_width` and is centered at the vertex.

    Parameters
    ----------
    mesh : pair of numpy.ndarray
        `mesh[0]` contains the 3d coordinates of the vertices
        (shape n_vertices, 3)
        `mesh[1]` contains, for each triangle, the indices into `mesh[0]` of its
        vertices (shape n_triangles, 3)

    affine : numpy.ndarray of shape (4, 4)
        Affine transformation from image voxels to the vertices' coordinate
        space.

    segment_half_width : float, optional
        Size in mm of the neighbourhood around each vertex in which to draw
        samples. Default=3.0.

    n_points : int, optional
        Number of samples to draw for each vertex. Default=10.

    depth : sequence of floats or None, optional
        Cortical depth, expressed as a fraction of segment_half_width.
        Overrides n_points.

    Returns
    -------
    sample_location_voxel_space : numpy array, shape (n_vertices, n_points, 3)
        The locations, in voxel space, from which to draw samples.
        First dimension iterates over mesh vertices, second dimension iterates
        over the sample points associated to a vertex, third dimension is x, y,
        z in voxel space.

    """
    vertices, faces = mesh
    normals = _vertex_outer_normals(mesh)
    if depth is None:
        offsets = np.linspace(
            segment_half_width, -segment_half_width, n_points)
    else:
        offsets = - segment_half_width * np.asarray(depth)
    sample_locations = vertices[
        np.newaxis, :, :] + normals * offsets[:, np.newaxis, np.newaxis]
    sample_locations = np.rollaxis(sample_locations, 1)
    sample_locations_voxel_space = np.asarray(
        resampling.coord_transform(
            *np.vstack(sample_locations).T,
            affine=np.linalg.inv(affine))).T.reshape(sample_locations.shape)
    return sample_locations_voxel_space


def _choose_kind(kind, inner_mesh):
    if kind == "depth" and inner_mesh is None:
        raise TypeError(
            "'inner_mesh' must be provided to use "
            "the 'depth' sampling strategy")
    if kind == "auto":
        kind = "line" if inner_mesh is None else "depth"
    return kind


def _sample_locations(mesh, affine, radius, kind='auto', n_points=None,
                      inner_mesh=None, depth=None):
    """Get either ball or line sample locations."""
    kind = _choose_kind(kind, inner_mesh)
    kwargs = ({} if n_points is None else {'n_points': n_points})
    projectors = {
        'line': (_line_sample_locations, {"segment_half_width": radius}),
        'ball': (_ball_sample_locations, {"ball_radius": radius}),
        'depth': (_sample_locations_between_surfaces,
                  {"inner_mesh": inner_mesh})
    }
    if kind not in projectors:
        raise ValueError(
            '"kind" must be one of {}'.format(tuple(projectors.keys())))
    projector, extra_kwargs = projectors[kind]
    # let the projector choose the default for n_points
    # (for example a ball probably needs more than a line)
    sample_locations = projector(
        mesh=mesh, affine=affine, depth=depth, **kwargs, **extra_kwargs)
    return sample_locations


def _masked_indices(sample_locations, img_shape, mask=None):
    """Get the indices of sample points which should be ignored.

    Parameters:
    -----------
    sample_locations : array, shape(n_sample_locations, 3)
        The coordinates of candidate interpolation points.

    img_shape : tuple
        The dimensions of the image to be sampled.

    mask : array of shape img_shape or None, optional
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
        indices = np.asarray(np.floor(sample_locations[kept]), dtype=int)
        kept[kept] = mask[
            indices[:, 0], indices[:, 1], indices[:, 2]] != 0
    return ~kept


def _projection_matrix(mesh, affine, img_shape, kind='auto', radius=3.,
                       n_points=None, mask=None, inner_mesh=None, depth=None):
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
        Affine transformation from image voxels to the vertices' coordinate
        space.

    img_shape : 3-tuple of integers
        The shape of the image to be projected.

    kind : {'auto', 'depth', 'line', 'ball'}, optional
        The strategy used to sample image intensities around each vertex.
        Ignored if `inner_mesh` is not None. Default='auto'.

        - 'auto':
            'depth' if `inner_mesh` is not `None`, otherwise 'line.
        - 'depth':
            Sampled at the specified cortical depths between corresponding
            nodes of `mesh` and `inner_mesh`.
        - 'line':
            Samples are placed along the normal to the mesh.
        - 'ball':
            Samples are regularly spaced inside a ball centered at the mesh
            vertex.

    radius : float, optional
        The size (in mm) of the neighbourhood from which samples are drawn
        around each node. Ignored if `inner_mesh` is not None.
        Default=3.0.

    n_points : int or None, optional
        How many samples are drawn around each vertex and averaged. If None,
        use a reasonable default for the chosen sampling strategy (20 for
        'ball' or 10 for lines ie using `line` or an `inner_mesh`).
        For performance reasons, if using kind="ball", choose `n_points` in
        [10, 20, 40, 80, 160] (default is 20), because cached positions are
        available.

    mask : array of shape img_shape or None, optional
        Part of the image to be masked. If None, don't apply any mask.

    inner_mesh : str or numpy.ndarray, optional
        Either a file containing surface mesh or a pair of ndarrays
        (coordinates, triangles). If provided this is an inner surface that is
        nested inside the one represented by `mesh` -- e.g. `mesh` is a pial
        surface and `inner_mesh` a white matter surface. In this case nodes in
        both meshes must correspond: node i in `mesh` is just across the gray
        matter thickness from node i in `inner_mesh`. Image values for index i
        are then sampled along the line joining these two points (if `kind` is
        'auto' or 'depth').

    depth : sequence of floats or None, optional
        Cortical depth, expressed as a fraction of segment_half_width.
        overrides n_points. Should be None if kind is 'ball'

    Returns
    -------
    proj : scipy.sparse.csr_matrix
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
        mesh, affine, kind=kind, radius=radius, n_points=n_points,
        inner_mesh=inner_mesh, depth=depth)
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


def _nearest_voxel_sampling(images, mesh, affine, kind='auto', radius=3.,
                            n_points=None, mask=None, inner_mesh=None,
                            depth=None):
    """In each image, measure the intensity at each node of the mesh.

    Image intensity at each sample point is that of the nearest voxel.
    A 2-d array is returned, where each row corresponds to an image and each
    column to a mesh vertex.
    See documentation of vol_to_surf for details.

    """
    proj = _projection_matrix(
        mesh, affine, images[0].shape, kind=kind, radius=radius,
        n_points=n_points, mask=mask, inner_mesh=inner_mesh, depth=depth)
    data = np.asarray(images).reshape(len(images), -1).T
    texture = proj.dot(data)
    # if all samples around a mesh vertex are outside the image,
    # there is no reasonable value to assign to this vertex.
    # in this case we return NaN for this vertex.
    texture[np.asarray(proj.sum(axis=1) == 0).ravel()] = np.nan
    return texture.T


def _interpolation_sampling(images, mesh, affine, kind='auto', radius=3,
                            n_points=None, mask=None, inner_mesh=None,
                            depth=None):
    """In each image, measure the intensity at each node of the mesh.

    Image intensity at each sample point is computed with trilinear
    interpolation.
    A 2-d array is returned, where each row corresponds to an image and each
    column to a mesh vertex.
    See documentation of vol_to_surf for details.

    """
    sample_locations = _sample_locations(
        mesh, affine, kind=kind, radius=radius, n_points=n_points,
        inner_mesh=inner_mesh, depth=depth)
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
                radius=3., interpolation='linear', kind='auto',
                n_samples=None, mask_img=None, inner_mesh=None, depth=None):
    """Extract surface data from a Nifti image.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    img : Niimg-like object, 3d or 4d.
        See http://nilearn.github.io/manipulating_images/input_output.html

    surf_mesh : str or numpy.ndarray or Mesh
        Either a file containing surface mesh geometry (valid formats
        are .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or two Numpy arrays organized in a list,
        tuple or a namedtuple with the fields "coordinates" and "faces", or
        a Mesh object with "coordinates" and "faces" attributes.

    radius : float, optional
        The size (in mm) of the neighbourhood from which samples are drawn
        around each node. Ignored if `inner_mesh` is provided.
        Default=3.0.

    interpolation : {'linear', 'nearest'}, optional
        How the image intensity is measured at a sample point.
        Default='linear'.

        - 'linear':
            Use a trilinear interpolation of neighboring voxels.
        - 'nearest':
            Use the intensity of the nearest voxel.

        For one image, the speed difference is small, 'linear' takes about x1.5
        more time. For many images, 'nearest' scales much better, up to x20
        faster.

    kind : {'auto', 'depth', 'line', 'ball'}, optional
        The strategy used to sample image intensities around each vertex.
        Default='auto'.

        - 'auto':
            Chooses 'depth' if `inner_mesh` is provided and 'line' otherwise.
        - 'depth':
            `inner_mesh` must be a mesh whose nodes correspond to those in
            `surf_mesh`. For example, `inner_mesh` could be a white matter
            surface mesh and `surf_mesh` a pial surface mesh. Samples are
            placed between each pair of corresponding nodes at the specified
            cortical depths (regularly spaced by default, see `depth`
            parameter).
        - 'line':
            Samples are placed along the normal to the mesh, at the positions
            specified by `depth`, or by default regularly spaced over the
            interval [- `radius`, + `radius`].
        - 'ball':
            Samples are regularly spaced inside a ball centered at the mesh
            vertex.

    n_samples : int or None, optional
        How many samples are drawn around each vertex and averaged. If
        ``None``, use a reasonable default for the chosen sampling strategy
        (20 for 'ball' or 10 for 'line').
        For performance reasons, if using `kind` ="ball", choose `n_samples` in
        [10, 20, 40, 80, 160] (default is 20), because cached positions are
        available.

    mask_img : Niimg-like object or None, optional
        Samples falling out of this mask or out of the image are ignored.
        If ``None``, don't apply any mask.

    inner_mesh : str or numpy.ndarray, optional
        Either a file containing a surface mesh or a pair of ndarrays
        (coordinates, triangles). If provided this is an inner surface that is
        nested inside the one represented by `surf_mesh` -- e.g. `surf_mesh` is
        a pial surface and `inner_mesh` a white matter surface. In this case
        nodes in both meshes must correspond: node i in `surf_mesh` is just
        across the gray matter thickness from node i in `inner_mesh`. Image
        values for index i are then sampled along the line joining these two
        points (if `kind` is 'auto' or 'depth').

    depth : sequence of floats or None, optional
        The cortical depth of samples. If provided, n_samples is ignored.
        When `inner_mesh` is provided, each element of `depth` is a fraction of
        the distance from `mesh` to `inner_mesh`: 0 is exactly on the outer
        surface, .5 is halfway, 1. is exactly on the inner surface. `depth`
        entries can be negative or greater than 1.
        When `inner_mesh` is not provided and `kind` is "line", each element of
        `depth` is a fraction of `radius` along the inwards normal at each mesh
        node. For example if `radius==1` and `depth==[-.5, 0.]`, for each node
        values will be sampled .5 mm outside of the surface and exactly at the
        node position.
        This parameter is not supported for the "ball" strategy so passing
        `depth` when `kind=="ball"` results in a `ValueError`.

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

    Three strategies are available to select these positions.

        - with 'depth', data is sampled at various cortical depths between
          corresponding nodes of `surface_mesh` and `inner_mesh` (which can be,
          for example, a pial surface and a white matter surface).
        - 'ball' uses points regularly spaced in a ball centered at the mesh
          vertex. The radius of the ball is controlled by the parameter
          `radius`.
        - 'line' starts by drawing the normal to the mesh passing through this
          vertex. It then selects a segment of this normal, centered at the
          vertex, of length 2 * `radius`. Image intensities are measured at
          points regularly spaced on this normal segment, or at positions
          determined by `depth`.
        - ('auto' chooses 'depth' if `inner_mesh` is provided and 'line'
          otherwise)

    You can control how many samples are drawn by setting `n_samples`, or their
    position by setting `depth`.

    Once the sampling positions are chosen, those that fall outside of the 3d
    image (or outside of the mask if you provided one) are discarded. If all
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

    Warnings
    --------
    This function is experimental and details such as the interpolation method
    are subject to change.

    """
    sampling_schemes = {'linear': _interpolation_sampling,
                        'nearest': _nearest_voxel_sampling}
    if interpolation not in sampling_schemes:
        raise ValueError('"interpolation" should be one of {}'.format(
            tuple(sampling_schemes.keys())))
    img = load_img(img)
    if mask_img is not None:
        mask_img = _utils.check_niimg(mask_img)
        mask = get_data(resampling.resample_to_img(
            mask_img, img, interpolation='nearest', copy=False))
    else:
        mask = None
    original_dimension = len(img.shape)
    img = _utils.check_niimg(img, atleast_4d=True)
    frames = np.rollaxis(get_data(img), -1)
    mesh = load_surf_mesh(surf_mesh)
    if inner_mesh is not None:
        inner_mesh = load_surf_mesh(inner_mesh)
    sampling = sampling_schemes[interpolation]
    texture = sampling(
        frames, mesh, img.affine, radius=radius, kind=kind,
        n_points=n_samples, mask=mask, inner_mesh=inner_mesh, depth=depth)
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
        .thickness, .curv, .sulc, .annot, .label), lists of 1D data files are
        returned as 2D arrays, or a Numpy array containing surface data.

    Returns
    -------
    data : numpy.ndarray
        An array containing surface data

    """
    # if the input is a filename, load it
    if isinstance(surf_data, str):

        # resolve globbing
        file_list = _resolve_globbing(surf_data)
        # _resolve_globbing handles empty lists

        for f in range(len(file_list)):
            surf_data = file_list[f]
            if (surf_data.endswith('nii') or surf_data.endswith('nii.gz') or
                    surf_data.endswith('mgz')):
                data_part = np.squeeze(get_data(nibabel.load(surf_data)))
            elif (
                surf_data.endswith('area')
                or surf_data.endswith('curv')
                or surf_data.endswith('sulc')
                or surf_data.endswith('thickness')
            ):
                data_part = fs.io.read_morph_data(surf_data)
            elif surf_data.endswith('annot'):
                data_part = fs.io.read_annot(surf_data)[0]
            elif surf_data.endswith('label'):
                data_part = fs.io.read_label(surf_data)
            elif surf_data.endswith('gii'):
                if LooseVersion(nibabel.__version__) >= LooseVersion('2.1.0'):
                    gii = nibabel.load(surf_data)
                else:
                    gii = gifti.read(surf_data)
                data_part = _gifti_img_to_data(gii)
            elif surf_data.endswith('gii.gz'):
                gii = _load_surf_files_gifti_gzip(surf_data)
                data_part = _gifti_img_to_data(gii)
            else:
                raise ValueError(('The input type is not recognized. %r was '
                                  'given while valid inputs are a Numpy array '
                                  'or one of the following file formats: .gii,'
                                  ' .gii.gz, .mgz, .nii, .nii.gz, Freesurfer '
                                  'specific files such as .area, .curv, .sulc,'
                                  ' .thickness, .annot, .label') % surf_data)

            if len(data_part.shape) == 1:
                data_part = data_part[:, np.newaxis]
            if f == 0:
                data = data_part
            elif f > 0:
                try:
                    data = np.concatenate((data, data_part), axis=1)
                except ValueError:
                    raise ValueError('When more than one file is input, all '
                                     'files must contain data with the same '
                                     'shape in axis=0')

    # if the input is a numpy array
    elif isinstance(surf_data, np.ndarray):
        data = surf_data
    else:
        raise ValueError('The input type is not recognized. '
                         'Valid inputs are a Numpy array or one of the '
                         'following file formats: .gii, .gii.gz, .mgz, .nii, '
                         '.nii.gz, Freesurfer specific files such as .area, '
                         '.curv, .sulc, .thickness, .annot, .label')
    return np.squeeze(data)


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
    surf_mesh : str or numpy.ndarray or Mesh
        Either a file containing surface mesh geometry (valid formats
        are .gii .gii.gz or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or two Numpy arrays organized in a list,
        tuple or a namedtuple with the fields "coordinates" and "faces", or a
        Mesh object with "coordinates" and "faces" attributes.

    Returns
    --------
    mesh : Mesh
        With the fields "coordinates" and "faces", each containing a
        numpy.ndarray

    """

    # if input is a filename, try to load it
    if isinstance(surf_mesh, str):
        # resolve globbing
        file_list = _resolve_globbing(surf_mesh)
        if len(file_list) == 1:
            surf_mesh = file_list[0]
        elif len(file_list) > 1:
            # empty list is handled inside _resolve_globbing function
            raise ValueError(("More than one file matching path: %s \n"
                             "load_surf_mesh can only load one file at a time")
                             % surf_mesh)

        if (surf_mesh.endswith('orig') or surf_mesh.endswith('pial') or
                surf_mesh.endswith('white') or surf_mesh.endswith('sphere') or
                surf_mesh.endswith('inflated')):
            coords, faces = fs.io.read_geometry(surf_mesh)
            mesh = Mesh(coordinates=coords, faces=faces)
        elif surf_mesh.endswith('gii'):
            if LooseVersion(nibabel.__version__) >= LooseVersion('2.1.0'):
                gifti_img = nibabel.load(surf_mesh)
            else:
                gifti_img = gifti.read(surf_mesh)
            coords, faces = _gifti_img_to_mesh(gifti_img)
            mesh = Mesh(coordinates=coords, faces=faces)
        elif surf_mesh.endswith('.gii.gz'):
            gifti_img = _load_surf_files_gifti_gzip(surf_mesh)
            coords, faces = _gifti_img_to_mesh(gifti_img)
            mesh = Mesh(coordinates=coords, faces=faces)
        else:
            raise ValueError(('The input type is not recognized. %r was given '
                              'while valid inputs are one of the following '
                              'file formats: .gii, .gii.gz, Freesurfer '
                              'specific files such as .orig, .pial, .sphere, '
                              '.white, .inflated or two Numpy arrays organized '
                              'in a list, tuple or a namedtuple with the '
                              'fields "coordinates" and "faces"'
                              ) % surf_mesh)
    elif isinstance(surf_mesh, (list, tuple)):
        try:
            coords, faces = surf_mesh
            mesh = Mesh(coordinates=coords, faces=faces)
        except Exception:
            raise ValueError(('If a list or tuple is given as input, '
                              'it must have two elements, the first is '
                              'a Numpy array containing the x-y-z coordinates '
                              'of the mesh vertices, the second is a Numpy '
                              'array containing  the indices (into coords) of '
                              'the mesh faces. The input was a list with '
                              '%r elements.') % len(surf_mesh))
    elif (hasattr(surf_mesh, "faces") and hasattr(surf_mesh, "coordinates")):
        coords, faces = surf_mesh.coordinates, surf_mesh.faces
        mesh = Mesh(coordinates=coords, faces=faces)

    else:
        raise ValueError('The input type is not recognized. '
                         'Valid inputs are one of the following file '
                         'formats: .gii, .gii.gz, Freesurfer specific files '
                         'such as .orig, .pial, .sphere, .white, .inflated '
                         'or two Numpy arrays organized in a list, tuple or '
                         'a namedtuple with the fields "coordinates" and '
                         '"faces"')

    return mesh


def load_surface(surface):
    """Loads a surface.

    Parameters
    ----------
    surface : Surface-like (see description)
        The surface to be loaded.
        A surface can be:
            - a nilearn.surface.Surface
            - a sequence (mesh, data) where:
                - mesh can be:
                    - a nilearn.surface.Mesh
                    - a path to .gii or .gii.gz etc.
                    - a sequence of two numpy arrays,
                    the first containing vertex coordinates
                    and the second containing triangles.
                - data can be:
                    - a path to .gii or .gii.gz etc.
                    - a numpy array with shape (n_vertices,)
                    or (n_time_points, n_vertices)

    Returns
    --------
    surface : Surface
        With the fields "mesh" (Mesh object) and "data" (numpy.ndarray).

    """
    # Handle the case where we received a Surface
    # object with mesh and data attributes
    if hasattr(surface, "mesh") and hasattr(surface, "data"):
        mesh = load_surf_mesh(surface.mesh)
        data = load_surf_data(surface.data)
    # Handle the case where we received a sequence
    # (mesh, data)
    elif isinstance(surface, (list, tuple, np.ndarray)):
        if len(surface) == 2:
            mesh = load_surf_mesh(surface[0])
            data = load_surf_data(surface[1])
        else:
            raise ValueError("`load_surface` accepts iterables "
                             "of length 2 to define a surface. "
                             "You provided a {} of length {}.".format(
                                 type(surface), len(surface)))
    else:
        raise ValueError("Wrong parameter `surface` in `load_surface`. "
                         "Please refer to the documentation for more information.")
    return Surface(mesh, data)


def _check_mesh(mesh):
    """Check that mesh data is either a str, or a dict with sufficient
    entries.

    Used by plotting.surf_plotting.plot_img_on_surf and
    plotting.html_surface.full_brain_info

    """
    if isinstance(mesh, str):
        return datasets.fetch_surf_fsaverage(mesh)
    if not isinstance(mesh, collections.Mapping):
        raise TypeError("The mesh should be a str or a dictionary, "
                        "you provided: {}.".format(type(mesh).__name__))
    missing = {'pial_left', 'pial_right', 'sulc_left', 'sulc_right',
               'infl_left', 'infl_right'}.difference(mesh.keys())
    if missing:
        raise ValueError(
            "{} {} missing from the provided mesh dictionary".format(
                missing, ('are' if len(missing) > 1 else 'is')))
    return mesh


def check_mesh_and_data(mesh, data):
    """Load surface mesh and data, check that they have compatible shapes.

    Parameters
    ----------
    mesh : str or numpy.ndarray or Mesh
        Either a file containing surface mesh geometry (valid formats
        are .gii .gii.gz or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or two Numpy arrays organized in a list,
        tuple or a namedtuple with the fields "coordinates" and "faces", or a
        Mesh object with "coordinates" and "faces" attributes.

    data : str or numpy.ndarray
        Either a file containing surface data (valid format are .gii,
        .gii.gz, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label),
        lists of 1D data files are returned as 2D arrays,
        or a Numpy array containing surface data.

    Returns
    -------
    mesh : Mesh
        Checked mesh.

    data : numpy.ndarray
        Checked data.

    """
    mesh = load_surf_mesh(mesh)
    data = load_surf_data(data)
    # Check that mesh coordinates has a number of nodes
    # equal to the size of the data.
    if len(data) != len(mesh.coordinates):
        raise ValueError(
            'Mismatch between number of nodes in mesh ({}) and '
            'size of surface data ({})'.format(len(mesh.coordinates), len(data)))
    # Check that the indices of faces are consistent with the
    # mesh coordinates. That is, we shouldn't have an index
    # larger or equal to the length of the coordinates array.
    if mesh.faces.max() >= len(mesh.coordinates):
        raise ValueError(
            "Mismatch between the indices of faces and the number of nodes. "
            "Maximum face index is {} while coordinates array has length {}.".format(
                mesh.faces.max(), len(mesh.coordinates)))
    return mesh, data


def check_surface(surface):
    """Load a surface as a Surface object.
    This function will make sure that the surfaces's
    mesh and data have compatible shapes.

    Parameters
    ----------
    surface : Surface-like (see description)
        The surface to be loaded.
        A surface can be:
            - a nilearn.surface.Surface
            - a sequence (mesh, data) where:
                - mesh can be:
                    - a nilearn.surface.Mesh
                    - a path to .gii or .gii.gz etc.
                    - a sequence of two numpy arrays,
                    the first containing vertex coordinates
                    and the second containing triangles.
                - data can be:
                    - a path to .gii or .gii.gz etc.
                    - a numpy array with shape (n_vertices,)
                    or (n_time_points, n_vertices)

    Returns
    -------
    surface : Surface
        Checked surface object.

    """
    surface = load_surface(surface)
    mesh, data = check_mesh_and_data(surface.mesh,
                                     surface.data)
    return Surface(mesh, data)
