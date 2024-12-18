"""Functions for surface manipulation."""

import abc
import gzip
import pathlib
import warnings
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import sklearn.cluster
import sklearn.preprocessing
from nibabel import freesurfer as fs
from nibabel import gifti, load, nifti1
from scipy import interpolate, sparse
from sklearn.exceptions import EfficiencyWarning

from nilearn import _utils
from nilearn._utils import stringify_path
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils.path_finding import resolve_globbing
from nilearn.image import get_data, load_img, resampling


def _uniform_ball_cloud(n_points=20, dim=3, n_monte_carlo=50000):
    """Get points uniformly spaced in the unit ball."""
    rng = np.random.RandomState(0)
    mc_cube = rng.uniform(-1, 1, size=(n_monte_carlo, dim))
    mc_ball = mc_cube[(mc_cube**2).sum(axis=1) <= 1.0]
    centroids, *_ = sklearn.cluster.k_means(
        mc_ball, n_clusters=n_points, random_state=0
    )
    return centroids


def _load_uniform_ball_cloud(n_points=20):
    stored_points = (
        Path(__file__, "..", "data", f"ball_cloud_{n_points}_samples.csv")
    ).resolve()
    if stored_points.is_file():
        points = np.loadtxt(stored_points)
        return points
    warnings.warn(
        "Cached sample positions are provided for "
        "n_samples = 10, 20, 40, 80, 160. Since the number of samples does "
        "have a big impact on the result, we strongly recommend using one "
        'of these values when using kind="ball" for much better performance.',
        EfficiencyWarning,
        stacklevel=3,
    )
    return _uniform_ball_cloud(n_points=n_points)


def _face_outer_normals(mesh):
    """Get the normal to each triangle in a mesh.

    They are the outer normals if the mesh respects the convention that the
    direction given by the direct order of a triangle's vertices (right-hand
    rule) points outwards.

    """
    mesh = load_surf_mesh(mesh)
    vertices = mesh.coordinates
    faces = mesh.faces
    face_vertices = vertices[faces]
    # The right-hand rule gives the direction of the outer normal
    normals = np.cross(
        face_vertices[:, 1, :] - face_vertices[:, 0, :],
        face_vertices[:, 2, :] - face_vertices[:, 0, :],
    )
    normals = sklearn.preprocessing.normalize(normals)
    return normals


def _surrounding_faces(mesh):
    """Get matrix indicating which faces the nodes belong to.

    i, j is set if node i is a vertex of triangle j.

    """
    mesh = load_surf_mesh(mesh)
    vertices = mesh.coordinates
    faces = mesh.faces
    n_faces = faces.shape[0]
    return sparse.csr_matrix(
        (
            np.ones(3 * n_faces),
            (faces.ravel(), np.tile(np.arange(n_faces), (3, 1)).T.ravel()),
        ),
        (vertices.shape[0], n_faces),
    )


def _vertex_outer_normals(mesh):
    """Get the normal at each vertex in a triangular mesh.

    They are the outer normals if the mesh respects the convention that the
    direction given by the direct order of a triangle's vertices (right-hand
    rule) points outwards.

    """
    vertex_faces = _surrounding_faces(mesh)
    face_normals = _face_outer_normals(mesh)
    normals = vertex_faces.dot(face_normals)
    return sklearn.preprocessing.normalize(normals)


def _sample_locations_between_surfaces(
    mesh, inner_mesh, affine, n_points=10, depth=None
):
    outer_vertices = load_surf_mesh(mesh).coordinates
    inner_vertices = load_surf_mesh(inner_mesh).coordinates

    if depth is None:
        steps = np.linspace(0, 1, n_points)[:, None, None]
    else:
        steps = np.asarray(depth)[:, None, None]

    sample_locations = outer_vertices + steps * (
        inner_vertices - outer_vertices
    )
    sample_locations = np.rollaxis(sample_locations, 1)

    sample_locations_voxel_space = np.asarray(
        resampling.coord_transform(
            *np.vstack(sample_locations).T, affine=np.linalg.inv(affine)
        )
    ).T.reshape(sample_locations.shape)
    return sample_locations_voxel_space


def _ball_sample_locations(
    mesh, affine, ball_radius=3.0, n_points=20, depth=None
):
    """Locations to draw samples from to project volume data onto a mesh.

    For each mesh vertex, the locations of `n_points` points evenly spread in a
    ball around the vertex are returned.

    Parameters
    ----------
    mesh : pair of :obj:`numpy.ndarray`s.
        `mesh[0]` contains the 3d coordinates of the vertices
        (shape n_vertices, 3)
        `mesh[1]` contains, for each triangle,
        the indices into `mesh[0]` of its vertices (shape n_triangles, 3)

    affine : :obj:`numpy.ndarray` of shape (4, 4)
        Affine transformation from image voxels to the vertices' coordinate
        space.

    ball_radius : :obj:`float`, default=3.0
        Size in mm of the neighbourhood around each vertex in which to draw
        samples.

    n_points : :obj:`int`, default=20
        Number of samples to draw for each vertex.

    depth : `None`
        Raises a `ValueError` if not `None` because incompatible with this
        sampling strategy.

    Returns
    -------
    sample_location_voxel_space : :obj:`numpy.ndarray`, \
        shape (n_vertices, n_points, 3)
        The locations, in voxel space, from which to draw samples.
        First dimension iterates over mesh vertices, second dimension iterates
        over the sample points associated to a vertex, third dimension is x, y,
        z in voxel space.

    """
    if depth is not None:
        raise ValueError(
            "The 'ball' sampling strategy does not support "
            "the 'depth' parameter.\n"
            "To avoid this error with this strategy, set 'depth' to None."
        )
    vertices = load_surf_mesh(mesh).coordinates
    offsets_world_space = (
        _load_uniform_ball_cloud(n_points=n_points) * ball_radius
    )
    mesh_voxel_space = np.asarray(
        resampling.coord_transform(*vertices.T, affine=np.linalg.inv(affine))
    ).T
    linear_map = np.eye(affine.shape[0])
    linear_map[:-1, :-1] = affine[:-1, :-1]
    offsets_voxel_space = np.asarray(
        resampling.coord_transform(
            *offsets_world_space.T, affine=np.linalg.inv(linear_map)
        )
    ).T
    sample_locations_voxel_space = (
        mesh_voxel_space[:, np.newaxis, :] + offsets_voxel_space[np.newaxis, :]
    )
    return sample_locations_voxel_space


def _line_sample_locations(
    mesh, affine, segment_half_width=3.0, n_points=10, depth=None
):
    """Locations to draw samples from to project volume data onto a mesh.

    For each mesh vertex, the locations of `n_points` points evenly spread in a
    segment of the normal to the vertex are returned. The line segment has
    length 2 * `segment_half_width` and is centered at the vertex.

    Parameters
    ----------
    mesh : pair of :obj:`numpy.ndarray`
        `mesh[0]` contains the 3d coordinates of the vertices
        (shape n_vertices, 3)
        `mesh[1]` contains, for each triangle,
        the indices into `mesh[0]` of its vertices (shape n_triangles, 3)

    affine : :obj:`numpy.ndarray` of shape (4, 4)
        Affine transformation from image voxels to the vertices' coordinate
        space.

    segment_half_width : :obj:`float`, default=3.0
        Size in mm of the neighbourhood around each vertex in which to draw
        samples.

    n_points : :obj:`int`, default=10
        Number of samples to draw for each vertex.

    depth : sequence of :obj:`float` or None, optional
        Cortical depth, expressed as a fraction of segment_half_width.
        Overrides n_points.

    Returns
    -------
    sample_location_voxel_space : :obj:`numpy.ndarray`, \
            shape (n_vertices, n_points, 3)
        The locations, in voxel space, from which to draw samples.
        First dimension iterates over mesh vertices, second dimension iterates
        over the sample points associated to a vertex, third dimension is x, y,
        z in voxel space.

    """
    vertices = load_surf_mesh(mesh).coordinates
    normals = _vertex_outer_normals(mesh)
    if depth is None:
        offsets = np.linspace(
            segment_half_width, -segment_half_width, n_points
        )
    else:
        offsets = -segment_half_width * np.asarray(depth)
    sample_locations = (
        vertices[np.newaxis, :, :]
        + normals * offsets[:, np.newaxis, np.newaxis]
    )
    sample_locations = np.rollaxis(sample_locations, 1)
    sample_locations_voxel_space = np.asarray(
        resampling.coord_transform(
            *np.vstack(sample_locations).T, affine=np.linalg.inv(affine)
        )
    ).T.reshape(sample_locations.shape)
    return sample_locations_voxel_space


def _choose_kind(kind, inner_mesh):
    if kind == "depth" and inner_mesh is None:
        raise TypeError(
            "'inner_mesh' must be provided to use "
            "the 'depth' sampling strategy"
        )
    if kind == "auto":
        kind = "line" if inner_mesh is None else "depth"
    return kind


def _sample_locations(
    mesh,
    affine,
    radius,
    kind="auto",
    n_points=None,
    inner_mesh=None,
    depth=None,
):
    """Get either ball or line sample locations."""
    kind = _choose_kind(kind, inner_mesh)
    kwargs = {} if n_points is None else {"n_points": n_points}
    projectors = {
        "line": (_line_sample_locations, {"segment_half_width": radius}),
        "ball": (_ball_sample_locations, {"ball_radius": radius}),
        "depth": (
            _sample_locations_between_surfaces,
            {"inner_mesh": inner_mesh},
        ),
    }
    if kind not in projectors:
        raise ValueError(f'"kind" must be one of {tuple(projectors.keys())}')
    projector, extra_kwargs = projectors[kind]
    # let the projector choose the default for n_points
    # (for example a ball probably needs more than a line)
    sample_locations = projector(
        mesh=mesh, affine=affine, depth=depth, **kwargs, **extra_kwargs
    )
    return sample_locations


def _masked_indices(sample_locations, img_shape, mask=None):
    """Get the indices of sample points which should be ignored.

    Parameters
    ----------
    sample_locations : :obj:`numpy.ndarray`, shape(n_sample_locations, 3)
        The coordinates of candidate interpolation points.

    img_shape : :obj:`tuple`
        The dimensions of the image to be sampled.

    mask : :obj:`numpy.ndarray` of shape img_shape or `None`, optional
        Part of the image to be masked. If `None`, don't apply any mask.

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
        kept[kept] = mask[indices[:, 0], indices[:, 1], indices[:, 2]] != 0
    return ~kept


def _projection_matrix(
    mesh,
    affine,
    img_shape,
    kind="auto",
    radius=3.0,
    n_points=None,
    mask=None,
    inner_mesh=None,
    depth=None,
):
    """Get a sparse matrix that projects volume data onto a mesh.

    Parameters
    ----------
    mesh : :obj:`str` or :obj:`numpy.ndarray`
        Either a file containing surface mesh geometry (valid formats
        are .gii or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or a list of two Numpy arrays,
        the first containing the x-y-z coordinates of the mesh
        vertices, the second containing the indices (into coords)
        of the mesh faces.

    affine : :obj:`numpy.ndarray` of shape (4, 4)
        Affine transformation from image voxels to the vertices' coordinate
        space.

    img_shape : 3-tuple of :obj:`int`
        The shape of the image to be projected.

    kind : {'auto', 'depth', 'line', 'ball'}, default='auto'
        The strategy used to sample image intensities around each vertex.
        Ignored if `inner_mesh` is not None.

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

    radius : :obj:`float`, default=3.0
        The size (in mm) of the neighbourhood from which samples are drawn
        around each node. Ignored if `inner_mesh` is not `None`.

    n_points : :obj:`int` or None, optional
        How many samples are drawn around each vertex and averaged. If `None`,
        use a reasonable default for the chosen sampling strategy (20 for
        'ball' or 10 for lines ie using `line` or an `inner_mesh`).
        For performance reasons, if using kind="ball", choose `n_points` in
        [10, 20, 40, 80, 160] (default is 20), because cached positions are
        available.

    mask : :obj:`numpy.ndarray` of shape img_shape or `None`, optional
        Part of the image to be masked. If `None`, don't apply any mask.

    inner_mesh : :obj:`str` or :obj:`numpy.ndarray`, optional
        Either a file containing surface mesh or a pair of ndarrays
        (coordinates, triangles). If provided this is an inner surface that is
        nested inside the one represented by `mesh` -- e.g. `mesh` is a pial
        surface and `inner_mesh` a white matter surface. In this case nodes in
        both meshes must correspond: node i in `mesh` is just across the gray
        matter thickness from node i in `inner_mesh`. Image values for index i
        are then sampled along the line joining these two points (if `kind` is
        'auto' or 'depth').

    depth : sequence of :obj:`float` or `None`, optional
        Cortical depth, expressed as a fraction of segment_half_width.
        overrides n_points. Should be None if kind is 'ball'

    Returns
    -------
    proj : :obj:`scipy.sparse.csr_matrix`
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
        raise ValueError("mask should have shape img_shape")
    mesh = load_surf_mesh(mesh)
    sample_locations = _sample_locations(
        mesh,
        affine,
        kind=kind,
        radius=radius,
        n_points=n_points,
        inner_mesh=inner_mesh,
        depth=depth,
    )
    sample_locations = np.asarray(np.round(sample_locations), dtype=int)
    n_vertices, n_points, _ = sample_locations.shape
    masked = _masked_indices(np.vstack(sample_locations), img_shape, mask=mask)
    sample_locations = np.rollaxis(sample_locations, -1)
    sample_indices = np.ravel_multi_index(
        sample_locations, img_shape, mode="clip"
    ).ravel()
    row_indices, _ = np.mgrid[:n_vertices, :n_points]
    row_indices = row_indices.ravel()
    row_indices = row_indices[~masked]
    sample_indices = sample_indices[~masked]
    weights = np.ones(len(row_indices))
    proj = sparse.csr_matrix(
        (weights, (row_indices, sample_indices.ravel())),
        shape=(n_vertices, np.prod(img_shape)),
    )
    proj = sklearn.preprocessing.normalize(proj, axis=1, norm="l1")
    return proj


def _nearest_voxel_sampling(
    images,
    mesh,
    affine,
    kind="auto",
    radius=3.0,
    n_points=None,
    mask=None,
    inner_mesh=None,
    depth=None,
):
    """In each image, measure the intensity at each node of the mesh.

    Image intensity at each sample point is that of the nearest voxel.
    A 2-d array is returned, where each row corresponds to an image and each
    column to a mesh vertex.
    See documentation of vol_to_surf for details.

    """
    proj = _projection_matrix(
        mesh,
        affine,
        images[0].shape,
        kind=kind,
        radius=radius,
        n_points=n_points,
        mask=mask,
        inner_mesh=inner_mesh,
        depth=depth,
    )
    data = np.asarray(images).reshape(len(images), -1).T
    texture = proj.dot(data)
    # if all samples around a mesh vertex are outside the image,
    # there is no reasonable value to assign to this vertex.
    # in this case we return NaN for this vertex.
    texture[np.asarray(proj.sum(axis=1) == 0).ravel()] = np.nan
    return texture.T


def _interpolation_sampling(
    images,
    mesh,
    affine,
    kind="auto",
    radius=3,
    n_points=None,
    mask=None,
    inner_mesh=None,
    depth=None,
):
    """In each image, measure the intensity at each node of the mesh.

    Image intensity at each sample point is computed with trilinear
    interpolation.
    A 2-d array is returned, where each row corresponds to an image and each
    column to a mesh vertex.
    See documentation of vol_to_surf for details.

    """
    sample_locations = _sample_locations(
        mesh,
        affine,
        kind=kind,
        radius=radius,
        n_points=n_points,
        inner_mesh=inner_mesh,
        depth=depth,
    )
    n_vertices, n_points, _ = sample_locations.shape
    grid = [np.arange(size) for size in images[0].shape]
    interp_locations = np.vstack(sample_locations)
    masked = _masked_indices(interp_locations, images[0].shape, mask=mask)
    # loop over images rather than building a big array to use less memory
    all_samples = []
    for img in images:
        interpolator = interpolate.RegularGridInterpolator(
            grid, img, bounds_error=False, method="linear", fill_value=None
        )
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


def vol_to_surf(
    img,
    surf_mesh,
    radius=3.0,
    interpolation="linear",
    kind="auto",
    n_samples=None,
    mask_img=None,
    inner_mesh=None,
    depth=None,
):
    """Extract surface data from a Nifti image.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    img : Niimg-like object, 3d or 4d.
        See :ref:`extracting_data`.

    surf_mesh : :obj:`str`, :obj:`pathlib.Path`, :obj:`numpy.ndarray`, or \
                :obj:`~nilearn.surface.InMemoryMesh`
        Either a file containing surface :term:`mesh` geometry
        (valid formats are .gii or Freesurfer specific files
        such as .orig, .pial, .sphere, .white, .inflated)
        or a :obj:`~nilearn.surface.InMemoryMesh` object with "coordinates"
        and "faces" attributes.

    radius : :obj:`float`, default=3.0
        The size (in mm) of the neighbourhood from which samples are drawn
        around each node. Ignored if `inner_mesh` is provided.

    interpolation : {'linear', 'nearest'}, default='linear'
        How the image intensity is measured at a sample point.

        - 'linear':
            Use a trilinear interpolation of neighboring voxels.
        - 'nearest':
            Use the intensity of the nearest voxel.

        For one image, the speed difference is small, 'linear' takes about x1.5
        more time. For many images, 'nearest' scales much better, up to x20
        faster.

    kind : {'auto', 'depth', 'line', 'ball'}, default='auto'
        The strategy used to sample image intensities around each vertex.

        - 'auto':
            Chooses 'depth' if `inner_mesh` is provided and 'line' otherwise.
        - 'depth':
            `inner_mesh` must be a :term:`mesh`
            whose nodes correspond to those in `surf_mesh`.
            For example, `inner_mesh` could be a white matter
            surface mesh and `surf_mesh` a pial surface :term:`mesh`.
            Samples are placed between each pair of corresponding nodes
            at the specified cortical depths
            (regularly spaced by default, see `depth` parameter).
        - 'line':
            Samples are placed along the normal to the mesh, at the positions
            specified by `depth`, or by default regularly spaced over the
            interval [- `radius`, + `radius`].
        - 'ball':
            Samples are regularly spaced inside a ball centered at the mesh
            vertex.

    n_samples : :obj:`int` or `None`, optional
        How many samples are drawn around each :term:`vertex` and averaged.
        If `None`, use a reasonable default for the chosen sampling strategy
        (20 for 'ball' or 10 for 'line').
        For performance reasons, if using `kind` ="ball", choose `n_samples` in
        [10, 20, 40, 80, 160] (default is 20), because cached positions are
        available.

    mask_img : Niimg-like object or `None`, optional
        Samples falling out of this mask or out of the image are ignored.
        If `None`, don't apply any mask.

    inner_mesh : :obj:`str` or :obj:`numpy.ndarray`, optional
        Either a file containing a surface :term:`mesh` or a pair of ndarrays
        (coordinates, triangles). If provided this is an inner surface that is
        nested inside the one represented by `surf_mesh` -- e.g. `surf_mesh` is
        a pial surface and `inner_mesh` a white matter surface. In this case
        nodes in both :term:`meshes<mesh>` must correspond:
        node i in `surf_mesh` is just across the gray matter thickness
        from node i in `inner_mesh`.
        Image values for index i are then sampled along the line
        joining these two points (if `kind` is 'auto' or 'depth').

    depth : sequence of :obj:`float` or `None`, optional
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
    texture : :obj:`numpy.ndarray`, 1d or 2d.
        If 3D image is provided, a 1d vector is returned, containing one value
        for each :term:`mesh` node.
        If 4D image is provided, a 2d array is returned, where each row
        corresponds to a :term:`mesh` node.

    Notes
    -----
    This function computes a value for each vertex of the :term:`mesh`.
    In order to do so,
    it selects a few points in the volume surrounding that vertex,
    interpolates the image intensities at these sampling positions,
    and averages the results.

    Three strategies are available to select these positions.

        - with 'depth', data is sampled at various cortical depths between
          corresponding nodes of `surface_mesh` and `inner_mesh` (which can be,
          for example, a pial surface and a white matter surface). This is the
          recommended strategy when both the pial and white matter surfaces are
          available, which is the case for the fsaverage :term:`meshes<mesh>`.
        - 'ball' uses points regularly spaced in a ball centered
          at the :term:`mesh` vertex.
          The radius of the ball is controlled by the parameter `radius`.
        - 'line' starts by drawing the normal to the :term:`mesh`
          passing through this vertex.
          It then selects a segment of this normal,
          centered at the vertex, of length 2 * `radius`.
          Image intensities are measured at points regularly spaced
          on this normal segment, or at positions determined by `depth`.
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
    voxel, and 'linear' performs trilinear interpolation of neighboring
    voxels. 'linear' may give better results - for example, the projected
    values are more stable when resampling the 3d image or applying affine
    transformations to it. For one image, the speed difference is small,
    'linear' takes about x1.5 more time. For many images, 'nearest' scales much
    better, up to x20 faster.

    Once the 3d image has been interpolated at each sample point, the
    interpolated values are averaged to produce the value associated to this
    particular :term:`mesh` vertex.

    Examples
    --------
    When both the pial and white matter surface are available, the recommended
    approach is to provide the `inner_mesh` to rely on the 'depth' sampling
    strategy::

     >>> from nilearn import datasets, surface
     >>> fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
     >>> img = datasets.load_mni152_template(2)
     >>> surf_data = surface.vol_to_surf(
     ...     img,
     ...     surf_mesh=fsaverage["pial_left"],
     ...     inner_mesh=fsaverage["white_left"],
     ... )

    """
    sampling_schemes = {
        "linear": _interpolation_sampling,
        "nearest": _nearest_voxel_sampling,
    }
    if interpolation not in sampling_schemes:
        raise ValueError(
            "'interpolation' should be one of "
            f"{tuple(sampling_schemes.keys())}"
        )
    img = load_img(img)
    if mask_img is not None:
        mask_img = _utils.check_niimg(mask_img)
        mask = get_data(
            resampling.resample_to_img(
                mask_img,
                img,
                interpolation="nearest",
                copy=False,
                force_resample=False,  # TODO update to True in 0.13.0
                copy_header=True,
            )
        )
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
        frames,
        mesh,
        img.affine,
        radius=radius,
        kind=kind,
        n_points=n_samples,
        mask=mask,
        inner_mesh=inner_mesh,
        depth=depth,
    )
    if original_dimension == 3:
        texture = texture[0]
    return texture.T


def _load_surf_files_gifti_gzip(surf_file):
    """Load surface data Gifti files which are gzipped.

    This function is used by load_surf_mesh and load_surf_data for
    extracting gzipped files.
    """
    with gzip.open(surf_file) as f:
        as_bytes = f.read()
    parser = gifti.GiftiImage.parser()
    parser.parse(as_bytes)
    return parser.img


def _gifti_img_to_data(gifti_img):
    """Load surface image e.g. sulcal depth or statistical map \
        in nibabel.gifti.GiftiImage to data.

    Used by load_surf_data function in common to surface sulcal data
    acceptable to .gii or .gii.gz

    """
    if not gifti_img.darrays:
        raise ValueError("Gifti must contain at least one data array")

    if len(gifti_img.darrays) == 1:
        return np.asarray([gifti_img.darrays[0].data]).T.squeeze()

    return np.asarray(
        [arr.data for arr in gifti_img.darrays], dtype=object
    ).T.squeeze()


FREESURFER_MESH_EXTENSIONS = ("orig", "pial", "sphere", "white", "inflated")

FREESURFER_DATA_EXTENSIONS = (
    "area",
    "curv",
    "sulc",
    "thickness",
    "label",
    "annot",
)

DATA_EXTENSIONS = ("gii", "gii.gz", "mgz", "nii", "nii.gz")


def _stringify(word_list):
    sep = "', '."
    return f"'.{sep.join(word_list)[:-3]}'"


# function to figure out datatype and load data
def load_surf_data(surf_data):
    """Load data to be represented on a surface mesh.

    Parameters
    ----------
    surf_data : :obj:`str`, :obj:`pathlib.Path`, or :obj:`numpy.ndarray`
        Either a file containing surface data (valid format are .gii,
        .gii.gz, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .curv, .sulc, .annot, .label), lists of 1D data files are
        returned as 2D arrays, or a Numpy array containing surface data.

    Returns
    -------
    data : :obj:`numpy.ndarray`
        An array containing surface data

    """
    # if the input is a filename, load it
    surf_data = stringify_path(surf_data)

    if not isinstance(surf_data, (str, np.ndarray)):
        raise ValueError(
            "The input type is not recognized. "
            "Valid inputs are a Numpy array or one of the "
            "following file formats: "
            f"{_stringify(DATA_EXTENSIONS)}, "
            "Freesurfer specific files such as "
            f"{_stringify(FREESURFER_DATA_EXTENSIONS)}."
        )

    if isinstance(surf_data, str):
        # resolve globbing
        file_list = resolve_globbing(surf_data)
        # resolve_globbing handles empty lists

        for i, surf_data in enumerate(file_list):
            surf_data = str(surf_data)

            check_extensions(
                surf_data, DATA_EXTENSIONS, FREESURFER_DATA_EXTENSIONS
            )

            if surf_data.endswith(("nii", "nii.gz", "mgz")):
                data_part = np.squeeze(get_data(load(surf_data)))
            elif surf_data.endswith(("area", "curv", "sulc", "thickness")):
                data_part = fs.io.read_morph_data(surf_data)
            elif surf_data.endswith("annot"):
                data_part = fs.io.read_annot(surf_data)[0]
            elif surf_data.endswith("label"):
                data_part = fs.io.read_label(surf_data)
            elif surf_data.endswith("gii"):
                data_part = _gifti_img_to_data(load(surf_data))
            elif surf_data.endswith("gii.gz"):
                gii = _load_surf_files_gifti_gzip(surf_data)
                data_part = _gifti_img_to_data(gii)

            if len(data_part.shape) == 1:
                data_part = data_part[:, np.newaxis]
            if i == 0:
                data = data_part
            else:
                try:
                    data = np.concatenate((data, data_part), axis=1)
                except ValueError:
                    raise ValueError(
                        "When more than one file is input, "
                        "all files must contain data "
                        "with the same shape in axis=0."
                    )

    # if the input is a numpy array
    elif isinstance(surf_data, np.ndarray):
        data = surf_data

    return np.squeeze(data)


def check_extensions(surf_data, data_extensions, freesurfer_data_extensions):
    """Check the extension of the input file.

    Should either be one one of the supported data formats
    or one of freesurfer data formats.

    Raises
    ------
    ValueError
        When the input is a string or a path with an extension
        that does not match one of the supported ones.
    """
    if isinstance(surf_data, Path):
        surf_data = str(surf_data)
    if isinstance(surf_data, str) and (
        not any(
            surf_data.endswith(x)
            for x in data_extensions + freesurfer_data_extensions
        )
    ):
        raise ValueError(
            "The input type is not recognized. "
            f"{surf_data!r} was given "
            "while valid inputs are a Numpy array "
            "or one of the following file formats: "
            f"{_stringify(data_extensions)}, "
            "Freesurfer specific files such as "
            f"{_stringify(freesurfer_data_extensions)}."
        )


def _gifti_img_to_mesh(gifti_img):
    """Load surface image in nibabel.gifti.GiftiImage to data.

    Used by load_surf_mesh function in common to surface mesh
    acceptable to .gii or .gii.gz

    """
    error_message = (
        "The surf_mesh input is not recognized. "
        "Valid Freesurfer surface mesh inputs are: "
        f"{_stringify(FREESURFER_MESH_EXTENSIONS)}."
        "You provided input which have "
        "no {0} or of empty value={1}"
    )
    try:
        coords = gifti_img.get_arrays_from_intent(
            nifti1.intent_codes["NIFTI_INTENT_POINTSET"]
        )[0].data
    except IndexError:
        raise ValueError(
            error_message.format(
                "NIFTI_INTENT_POINTSET",
                gifti_img.get_arrays_from_intent(
                    nifti1.intent_codes["NIFTI_INTENT_POINTSET"]
                ),
            )
        )
    try:
        faces = gifti_img.get_arrays_from_intent(
            nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]
        )[0].data
    except IndexError:
        raise ValueError(
            error_message.format(
                "NIFTI_INTENT_TRIANGLE",
                gifti_img.get_arrays_from_intent(
                    nifti1.intent_codes["NIFTI_INTENT_TRIANGLE"]
                ),
            )
        )
    return coords, faces


def check_mesh_is_fsaverage(mesh):
    """Check that :term:`mesh` data is either a :obj:`str`, or a :obj:`dict`
    with sufficient entries. Basically ensures that the mesh data is
    Freesurfer-like fsaverage data.

    Used by plotting.surf_plotting.plot_img_on_surf and
    plotting.html_surface._full_brain_info.
    """
    if isinstance(mesh, str):
        # avoid circular imports
        from nilearn.datasets import fetch_surf_fsaverage

        return fetch_surf_fsaverage(mesh)
    if not isinstance(mesh, Mapping):
        raise TypeError(
            "The mesh should be a str or a dictionary, "
            f"you provided: {type(mesh).__name__}."
        )
    missing = {
        "pial_left",
        "pial_right",
        "sulc_left",
        "sulc_right",
        "infl_left",
        "infl_right",
    }.difference(mesh.keys())
    if missing:
        raise ValueError(
            f"{missing} {'are' if len(missing) > 1 else 'is'} "
            "missing from the provided mesh dictionary"
        )
    return mesh


def check_mesh_and_data(mesh, data):
    """Load surface :term:`mesh` and data, \
       check that they have compatible shapes.

    Parameters
    ----------
    mesh : :obj:`str` or :obj:`numpy.ndarray` or \
           :obj:`~nilearn.surface.InMemoryMesh`
        Either a file containing surface :term:`mesh` geometry (valid formats
        are .gii .gii.gz or Freesurfer specific files such as .orig, .pial,
        .sphere, .white, .inflated) or two Numpy arrays organized in a list,
        tuple or a namedtuple with the fields "coordinates" and "faces", or a
        :obj:`~nilearn.surface.InMemoryMesh` object with "coordinates" and
        "faces" attributes.
    data : :obj:`str` or :obj:`numpy.ndarray`
        Either a file containing surface data (valid format are .gii,
        .gii.gz, .mgz, .nii, .nii.gz, or Freesurfer specific files such as
        .thickness, .area, .curv, .sulc, .annot, .label),
        lists of 1D data files are returned as 2D arrays,
        or a Numpy array containing surface data.

    Returns
    -------
    mesh : :obj:`~nilearn.surface.InMemoryMesh`
        Checked :term:`mesh`.
    data : :obj:`numpy.ndarray`
        Checked data.
    """
    mesh = load_surf_mesh(mesh)
    data = load_surf_data(data)
    # Check that mesh coordinates has a number of nodes
    # equal to the size of the data.
    if len(data) != len(mesh.coordinates):
        raise ValueError(
            "Mismatch between number of nodes "
            f"in mesh ({len(mesh.coordinates)}) and "
            f"size of surface data ({len(data)})"
        )
    # Check that the indices of faces are consistent with the
    # mesh coordinates. That is, we shouldn't have an index
    # larger or equal to the length of the coordinates array.
    if mesh.faces.max() >= len(mesh.coordinates):
        raise ValueError(
            "Mismatch between the indices of faces and the number of nodes. "
            f"Maximum face index is {mesh.faces.max()} "
            f"while coordinates array has length {len(mesh.coordinates)}."
        )
    return mesh, data


# function to figure out datatype and load data
def load_surf_mesh(surf_mesh):
    """Load a surface :term:`mesh` geometry.

    Parameters
    ----------
    surf_mesh : :obj:`str`, :obj:`pathlib.Path`, or \
        :obj:`numpy.ndarray` or :obj:`~nilearn.surface.InMemoryMesh`
        Either a file containing surface :term:`mesh` geometry
        (valid formats are .gii .gii.gz or Freesurfer specific files
        such as .orig, .pial, .sphere, .white, .inflated)
        or two Numpy arrays organized in a list,
        tuple or a namedtuple with the fields "coordinates" and "faces",
        or an :obj:`~nilearn.surface.InMemoryMesh` object with "coordinates"
        and "faces" attributes.

    Returns
    -------
    mesh : :obj:`~nilearn.surface.InMemoryMesh`
        With the attributes "coordinates" and "faces", each containing a
        :obj:`numpy.ndarray`

    """
    # if input is a filename, try to load it
    surf_mesh = stringify_path(surf_mesh)
    if isinstance(surf_mesh, str):
        # resolve globbing
        file_list = resolve_globbing(surf_mesh)
        if len(file_list) > 1:
            # empty list is handled inside resolve_globbing function
            raise ValueError(
                f"More than one file matching path: {surf_mesh}\n"
                "load_surf_mesh can only load one file at a time."
            )
        surf_mesh = str(file_list[0])

        if any(surf_mesh.endswith(x) for x in FREESURFER_MESH_EXTENSIONS):
            coords, faces, header = fs.io.read_geometry(
                surf_mesh, read_metadata=True
            )
            # See https://github.com/nilearn/nilearn/pull/3235
            if "cras" in header:
                coords += header["cras"]
            mesh = InMemoryMesh(coordinates=coords, faces=faces)
        elif surf_mesh.endswith("gii"):
            coords, faces = _gifti_img_to_mesh(load(surf_mesh))
            mesh = InMemoryMesh(coordinates=coords, faces=faces)
        elif surf_mesh.endswith("gii.gz"):
            gifti_img = _load_surf_files_gifti_gzip(surf_mesh)
            coords, faces = _gifti_img_to_mesh(gifti_img)
            mesh = InMemoryMesh(coordinates=coords, faces=faces)
        else:
            raise ValueError(
                "The input type is not recognized. "
                f"{surf_mesh!r} was given "
                "while valid inputs are one of the following "
                "file formats: .gii, .gii.gz, "
                "Freesurfer specific files such as "
                f"{_stringify(FREESURFER_MESH_EXTENSIONS)}, "
                "two Numpy arrays organized in a list, tuple "
                "or a namedtuple with the "
                'fields "coordinates" and "faces".'
            )
    elif isinstance(surf_mesh, (list, tuple)):
        try:
            coords, faces = surf_mesh
            mesh = InMemoryMesh(coordinates=coords, faces=faces)
        except Exception:
            raise ValueError(
                "If a list or tuple is given as input, "
                "it must have two elements, the first is "
                "a Numpy array containing the x-y-z coordinates "
                "of the mesh vertices, the second is a Numpy "
                "array containing  the indices (into coords) of "
                "the mesh faces. The input was a list with "
                f"{len(surf_mesh)} elements."
            )
    elif hasattr(surf_mesh, "faces") and hasattr(surf_mesh, "coordinates"):
        coords, faces = surf_mesh.coordinates, surf_mesh.faces
        mesh = InMemoryMesh(coordinates=coords, faces=faces)

    else:
        raise ValueError(
            "The input type is not recognized. "
            "Valid inputs are one of the following file "
            "formats: .gii, .gii.gz, "
            "Freesurfer specific files such as "
            f"{_stringify(FREESURFER_MESH_EXTENSIONS)}"
            "or two Numpy arrays organized in a list, tuple or "
            'a namedtuple with the fields "coordinates" and '
            '"faces"'
        )

    return mesh


class PolyData:
    """A collection of data arrays.

    It is a shallow wrapper around the ``parts`` dictionary, which cannot be
    empty and whose keys must be a subset of {"left", "right"}.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    left : 1/2D :obj:`numpy.ndarray` or :obj:`str` or :obj:`pathlib.Path` \
           or None, default = None

    right : 1/2D :obj:`numpy.ndarray` or :obj:`str` or :obj:`pathlib.Path` \
            or None, default = None

    Attributes
    ----------
    parts : :obj:`dict` of 2D :obj:`numpy.ndarray` (n_vertices, n_timepoints)

    shape : :obj:`tuple` of :obj:`int`
            The first dimension corresponds to the vertices:
            the typical shape of the
            data for a hemisphere is ``(n_vertices, n_time_points)``.

    Examples
    --------
    >>> import numpy as np
    >>> from nilearn.surface import PolyData
    >>> n_time_points = 10
    >>> n_left_vertices = 5
    >>> n_right_vertices = 7
    >>> left = np.ones((n_left_vertices, n_time_points))
    >>> right = np.ones((n_right_vertices, n_time_points))
    >>> PolyData(left=left, right=right)
    <PolyData (12, 10)>
    >>> PolyData(right=right)
    <PolyData (7, 10)>

    >>> PolyData()
    Traceback (most recent call last):
        ...
    ValueError: Cannot create an empty PolyData. ...
    """

    def __init__(self, left=None, right=None):
        if left is None and right is None:
            raise ValueError(
                "Cannot create an empty PolyData. "
                "Either left or right (or both) must be provided."
            )

        parts = {}
        for hemi, param in zip(["left", "right"], [left, right]):
            if param is not None:
                if not isinstance(param, np.ndarray):
                    param = load_surf_data(param)
                parts[hemi] = param
        self.parts = parts

        self._check_parts()

    def _check_parts(self):
        parts = self.parts

        if len(parts) == 1:
            return

        if len(parts["left"].shape) != len(parts["right"].shape) or (
            len(parts["left"].shape) > 1
            and len(parts["right"].shape) > 1
            and parts["left"].shape[-1] != parts["right"].shape[-1]
        ):
            raise ValueError(
                f"Data arrays for keys 'left' and 'right' "
                "have incompatible shapes: "
                f"{parts['left'].shape} and {parts['right'].shape}"
            )

    @property
    def shape(self):
        """Shape of the data."""
        if len(self.parts) == 1:
            return next(iter(self.parts.values())).shape

        tmp = next(iter(self.parts.values()))

        sum_vertices = sum(p.shape[0] for p in self.parts.values())
        return (
            (sum_vertices, tmp.shape[1])
            if len(tmp.shape) == 2
            else (sum_vertices,)
        )

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.shape}>"

    def to_filename(self, filename):
        """Save data to gifti.

        Parameters
        ----------
        filename : :obj:`str` or :obj:`pathlib.Path`
                   If the filename contains `hemi-L`
                   then only the left part of the mesh will be saved.
                   If the filename contains `hemi-R`
                   then only the right part of the mesh will be saved.
                   If the filename contains neither of those,
                   then `_hemi-L` and `_hemi-R`
                   will be appended to the filename and both will be saved.
        """
        filename = _sanitize_filename(filename)

        if "hemi-L" not in filename.stem and "hemi-R" not in filename.stem:
            for hemi in ["L", "R"]:
                self.to_filename(
                    filename.with_stem(f"{filename.stem}_hemi-{hemi}")
                )
            return None

        if "hemi-L" in filename.stem:
            data = self.parts["left"]
        if "hemi-R" in filename.stem:
            data = self.parts["right"]

        _data_to_gifti(data, filename)


def at_least_2d(input):
    """Force surface image or polydata to be 2d."""
    if len(input.shape) == 2:
        return input

    if isinstance(input, SurfaceImage):
        input.data = at_least_2d(input.data)
        return input

    if len(input.shape) == 1:
        for k, v in input.parts.items():
            input.parts[k] = v.reshape((v.shape[0], 1))

    return input


class SurfaceMesh(abc.ABC):
    """A surface :term:`mesh` having vertex, \
    coordinates and faces (triangles).

    .. versionadded:: 0.11.0

    Attributes
    ----------
    n_vertices : int
        number of vertices
    """

    n_vertices: int

    # TODO those are properties are for compatibility with plot_surf_img.
    # But they should probably become functions as they can take some time to
    # return or even fail
    coordinates: np.ndarray
    faces: np.ndarray

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} with "
            f"{self.n_vertices} vertices and "
            f"{len(self.faces)} faces.>"
        )

    def to_gifti(self, gifti_file):
        """Write surface mesh to a Gifti file on disk.

        Parameters
        ----------
        gifti_file : :obj:`str` or :obj:`pathlib.Path`
            Filename to save the mesh to.
        """
        _mesh_to_gifti(self.coordinates, self.faces, gifti_file)


class InMemoryMesh(SurfaceMesh):
    """A surface mesh stored as in-memory numpy arrays.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    coordinates : :obj:`numpy.ndarray`

    faces : :obj:`numpy.ndarray`

    Attributes
    ----------
    n_vertices : int
        number of vertices
    """

    n_vertices: int

    coordinates: np.ndarray

    faces: np.ndarray

    def __init__(self, coordinates, faces):
        self.coordinates = coordinates
        self.faces = faces
        self.n_vertices = coordinates.shape[0]

    def __getitem__(self, index):
        if index == 0:
            return self.coordinates
        elif index == 1:
            return self.faces
        else:
            raise IndexError(
                "Index out of range. Use 0 for coordinates and 1 for faces."
            )

    def __iter__(self):
        return iter([self.coordinates, self.faces])


class FileMesh(SurfaceMesh):
    """A surface mesh stored in a Gifti or Freesurfer file.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    file_path : :obj:`str` or :obj:`pathlib.Path`
            Filename to read mesh from.
    """

    n_vertices: int

    file_path: pathlib.Path

    def __init__(self, file_path):
        self.file_path = pathlib.Path(file_path)
        self.n_vertices = load_surf_mesh(self.file_path).coordinates.shape[0]

    @property
    def coordinates(self):
        """Get x, y, z, values for each mesh vertex.

        Returns
        -------
        :obj:`numpy.ndarray`
        """
        return load_surf_mesh(self.file_path).coordinates

    @property
    def faces(self):
        """Get array of adjacent vertices.

        Returns
        -------
        :obj:`numpy.ndarray`
        """
        return load_surf_mesh(self.file_path).faces

    def loaded(self):
        """Load surface mesh into memory.

        Returns
        -------
        :obj:`nilearn.surface.InMemoryMesh`
        """
        loaded = load_surf_mesh(self.file_path)
        return InMemoryMesh(loaded.coordinates, loaded.faces)


class PolyMesh:
    """A collection of meshes.

    It is a shallow wrapper around the ``parts`` dictionary, which cannot be
    empty and whose keys must be a subset of {"left", "right"}.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    left : :obj:`str` or :obj:`pathlib.Path` \
                or :obj:`nilearn.surface.SurfaceMesh` or None, default=None
            SurfaceMesh for the left hemisphere.

    right : :obj:`str` or :obj:`pathlib.Path` \
                or :obj:`nilearn.surface.SurfaceMesh` or None, default=None
            SurfaceMesh for the right hemisphere.

    Attributes
    ----------
    n_vertices : int
        number of vertices
    """

    n_vertices: int

    def __init__(self, left=None, right=None) -> None:
        if left is None and right is None:
            raise ValueError(
                "Cannot create an empty PolyMesh. "
                "Either left or right (or both) must be provided."
            )

        self.parts = {}
        if left is not None:
            if not isinstance(left, SurfaceMesh):
                left = FileMesh(left).loaded()
            self.parts["left"] = left
        if right is not None:
            if not isinstance(right, SurfaceMesh):
                right = FileMesh(right).loaded()
            self.parts["right"] = right

        self.n_vertices = sum(p.n_vertices for p in self.parts.values())

    def to_filename(self, filename):
        """Save mesh to gifti.

        Parameters
        ----------
        filename : :obj:`str` or :obj:`pathlib.Path`
                   If the filename contains `hemi-L`
                   then only the left part of the mesh will be saved.
                   If the filename contains `hemi-R`
                   then only the right part of the mesh will be saved.
                   If the filename contains neither of those,
                   then `_hemi-L` and `_hemi-R`
                   will be appended to the filename and both will be saved.
        """
        filename = _sanitize_filename(filename)

        if "hemi-L" not in filename.stem and "hemi-R" not in filename.stem:
            for hemi in ["L", "R"]:
                self.to_filename(
                    filename.with_stem(f"{filename.stem}_hemi-{hemi}")
                )
            return None

        if "hemi-L" in filename.stem:
            mesh = self.parts["left"]
        if "hemi-R" in filename.stem:
            mesh = self.parts["right"]

        mesh.to_gifti(filename)


def _check_data_and_mesh_compat(mesh, data):
    """Check that mesh and data have the same keys and that shapes match.

    mesh : :obj:`nilearn.surface.PolyMesh`

    data : :obj:`nilearn.surface.PolyData`
    """
    data_keys, mesh_keys = set(data.parts.keys()), set(mesh.parts.keys())
    if data_keys != mesh_keys:
        diff = data_keys.symmetric_difference(mesh_keys)
        raise ValueError(
            "Data and mesh do not have the same keys. "
            f"Offending keys: {diff}"
        )
    for key in mesh_keys:
        if data.parts[key].shape[0] != mesh.parts[key].n_vertices:
            raise ValueError(
                f"Data shape does not match number of vertices for '{key}':\n"
                f"- data shape: {data.parts[key].shape}\n"
                f"- n vertices: {mesh.parts[key].n_vertices}"
            )


def _mesh_to_gifti(coordinates, faces, gifti_file):
    """Write surface mesh to gifti file on disk.

    Parameters
    ----------
    coordinates : :obj:`numpy.ndarray`
        a Numpy array containing the x-y-z coordinates of the mesh vertices

    faces : :obj:`numpy.ndarray`
        a Numpy array containing the indices (into coords) of the mesh faces.

    gifti_file : :obj:`str` or :obj:`pathlib.Path`
        name for the output gifti file.
    """
    gifti_file = Path(gifti_file)
    gifti_img = gifti.GiftiImage()
    coords_array = gifti.GiftiDataArray(
        coordinates, intent="NIFTI_INTENT_POINTSET", datatype="float32"
    )
    faces_array = gifti.GiftiDataArray(
        faces, intent="NIFTI_INTENT_TRIANGLE", datatype="int32"
    )
    gifti_img.add_gifti_data_array(coords_array)
    gifti_img.add_gifti_data_array(faces_array)
    gifti_img.to_filename(gifti_file)


def _data_to_gifti(data, gifti_file):
    """Save data from Polydata to a gifti file.

    Parameters
    ----------
    data : :obj:`numpy.ndarray`
        The data will be cast to np.uint8, np.int32) or np.float32
        as only the following are 'supported' for now:
        - NIFTI_TYPE_UINT8
        - NIFTI_TYPE_INT32
        - NIFTI_TYPE_FLOAT32
        See https://github.com/nipy/nibabel/blob/master/nibabel/gifti/gifti.py

    gifti_file : :obj:`str` or :obj:`pathlib.Path`
        name for the output gifti file.
    """
    if data.dtype in [np.uint16, np.uint32, np.uint64]:
        data = data.astype(np.uint8)
    elif data.dtype in [np.int8, np.int16, np.int64]:
        data = data.astype(np.int32)
    elif data.dtype in [np.float64]:
        data = data.astype(np.float32)

    if data.dtype == np.uint8:
        datatype = "NIFTI_TYPE_UINT8"
    elif data.dtype == np.int32:
        datatype = "NIFTI_TYPE_INT32"
    elif data.dtype == np.float32:
        datatype = "NIFTI_TYPE_FLOAT32"
    else:
        datatype = None
    darray = gifti.GiftiDataArray(data=data, datatype=datatype)

    gii = gifti.GiftiImage(darrays=[darray])
    gii.to_filename(Path(gifti_file))


def _sanitize_filename(filename):
    """Check filenames to write gifti.

    - add suffix .gii if missing
    - make sure that there is only one hemi entity in the filename

    Parameters
    ----------
    filename : :obj:`str` or :obj:`pathlib.Path`
        filename to check

    Returns
    -------
    :obj:`pathlib.Path`
    """
    filename = Path(filename)

    if not filename.suffix:
        filename = filename.with_suffix(".gii")
    if filename.suffix != ".gii":
        raise ValueError(
            "SurfaceMesh / Data should be saved as gifti files "
            "with the extension '.gii'.\n"
            f"Got '{filename.suffix}'."
        )

    if "hemi-L" in filename.stem and "hemi-R" in filename.stem:
        raise ValueError(
            "'filename' cannot contain both "
            "'hemi-L' and 'hemi-R'.\n"
            f"Got: {filename}"
        )
    return filename


class SurfaceImage:
    """Surface image containing meshes & data for both hemispheres.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    mesh : :obj:`nilearn.surface.PolyMesh`, \
           or :obj:`dict` of  \
           :obj:`nilearn.surface.SurfaceMesh`, \
           :obj:`str`, \
           :obj:`pathlib.Path`
           Meshes for the both hemispheres.

    data : :obj:`nilearn.surface.PolyData`, \
           or :obj:`dict` of  \
           :obj:`numpy.ndarray`, \
           :obj:`str`, \
           :obj:`pathlib.Path`
           Data for the both hemispheres.

    squeeze_on_save : :obj:`bool` or None, default=None
            If ``True`` axes of length one from the data
            will be removed before saving them to file.
            If ``None`` is passed,
            then the value will be set to ``True``
            if any of the data parts is one dimensional.

    Attributes
    ----------
    shape : (int, int)
        shape of the surface data array
    """

    def __init__(self, mesh, data):
        """Create a SurfaceImage instance."""
        self.mesh = mesh if isinstance(mesh, PolyMesh) else PolyMesh(**mesh)

        if not isinstance(data, (PolyData, dict)):
            raise TypeError(
                "'data' must be one of"
                "[PolyData, dict].\n"
                f"Got {type(data)}"
            )

        if isinstance(data, PolyData):
            self.data = data
        elif isinstance(data, dict):
            self.data = PolyData(**data)

        _check_data_and_mesh_compat(self.mesh, self.data)

    @property
    def shape(self):
        """Shape of the data."""
        return self.data.shape

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.shape}>"

    @classmethod
    def from_volume(
        cls, mesh, volume_img, inner_mesh=None, **vol_to_surf_kwargs
    ):
        """Create surface image from volume image.

        Parameters
        ----------
        mesh : :obj:`nilearn.surface.PolyMesh` \
             or :obj:`dict` of  \
             :obj:`nilearn.surface.SurfaceMesh`, \
             :obj:`str`, \
             :obj:`pathlib.Path`
             Surface mesh.

        volume_img : Niimg-like object
            3D or 4D volume image to project to the surface mesh.

        inner_mesh : :obj:`nilearn.surface.PolyMesh` \
             or :obj:`dict` of  \
             :obj:`nilearn.surface.SurfaceMesh`, \
             :obj:`str`, \
             :obj:`pathlib.Path`, default=None
            Inner mesh to pass to :func:`nilearn.surface.vol_to_surf`.

        vol_to_surf_kwargs : dict[str, Any]
            Dictionary of extra key-words arguments to pass
            to :func:`nilearn.surface.vol_to_surf`.

        Examples
        --------
        >>> from nilearn.surface import SurfaceImage
        >>> from nilearn.datasets import load_fsaverage
        >>> from nilearn.datasets import load_sample_motor_activation_image

        >>> fsavg = load_fsaverage()
        >>> vol_img = load_sample_motor_activation_image()
        >>> img = SurfaceImage.from_volume(fsavg["white_matter"], vol_img)
        >>> img
        <SurfaceImage (20484,)>
        >>> img = SurfaceImage.from_volume(
        ...     fsavg["white_matter"], vol_img, inner_mesh=fsavg["pial"]
        ... )
        >>> img
        <SurfaceImage (20484,)>
        """
        mesh = mesh if isinstance(mesh, PolyMesh) else PolyMesh(**mesh)
        if inner_mesh is not None:
            inner_mesh = (
                inner_mesh
                if isinstance(inner_mesh, PolyMesh)
                else PolyMesh(**inner_mesh)
            )
            left_kwargs = {"inner_mesh": inner_mesh.parts["left"]}
            right_kwargs = {"inner_mesh": inner_mesh.parts["right"]}
        else:
            left_kwargs, right_kwargs = {}, {}

        if isinstance(volume_img, (str, Path)):
            volume_img = check_niimg(volume_img)

        texture_left = vol_to_surf(
            volume_img, mesh.parts["left"], **vol_to_surf_kwargs, **left_kwargs
        )

        texture_right = vol_to_surf(
            volume_img,
            mesh.parts["right"],
            **vol_to_surf_kwargs,
            **right_kwargs,
        )

        data = PolyData(left=texture_left, right=texture_right)

        return cls(mesh=mesh, data=data)
