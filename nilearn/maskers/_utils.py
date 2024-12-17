import copy

import numpy as np

from nilearn import image
from nilearn.surface import SurfaceImage
from nilearn.surface.surface import at_least_2d


def _check_dims(imgs):
    # check dims of one image if given a list
    if isinstance(imgs, list):
        im = imgs[0]
        dim = image.load_img(im).shape
        # in case of 4D (timeseries) + 1D (subjects) return first subject
        return (im, (*dim, 1)) if len(dim) == 4 else (imgs, (*dim, 1))
    else:
        dim = image.load_img(imgs).shape
        return imgs, dim


def compute_middle_image(img):
    """Compute middle image of timeseries (4D data)."""
    img, dim = _check_dims(img)
    if len(dim) == 4 or len(dim) == 5:
        img = image.index_img(img, dim[-1] // 2)
    return img, len(dim)


def check_same_n_vertices(mesh_1, mesh_2):
    """Check that 2 PolyMesh have the same keys and that n vertices match.

    Parameters
    ----------
    mesh_1: PolyMesh

    mesh_2: PolyMesh
    """
    keys_1, keys_2 = set(mesh_1.parts.keys()), set(mesh_2.parts.keys())
    if keys_1 != keys_2:
        diff = keys_1.symmetric_difference(keys_2)
        raise ValueError(
            f"Meshes do not have the same keys. Offending keys: {diff}"
        )
    for key in keys_1:
        if mesh_1.parts[key].n_vertices != mesh_2.parts[key].n_vertices:
            raise ValueError(
                f"Number of vertices do not match for '{key}'."
                "number of vertices in mesh_1: "
                f"{mesh_1.parts[key].n_vertices}; "
                f"in mesh_2: {mesh_2.parts[key].n_vertices}"
            )


def compute_mean_surface_image(img):
    """Compute mean of SurfaceImage over time points (for 'time series').

    Parameters
    ----------
    img : SurfaceImage

    Returns
    -------
    SurfaceImage
    """
    if len(img.shape) < 2 or img.shape[1] < 2:
        return img

    data = {
        part: np.mean(value, axis=1).astype(float)
        for part, value in img.data.parts.items()
    }
    return SurfaceImage(mesh=img.mesh, data=data)


def get_min_max_surface_image(img):
    """Get min and max across hemisphere for a SurfaceImage.

    Parameters
    ----------
    img : SurfaceImage

    Returns
    -------
    vmin : float

    vmax : float
    """
    vmin = min(min(x.ravel()) for x in img.data.parts.values())
    vmax = max(max(x.ravel()) for x in img.data.parts.values())
    return vmin, vmax


def concatenate_surface_images(imgs):
    """Concatenate the data of a list or tuple of SurfaceImages.

    Assumes all images have same meshes.

    Parameters
    ----------
    imgs : :obj:`list` or :obj:`tuple` of SurfaceImage object

    Returns
    -------
    SurfaceImage object
    """
    if not isinstance(imgs, (tuple, list)) or any(
        not isinstance(x, SurfaceImage) for x in imgs
    ):
        raise TypeError(
            "'imgs' must be a list or a tuple of SurfaceImage instances."
        )

    if len(imgs) == 1:
        return imgs[0]

    for i, img in enumerate(imgs):
        check_same_n_vertices(img.mesh, imgs[0].mesh)
        imgs[i] = at_least_2d(img)

    output_data = {}
    for part in imgs[0].data.parts:
        tmp = [img.data.parts[part] for img in imgs]
        output_data[part] = np.concatenate(tmp, axis=1)

    output = SurfaceImage(mesh=imgs[0].mesh, data=output_data)

    return output


def check_surface_data_ndims(img, dim, var_name="img"):
    """Check if the data of a SurfaceImage is of a given dimension,
    raise error if not.

    Parameters
    ----------
    img : :obj:`~nilearn.surface.SurfaceImage`
        SurfaceImage to check.

    dim : int
        Dimensions the data should have.

    var_name : str, optional
        Name of the variable to include in the error message.

    Returns
    -------
    raise ValueError if the data of the SurfaceImage is not of the given
    dimension.
    """
    n_dim_left = img.data.parts["left"].ndim
    n_dim_right = img.data.parts["right"].ndim
    if not all(x == dim for x in [n_dim_left, n_dim_right]):
        raise ValueError(
            f"Data for each hemisphere of {var_name} should be {dim}D, "
            f"but found {n_dim_left}D for left and {n_dim_right}D for right."
        )


def concat_extract_surface_data_parts(img):
    """Concatenate the data of a SurfaceImage across hemispheres and return
    as a numpy array.

    Parameters
    ----------
    img : :obj:`~nilearn.surface.SurfaceImage` object
        SurfaceImage whose data to concatenate and extract.

    Returns
    -------
    :obj:`~numpy.ndarray`
        Concatenated data across hemispheres.
    """
    return np.concatenate(list(img.data.parts.values()), axis=0)


def deconcatenate_surface_images(img):
    """Deconcatenate a 3D Surface image into a a list of SurfaceImages.

    Parameters
    ----------
    img : SurfaceImage object

    Returns
    -------
    :obj:`list` or :obj:`tuple` of SurfaceImage object
    """
    if not isinstance(img, SurfaceImage):
        raise TypeError("Input must a be SurfaceImage.")

    if len(img.shape) < 2 or img.shape[1] < 2:
        return [img]

    mesh = img.mesh

    return [
        SurfaceImage(
            mesh=copy.deepcopy(mesh),
            data=_extract_surface_image_data(img, i),
        )
        for i in range(img.shape[1])
    ]


def _extract_surface_image_data(surface_image, index):
    mesh = surface_image.mesh
    data = surface_image.data

    return {
        hemi: data.parts[hemi][..., index]
        .copy()
        .reshape(mesh.parts[hemi].n_vertices, 1)
        for hemi in data.parts
    }
