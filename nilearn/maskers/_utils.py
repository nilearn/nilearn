import copy

import numpy as np

from nilearn import image
from nilearn.surface import SurfaceImage


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
