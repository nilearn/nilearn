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
