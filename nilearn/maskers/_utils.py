import numpy as np

from nilearn import image


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
