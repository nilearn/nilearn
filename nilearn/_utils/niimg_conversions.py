"""
Conversion utilities.
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import numpy as np
import inspect
import itertools
from sklearn.externals.joblib import Memory

from .cache_mixin import cache
from .niimg import _safe_get_data, load_niimg, new_img_like
from .compat import _basestring


def _check_fov(img, affine, shape):
    """ Return True if img's field of view correspond to given
        shape and affine, False elsewhere.
    """
    img = check_niimg(img)
    return (img.shape[:3] == shape and
            np.allclose(img.get_affine(), affine))


def _check_same_fov(img1, img2):
    """ Return True if img1 and img2 have the same field of view
        (shape and affine), False elsewhere.
    """
    img1 = check_niimg(img1)
    img2 = check_niimg(img2)
    return (img1.shape[:3] == img2.shape[:3]
            and np.allclose(img1.get_affine(), img2.get_affine()))


def _index_img(img, index):
    """Helper function for check_niimg_4d."""
    return new_img_like(
        img, img.get_data()[:, :, :, index], img.get_affine(),
        copy_header=True)


def _iter_check_niimg(niimgs, ndim=None, atleast_4d=False,
                      target_fov=None,
                      memory=Memory(cachedir=None),
                      memory_level=0, verbose=0):
    ref_fov = None
    ndim_minus_one = ndim - 1 if ndim is not None else None
    if target_fov is not None and target_fov != "first":
        ref_fov = target_fov
    for i, niimg in enumerate(niimgs):
        try:
            niimg = check_niimg(
                niimg, ndim=ndim_minus_one, atleast_4d=atleast_4d)
            if i == 0:
                ndim_minus_one = len(niimg.shape)
                if ref_fov is None:
                    ref_fov = (niimg.get_affine(), niimg.shape[:3])

            if not _check_fov(niimg, ref_fov[0], ref_fov[1]):
                if target_fov is not None:
                    from nilearn import image  # we avoid a circular import
                    niimg = cache(
                        image.resample_img, memory, func_memory_level=2,
                        memory_level=memory_level)(
                            niimg, target_affine=ref_fov[0],
                            target_shape=ref_fov[1])
                else:
                    raise ValueError(
                        "Field of view of image #%d is different from "
                        "reference FOV.\n"
                        "Reference affine:\n%r\nImage affine:\n%r\n"
                        "Reference shape:\n%r\nImage shape:\n%r\n"
                        % (i, ref_fov[0], niimg.get_affine(), ref_fov[1],
                           niimg.shape))
            yield niimg
        except TypeError as exc:
            exc.args = (('Error encountered while loading image #%d' % i,)
                        + exc.args)
            raise


def check_niimg(niimg, ndim=None, atleast_4d=False, return_iterator=False):
    """Check that niimg is a proper 3D/4D niimg. Turn filenames into objects.

    Parameters
    ----------
    niimg: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. If it is an object, check if get_data()
        and get_affine() methods are present, raise TypeError otherwise.

    ndim: integer {3, 4}, optional
        Indicate the dimensionality of the expected niimg. An
        error is raised if the niimg is of another dimensionality.

    atleast_4d: boolean, optional
        Indicates if a 3d image should be turned into a single-scan 4d niimg.

    Returns
    -------
    result: 3D/4D Niimg-like object
        Result can be nibabel.Nifti1Image or the input, as-is. It is guaranteed
        that the returned object has get_data() and get_affine() methods.

    Notes
    -----
    In nilearn, special care has been taken to make image manipulation easy.
    This method is a kind of pre-requisite for any data processing method in
    nilearn because it checks if data have a correct format and loads them if
    necessary.

    Its application is idempotent.
    """
    # in case of an iterable
    if hasattr(niimg, "__iter__") and not isinstance(niimg, _basestring):
        if return_iterator:
            return _iter_check_niimg(niimg, ndim)
        return concat_niimgs(niimg)

    # Otherwise, it should be a filename or a SpatialImage, we load it
    niimg = load_niimg(niimg)

    if ndim == 3 and len(niimg.shape) == 4 and niimg.shape[3] == 1:
        # "squeeze" the image.
        data = _safe_get_data(niimg)
        affine = niimg.get_affine()
        niimg = new_img_like(niimg, data[:, :, :, 0], affine)
    if atleast_4d and len(niimg.shape) == 3:
        data = niimg.get_data().view()
        data.shape = data.shape + (1, )
        niimg = new_img_like(niimg, data, niimg.get_affine())

    if ndim is not None and len(niimg.shape) != ndim:
        raise TypeError(
            "Data must be a %iD Niimg-like object but you provided an "
            "image of shape %s. See "
            "http://nilearn.github.io/building_blocks/"
            "manipulating_mr_images.html#niimg." % (ndim, niimg.shape))

    if return_iterator:
        return (_index_img(niimg, i) for i in range(niimg.shape[3]))

    return niimg


def check_niimg_3d(niimg):
    """Check that niimg is a proper 3D niimg-like object and load it.
    Parameters
    ----------
    niimg: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. If it is an object, check if get_data()
        and get_affine() methods are present, raise TypeError otherwise.

    Returns
    -------
    result: 3D Niimg-like object
        Result can be nibabel.Nifti1Image or the input, as-is. It is guaranteed
        that the returned object has get_data() and get_affine() methods.

    Notes
    -----
    In nilearn, special care has been taken to make image manipulation easy.
    This method is a kind of pre-requisite for any data processing method in
    nilearn because it checks if data have a correct format and loads them if
    necessary.

    Its application is idempotent.
    """
    return check_niimg(niimg, ndim=3)


def check_niimg_4d(niimg, return_iterator=False):
    """Check that niimg is a proper 4D niimg-like object and load it.

    Parameters
    ----------
    niimg: 4D Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        If niimgs is an iterable, checks if data is really 4D. Then,
        considering that it is a list of niimg and load them one by one.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. If it is an object, check if get_data
        and get_affine methods are present, raise an Exception otherwise.

    return_iterator: boolean
        If True, an iterator of 3D images is returned. This reduces the memory
        usage when `niimgs` contains 3D images.
        If False, a single 4D image is returned. When `niimgs` contains 3D
        images they are concatenated together.

    Returns
    -------
    niimg: 4D nibabel.Nifti1Image or iterator of 3D nibabel.Nifti1Image

    Notes
    -----
    This function is the equivalent to check_niimg_3d() for Niimg-like objects
    with a session level.

    Its application is idempotent.
    """
    return check_niimg(niimg, ndim=4, return_iterator=return_iterator)


def concat_niimgs(niimgs, dtype=np.float32,
                  memory=Memory(cachedir=None), memory_level=0,
                  auto_resample=False, verbose=0):
    """Concatenate a list of 3D/4D niimgs of varying lengths.

    The niimgs list can contain niftis/paths to images of varying dimensions
    (i.e., 3D or 4D) as well as different 3D shapes and affines, as they
    will be matched to the first image in the list if auto_resample=True.

    Parameters
    ----------
    niimgs: iterable of Niimg-like objects
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        Niimgs to concatenate.

    dtype: numpy dtype, optional
        the dtype of the returned image

    auto_resample: boolean
        Converts all images to the space of the first one.

    verbose: int
        Controls the amount of verbosity (0 means no messages).

    memory : instance of joblib.Memory or string
        Used to cache the resampling process.
        By default, no caching is done. If a string is given, it is the
        path to the caching directory.

    memory_level : integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    Returns
    -------
    concatenated: nibabel.Nifti1Image
        A single image.
    """

    # Optimizations

    # Case 1a: all niimgs are already loaded
    # -------------------------------------

    # In that case, we can browse the niimgs and count the lengths of the
    # final sequence to preallocate a numpy array

    # Case 1b: all niimgs are filepaths or memory mapped
    # -------------------------------------------------

    # Same as above, we do a first pass. It is less costly because we will
    # only read file headers.
    # Note: this remark is theoritically but the actual implementation of
    # check_niimg forces data loading so it is still less costly in term of
    # memory but it may be slower than the strategy used in case 2

    # Case 2: niimgs is a generator
    # -----------------------------

    # This is the only case in which we can't browse it several times: we
    # accumulate data in a list and concatenate it in the end

    target_fov = 'first' if auto_resample else None
    first_niimg = None
    if not inspect.isgenerator(niimgs):
        iterator, literator = itertools.tee(iter(niimgs))
        try:
            first_niimg = check_niimg(next(literator))
        except StopIteration:
            raise TypeError('Cannot concatenate empty objects')

        ndim = len(first_niimg.shape)
        lengths = [first_niimg.shape[-1] if ndim == 4 else 1]
        for niimg in literator:
            niimg = check_niimg(niimg, ndim=ndim)
            lengths.append(niimg.shape[-1] if ndim == 4 else 1)

        target_shape = first_niimg.shape[:3]
        data = np.ndarray(target_shape + (sum(lengths), ),
                          order="F", dtype=dtype)
        cur_4d_index = 0
        for index, (size, niimg) in enumerate(zip(lengths, _iter_check_niimg(
                niimgs, atleast_4d=True, target_fov=target_fov,
                memory=memory, memory_level=memory_level))):

            if verbose > 0:
                if isinstance(niimg, _basestring):
                    nii_str = "image " + niimg
                else:
                    nii_str = "image #" + str(index)
                print("Concatenating {0}: {1}".format(index + 1, nii_str))

            data[..., cur_4d_index:cur_4d_index + size] = niimg.get_data()
            cur_4d_index += size
    else:
        data = []  # use a list for dynamic memory allocation
        for index, niimg in enumerate(_iter_check_niimg(
                niimgs, atleast_4d=True, target_fov=target_fov,
                memory=memory, memory_level=memory_level)):

            if index == 0:
                first_niimg = niimg
            if verbose > 0:
                if isinstance(niimg, _basestring):
                    nii_str = "image " + niimg
                else:
                    nii_str = "image #" + str(index)
                print("Concatenating {0}: {1}".format(index + 1, nii_str))

            data.append(niimg.get_data())
        data = np.concatenate(data, axis=-1)

    return new_img_like(first_niimg, data, first_niimg.get_affine())
