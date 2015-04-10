"""
Conversion utilities.
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import types
import collections

import numpy as np
from sklearn.externals.joblib import Memory

from .cache_mixin import cache
from .niimg import (_safe_get_data, load_niimg, new_img_like,
                    short_repr)
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


def _iter_check_niimg(niimgs, ndim=None, atleast_4d=False,
                      auto_resample=False, memory=Memory(cachedir=None),
                      memory_level=0, verbose=0):
    affine = None
    shape = None
    for i, niimg in enumerate(niimgs):
        try:
            niimg = check_niimg(niimg,
                                ndim=(ndim - 1 if ndim is not None else None),
                                atleast_4d=atleast_4d)
            if affine is None:
                affine = niimg.get_affine()
                shape = niimg.shape[:3]
                ndim = len(niimg.shape) + 1

            if (auto_resample == True) and (not np.array_equal(
                niimg.get_affine(), affine) or niimg.shape[:3] != shape):
                # if not auto_resample:
                #     raise ValueError("Affine of image is different"
                #                      " from reference affine"
                #                      "\nReference affine:\n%r\n"
                #                      "Wrong affine:\n%r"
                #                      % (affine,
                #                         niimg.get_affine()))
                if verbose > 0:
                    print("...resampled to first nifti!")

                from nilearn import image  # we avoid a circular import
                niimg = cache(image.resample_img, memory, func_memory_level=2,memory_level=memory_level)(niimg,target_affine=affine,target_shape=shape)

            if not _check_fov(niimg, affine, shape):
                raise ValueError(
                    "Field of view of image #%d is different from reference "
                    "FOV.\n"
                    "Reference affine:\n%r\nImage affine:\n%r\n"
                    "Reference shape:\n%r\nImage shape:\n%r\n"
                    % (i, affine, niimg.get_affine(), shape,
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

    ndim: boolean, optional
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
    # in case of an iterator:
    if hasattr(niimg, "__iter__") and not isinstance(niimg, _basestring):
        if return_iterator:
            return _iter_check_niimg(niimg, ndim)
        return concat_niimgs(niimg)

    # Otherwise, it should be a filename or a SpatialImage, we load it
    niimg = load_niimg(niimg)

    if (ndim == 3 and len(niimg.shape) == 4 and niimg.shape[3] == 1):
        # "squeeze" the image.
        data = _safe_get_data(niimg)
        affine = niimg.get_affine()
        niimg = new_img_like(niimg, data[:, :, :, 0], affine)
    if atleast_4d and len(niimg.shape) == 3:
        data = niimg.get_data().view()
        data.shape = data.shape + (1, )
        niimg = new_img_like(niimg, data, niimg.get_affine())

    if ndim is not None and len(niimg.shape) != ndim:
        raise(TypeError(
            "Data must be a %iD Niimg-like object but you provided an "
            "image of shape %s. See "
            "http://nilearn.github.io/building_blocks/"
            "manipulating_mr_images.html#niimg." % (ndim, niimg.shape)))

    if return_iterator:
        return (_index_niimgs(niimg, i) for i in range(niimg.shape[3]))

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


# class NiimgIter(collections.Iterable):
# 
#     def __init__(self, niimgs, dtype=None, auto_resample=False,
#                  verbose=0):
#         self.dtype = dtype
#         self.auto_resample = auto_resample
#         self.verbose = verbose
# 
#     def __iter__(self):
#         self.first_iter = True
#         self.iter_niimgs = iter(niimgs)
#         return self
# 
#     def __next__():
#         niimg = check_niimg(self.iter_niimgs.next())
# 
#         if self.first_iter:
#             self.first_iter = False
#             if auto_resample:
#                 self.affine_ = niimg.get_affine()
#                 self.shape_ = niimg.shape
# 
# 
# class ConcatIter(NiimgIter):
# 
#     def __init__(self, niimgs, dtype=np.float32, buffer=None, auto_resample=False,
#                  verbose=0):
#         super(ConcatIter, self).__init__(
#             niimgs, dtype=dtype, auto_resample=auto_resample, verbose=verbose)
#         self.buffer = buffer
# 
#     def __iter__(self):
#         super(ConcatIter, self).__iter__()
#         return self
# 
#     def __next__():
#         pass


def concat_niimgs(niimgs, dtype=np.float32, accept_4d=False,
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

    accept_4d: boolean, optional
        Also parse niimg-like objects in niimgs when one of them reflects
        more than one image (i.e., has a time dimension)

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


    data = []  # use a list for dynamic memory allocation
    cur_4d_index = 0
    first_niimg = None
    # for index, (iter_niimg, size) in enumerate(zip(niimgs, lengths)):
    for index, niimg in enumerate(_iter_check_niimg(
        niimgs, atleast_4d=True, auto_resample=auto_resample)):
        # get properties from first image
        if index == 0:
            first_niimg = niimg
            target_affine = niimg.get_affine()
            target_item_shape = niimg.shape[:3]  # skip 4th/time dimension

        # talk to user
        if (isinstance(niimg, _basestring)):
            nii_str = "image " + niimg
        else:
            nii_str = "image #" + str(index)
        if verbose > 0:
            print("Concatenating {0}: {1}".format(index + 1, nii_str))

        if niimg.shape[3] > 1 and not accept_4d:
            i_error = "Image #" + str(index)
            raise ValueError("%s is a 4D shape (shape: %s), but this "
                             "function accepts only 3D images"
                             % (i_error, niimg.shape))

        # if (np.array_equal(niimg.get_affine(), target_affine) and
        #         target_item_shape == niimg.shape[:3]):
        #     this_data = niimg.get_data()
        # else:
        #     if not auto_resample:
        #         raise ValueError("Affine of %s is different"
        #                          " from reference affine"
        #                          "\nReference affine:\n%r\n"
        #                          "Wrong affine:\n%r"
        #                          % (nii_str,
        #                             target_affine,
        #                             niimg.get_affine()))
        #     if verbose > 0:
        #         print("...resampled to first nifti!")
        # 
        #     from .. import image  # we avoid a circular import
        #     niimg = cache(image.resample_img, memory, func_memory_level=2,
        #                   memory_level=memory_level)(
        #                         niimg,
        #                         target_affine=target_affine,
        #                         target_shape=target_item_shape)
        #     this_data = niimg.get_data()

        for i_4d in np.arange(niimg.shape[3]):
            data.append(niimg.get_data()[..., i_4d])

    if first_niimg is None:  # iterator was empty
        raise TypeError('Cannot concatenate empty objects')

    data = np.asarray(data)  # convert list to numpy array
    if data.shape[0] == 1:  # remind that we enforced 4d images
        data = np.squeeze(data)
    else:
        data = np.rollaxis(data, 0, 4)
    return new_img_like(
        first_niimg,
        data,
        target_affine)



#def concat_niimgs(niimgs, dtype=np.float32, axis=-1,
#                  auto_resample=False, verbose=0,
#                  memory=Memory(cachedir=None), memory_level=0):
#    """Concatenate a list of 3D/4D niimgs of varying lengths.
#
#    The niimgs list can contain niftis/paths to images of varying dimensions
#    (i.e., 3D or 4D) as well as different 3D shapes and affines, as they
#    will be matched to the first image in the list if auto_resample=True.
#
#    Parameters
#    ----------
#    niimgs: iterable of Niimg-like objects
#        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
#        Niimgs to concatenate.
#
#    dtype: numpy dtype, optional
#        the dtype of the returned image
#
#    accept_4d: boolean, optional
#        Accept 4D images
#
#    auto_resample: boolean
#        Converts all images to the space of the first one.
#
#    verbose: int
#        Controls the amount of verbosity (0 means no messages).
#
#    memory : instance of joblib.Memory or string
#        Used to cache the resampling process.
#        By default, no caching is done. If a string is given, it is the
#        path to the caching directory.
#
#    memory_level : integer, optional
#        Rough estimator of the amount of memory used by caching. Higher value
#        means more memory for caching.
#
#    Returns
#    -------
#    concatenated: nibabel.Nifti1Image
#        A single image.
#    """
#
#    #try:
#    #    first_niimg = check_niimg(next(iter(niimgs)), atleast_4d=True)
#    #except StopIteration:
#    #    raise TypeError('Cannot concatenate empty objects')
#
#    buffer = None
#
#    if not isinstance(niimgs, types.GeneratorType):
#        # Compute the length of the niimgs along the axis to allocate a buffer
#        length = 0
#        for niimg in niimgs:
#            niimg = check_niimg(niimg)
#            if axis < len(niimg.shape):
#                length += niimg.shape[axis]
#            else:
#                # Axis is new
#                length += 1
#        buffer = np.empty()
#
#    for index, niimg in enumerate(niimgs):
#        this_shape = check_niimg(niimg).shape
#        if len(this_shape) == 3:
#            lengths.append(1)
#        else:
#            if not accept_4d:
#                if (isinstance(niimg, _basestring)):
#                    i_error = "Image " + niimg
#                else:
#                    i_error = "Image #" + str(index)
#                raise ValueError("%s is a 4D shape (shape: %s), but this "
#                                 "function accepts only 3D images"
#                                 % (i_error, this_shape))
#            lengths.append(this_shape[3])
#
#    # Using fortran order makes concatenation much faster than with C order,
#    # because the voxels for a given image are grouped together in memory.
#    data = np.ndarray(target_item_shape + (sum(lengths), ),
#                      order="F", dtype=dtype)
#
#    data[..., :lengths[0]] = first_data
#    cur_4d_index = 0
#    for index, (iter_niimg, size) in enumerate(zip(niimgs, lengths)):
#        # talk to user
#        if (isinstance(iter_niimg, _basestring)):
#            nii_str = "image " + iter_niimg
#        else:
#            nii_str = "image #" + str(index)
#        if verbose > 0:
#            print("Concatenating {0}/{1}: {2}".format(index + 1, sum(lengths),
#                                                   nii_str))
#
#        if index == 0:  # we have already loaded the first one
#            cur_4d_index += size
#            continue
#
#        niimg = check_niimg(iter_niimg, atleast_4d=True)
#        if (np.array_equal(niimg.get_affine(), target_affine) and
#                target_item_shape == niimg.shape[:3]):
#            this_data = niimg.get_data()
#        else:
#            if not auto_resample:
#                raise ValueError("Affine of %s is different"
#                                 " from reference affine"
#                                 "\nReference affine:\n%r\n"
#                                 "Wrong affine:\n%r"
#                                 % (nii_str,
#                                    target_affine,
#                                    niimg.get_affine()))
#            if verbose > 0:
#                print("...resampled to first nifti!")
#
#            from .. import image  # we avoid a circular import
#            niimg = cache(image.resample_img, memory, func_memory_level=2,
#                          memory_level=memory_level)(
#                                niimg,
#                                target_affine=target_affine,
#                                target_shape=target_item_shape)
#            this_data = niimg.get_data()
#
#        data[..., cur_4d_index:cur_4d_index + size] = this_data
#        cur_4d_index += size
#
#    return new_img_like(first_niimg, data, target_affine)


def _iter_check_niimg_4d(niimgs):
    affine = None
    shape = None
    for i, niimg in enumerate(niimgs):
        try:
            niimg = check_niimg_3d(niimg)
            if affine is None:
                affine = niimg.get_affine()
                shape = niimg.shape
            if not _check_fov(niimg, affine, shape):
                raise ValueError(
                    "Field of view of image #%d is different from reference "
                    "FOV.\n"
                    "Reference affine:\n%r\nImage affine:\n%r\n"
                    "Reference shape:\n%r\nImage shape:\n%r\n"
                    % (i, affine, niimg.get_affine(), shape,
                       niimg.shape))
            yield niimg
        except TypeError as exc:
            exc.args = (('Error encountered while loading image #%d' % i,) + 
                        exc.args)
            raise


def _index_niimgs(niimgs, index):
    """Helper function for check_niimg_4d."""
    return new_img_like(
        niimgs,
        niimgs.get_data()[:, :, :, index],
        niimgs.get_affine(),
        copy_header=True)
