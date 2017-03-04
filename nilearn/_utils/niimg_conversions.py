"""
Conversion utilities.
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD
import warnings
import os.path
import glob

import nilearn as ni
import numpy as np
import itertools
from sklearn.externals.joblib import Memory

from .cache_mixin import cache
from .niimg import _safe_get_data, load_niimg
from .compat import _basestring, izip, get_affine

from .exceptions import DimensionError


def _check_fov(img, affine, shape):
    """ Return True if img's field of view correspond to given
        shape and affine, False elsewhere.
    """
    img = check_niimg(img)
    return (img.shape[:3] == shape and
            np.allclose(get_affine(img), affine))


def _check_same_fov(*args, **kwargs):
    """Returns True if provided images has the same field of view (shape and
       affine). Returns False or raise an error elsewhere, depending on the
       `raise_error` argument. This function can take an unlimited number of
       images as arguments or keyword arguments and raise a user-friendly
       ValueError if asked.

    Parameters
    ----------

    args: images
        Images to be checked. Images passed without keywords will be labelled
        as img_#1 in the error message (replace 1 with the appropriate index).

    kwargs: images
        Images to be checked. In case of error, images will be reference by
        their keyword name in the error message.

    raise_error: boolean, optional
        If True, an error will be raised in case of error.
    """
    raise_error = kwargs.pop('raise_error', False)
    for i, arg in enumerate(args):
        kwargs['img_#%i' % i] = arg
    errors = []
    for (a_name, a_img), (b_name, b_img) in itertools.combinations(
            kwargs.items(), 2):
        if not a_img.shape[:3] == b_img.shape[:3]:
            errors.append((a_name, b_name, 'shape'))
        if not np.allclose(get_affine(a_img), get_affine(b_img)):
            errors.append((a_name, b_name, 'affine'))
    if len(errors) > 0 and raise_error:
        raise ValueError('Following field of view errors were detected:\n' +
                         '\n'.join(['- %s and %s do not have the same %s' % e
                                    for e in errors]))
    return (len(errors) == 0)


def _index_img(img, index):
    from ..image import new_img_like  # avoid circular imports

    """Helper function for check_niimg_4d."""
    return new_img_like(
        img, img.get_data()[:, :, :, index], get_affine(img),
        copy_header=True)


def _resolve_globbing(path):
    if isinstance(path, _basestring):
        path_list = sorted(glob.glob(os.path.expanduser(path)))
        # Raise an error in case the niimgs list is empty.
        if len(path_list) == 0:
            raise ValueError("No files matching path: %s" % path)
        path = path_list

    return path


def _iter_check_niimg(niimgs, ensure_ndim=None, atleast_4d=False,
                      target_fov=None, dtype=None,
                      memory=Memory(cachedir=None),
                      memory_level=0, verbose=0):
    """Iterate over a list of niimgs and do sanity checks and resampling

    Parameters
    ----------

    niimgs: list of niimg or glob pattern
        Image to iterate over

    ensure_ndim: integer, optional
        If specified, an error is raised if the data does not have the
        required dimension.

    atleast_4d: boolean, optional
        If True, any 3D image is converted to a 4D single scan.

    target_fov: tuple of affine and shape
       If specified, images are resampled to this field of view

    dtype: {dtype, "auto"}
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    See also
    --------
        check_niimg, check_niimg_3d, check_niimg_4d
    """
    # If niimgs is a string, use glob to expand it to the matching filenames.
    niimgs = _resolve_globbing(niimgs)

    ref_fov = None
    resample_to_first_img = False
    ndim_minus_one = ensure_ndim - 1 if ensure_ndim is not None else None
    if target_fov is not None and target_fov != "first":
        ref_fov = target_fov
    i = -1
    for i, niimg in enumerate(niimgs):
        try:
            niimg = check_niimg(
                niimg, ensure_ndim=ndim_minus_one, atleast_4d=atleast_4d,
                dtype=dtype)
            if i == 0:
                ndim_minus_one = len(niimg.shape)
                if ref_fov is None:
                    ref_fov = (get_affine(niimg), niimg.shape[:3])
                    resample_to_first_img = True

            if not _check_fov(niimg, ref_fov[0], ref_fov[1]):
                if target_fov is not None:
                    from nilearn import image  # we avoid a circular import
                    if resample_to_first_img:
                        warnings.warn('Affine is different across subjects.'
                                      ' Realignement on first subject '
                                      'affine forced')
                    niimg = cache(image.resample_img, memory,
                                  func_memory_level=2,
                                  memory_level=memory_level)(
                            niimg, target_affine=ref_fov[0],
                            target_shape=ref_fov[1])
                else:
                    raise ValueError(
                        "Field of view of image #%d is different from "
                        "reference FOV.\n"
                        "Reference affine:\n%r\nImage affine:\n%r\n"
                        "Reference shape:\n%r\nImage shape:\n%r\n"
                        % (i, ref_fov[0], get_affine(niimg), ref_fov[1],
                           niimg.shape))
            yield niimg
        except DimensionError as exc:
            # Keep track of the additional dimension in the error
            exc.increment_stack_counter()
            raise
        except TypeError as exc:
            img_name = ''
            if isinstance(niimg, _basestring):
                img_name = " (%s) " % niimg

            exc.args = (('Error encountered while loading image #%d%s'
                         % (i, img_name),) + exc.args)
            raise

    # Raising an error if input generator is empty.
    if i == -1:
        raise ValueError("Input niimgs list is empty.")


def check_niimg(niimg, ensure_ndim=None, atleast_4d=False, dtype=None,
                return_iterator=False,
                wildcards=True):
    """Check that niimg is a proper 3D/4D niimg. Turn filenames into objects.

    Parameters
    ----------
    niimg: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. The '~' symbol is expanded to the user home
        folder.
        If it is an object, check if the get_data() method
        and affine attribute are present, raise TypeError otherwise.

    ensure_ndim: integer {3, 4}, optional
        Indicate the dimensionality of the expected niimg. An
        error is raised if the niimg is of another dimensionality.

    atleast_4d: boolean, optional
        Indicates if a 3d image should be turned into a single-scan 4d niimg.

    dtype: {dtype, "auto"}
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    return_iterator: boolean, optional
        Returns an iterator on the content of the niimg file input

    wildcards: boolean, optional
        Use niimg as a regular expression to get a list of matching input
        filenames.
        If multiple files match, the returned list is sorted using an ascending
        order.
        If no file matches the regular expression, a ValueError exception is
        raised.

    Returns
    -------
    result: 3D/4D Niimg-like object
        Result can be nibabel.Nifti1Image or the input, as-is. It is guaranteed
        that the returned object has get_data() method and affine attribute.

    Notes
    -----
    In nilearn, special care has been taken to make image manipulation easy.
    This method is a kind of pre-requisite for any data processing method in
    nilearn because it checks if data have a correct format and loads them if
    necessary.

    Its application is idempotent.

    See also
    --------
        _iter_check_niimg, check_niimg_3d, check_niimg_4d
    """
    from ..image import new_img_like  # avoid circular imports

    if isinstance(niimg, _basestring):
        if wildcards and ni.EXPAND_PATH_WILDCARDS:
            # Ascending sorting + expand user path
            filenames = sorted(glob.glob(os.path.expanduser(niimg)))

            # processing filenames matching globbing expression
            if len(filenames) >= 1 and glob.has_magic(niimg):
                niimg = filenames  # iterable case
            # niimg is an existing filename
            elif [niimg] == filenames:
                niimg = filenames[0]
            # No files found by glob
            elif glob.has_magic(niimg):
                # No files matching the glob expression, warn the user
                message = ("No files matching the entered niimg expression: "
                            "'%s'.\n You may have left wildcards usage "
                           "activated: please set the global constant "
                           "'nilearn.EXPAND_PATH_WILDCARDS' to False to "
                           "deactivate this behavior.") % niimg
                raise ValueError(message)
            else:
                raise ValueError("File not found: '%s'" % niimg)
        elif not os.path.exists(niimg):
            raise ValueError("File not found: '%s'" % niimg)

    # in case of an iterable
    if hasattr(niimg, "__iter__") and not isinstance(niimg, _basestring):
        if return_iterator:
            return _iter_check_niimg(niimg, ensure_ndim=ensure_ndim,
                                     dtype=dtype)
        return concat_niimgs(niimg, ensure_ndim=ensure_ndim, dtype=dtype)

    # Otherwise, it should be a filename or a SpatialImage, we load it
    niimg = load_niimg(niimg, dtype=dtype)

    if ensure_ndim == 3 and len(niimg.shape) == 4 and niimg.shape[3] == 1:
        # "squeeze" the image.
        data = _safe_get_data(niimg)
        affine = get_affine(niimg)
        niimg = new_img_like(niimg, data[:, :, :, 0], affine)
    if atleast_4d and len(niimg.shape) == 3:
        data = niimg.get_data().view()
        data.shape = data.shape + (1, )
        niimg = new_img_like(niimg, data, get_affine(niimg))

    if ensure_ndim is not None and len(niimg.shape) != ensure_ndim:
        raise DimensionError(len(niimg.shape), ensure_ndim)

    if return_iterator:
        return (_index_img(niimg, i) for i in range(niimg.shape[3]))

    return niimg


def check_niimg_3d(niimg, dtype=None):
    """Check that niimg is a proper 3D niimg-like object and load it.
    Parameters
    ----------
    niimg: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. If it is an object, check if the get_data()
        method and affine attribute are present, raise TypeError otherwise.

    dtype: {dtype, "auto"}
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    Returns
    -------
    result: 3D Niimg-like object
        Result can be nibabel.Nifti1Image or the input, as-is. It is guaranteed
        that the returned object has get_data() method and affine attribute.

    Notes
    -----
    In nilearn, special care has been taken to make image manipulation easy.
    This method is a kind of pre-requisite for any data processing method in
    nilearn because it checks if data have a correct format and loads them if
    necessary.

    Its application is idempotent.
    """
    return check_niimg(niimg, ensure_ndim=3, dtype=dtype)


def check_niimg_4d(niimg, return_iterator=False, dtype=None):
    """Check that niimg is a proper 4D niimg-like object and load it.

    Parameters
    ----------
    niimg: 4D Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        If niimgs is an iterable, checks if data is really 4D. Then,
        considering that it is a list of niimg and load them one by one.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. If it is an object, check if the get_data()
        method and affine attribute are present, raise an Exception otherwise.

    dtype: {dtype, "auto"}
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

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
    return check_niimg(niimg, ensure_ndim=4, return_iterator=return_iterator,
                       dtype=dtype)


def concat_niimgs(niimgs, dtype=np.float32, ensure_ndim=None,
                  memory=Memory(cachedir=None), memory_level=0,
                  auto_resample=False, verbose=0):
    """Concatenate a list of 3D/4D niimgs of varying lengths.

    The niimgs list can contain niftis/paths to images of varying dimensions
    (i.e., 3D or 4D) as well as different 3D shapes and affines, as they
    will be matched to the first image in the list if auto_resample=True.

    Parameters
    ----------
    niimgs: iterable of Niimg-like objects or glob pattern
        See http://nilearn.github.io/manipulating_images/input_output.html
        Niimgs to concatenate.

    dtype: numpy dtype, optional
        the dtype of the returned image

    ensure_ndim: integer, optional
        Indicate the dimensionality of the expected niimg. An
        error is raised if the niimg is of another dimensionality.

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

    See Also
    --------
    nilearn.image.index_img

    """
    from ..image import new_img_like  # avoid circular imports

    target_fov = 'first' if auto_resample else None

    # We remove one to the dimensionality because of the list is one dimension.
    ndim = None
    if ensure_ndim is not None:
        ndim = ensure_ndim - 1

    # If niimgs is a string, use glob to expand it to the matching filenames.
    niimgs = _resolve_globbing(niimgs)

    # First niimg is extracted to get information and for new_img_like
    first_niimg = None

    iterator, literator = itertools.tee(iter(niimgs))
    try:
        first_niimg = check_niimg(next(literator), ensure_ndim=ndim)
    except StopIteration:
        raise TypeError('Cannot concatenate empty objects')
    except DimensionError as exc:
        # Keep track of the additional dimension in the error
        exc.increment_stack_counter()
        raise

    # If no particular dimensionality is asked, we force consistency wrt the
    # first image
    if ndim is None:
        ndim = len(first_niimg.shape)

    if ndim not in [3, 4]:
        raise TypeError('Concatenated images must be 3D or 4D. You gave a '
                        'list of %dD images' % ndim)

    lengths = [first_niimg.shape[-1] if ndim == 4 else 1]
    for niimg in literator:
        # We check the dimensionality of the niimg
        try:
            niimg = check_niimg(niimg, ensure_ndim=ndim)
        except DimensionError as exc:
            # Keep track of the additional dimension in the error
            exc.increment_stack_counter()
            raise
        lengths.append(niimg.shape[-1] if ndim == 4 else 1)

    target_shape = first_niimg.shape[:3]
    if dtype == None:
        dtype = first_niimg.get_data().dtype
    data = np.ndarray(target_shape + (sum(lengths), ),
                      order="F", dtype=dtype)
    cur_4d_index = 0
    for index, (size, niimg) in enumerate(izip(lengths, _iter_check_niimg(
            iterator, atleast_4d=True, target_fov=target_fov,
            memory=memory, memory_level=memory_level))):

        if verbose > 0:
            if isinstance(niimg, _basestring):
                nii_str = "image " + niimg
            else:
                nii_str = "image #" + str(index)
            print("Concatenating {0}: {1}".format(index + 1, nii_str))

        data[..., cur_4d_index:cur_4d_index + size] = niimg.get_data()
        cur_4d_index += size

    return new_img_like(first_niimg, data, get_affine(first_niimg), copy_header=True)
