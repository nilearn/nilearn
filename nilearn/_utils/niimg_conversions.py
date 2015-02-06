"""
Conversion utilities.
"""
# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
# License: simplified BSD

import collections
import copy
import gc

import numpy as np

import nibabel

from sklearn.externals.joblib import Memory
from cache_mixin import cache

def is_img(obj):
    """ Check for get_data and get_affine method in an object

    Parameters
    ----------
    obj: any object
        Tested object

    Returns
    -------
    is_img: boolean
        True if get_data and get_affine methods are present and callable,
        False otherwise.
    """

    # We use a try/except here because this is the way hasattr works
    try:
        get_data = getattr(obj, "get_data")
        get_affine = getattr(obj, "get_affine")
        return callable(get_data) and callable(get_affine)
    except AttributeError:
        return False


def _get_shape(img):
    # Use the fact that Nifti1Image has a shape attribute that is
    # faster than loading the data from disk
    if hasattr(img, 'shape'):
        shape = img.shape
    else:
        shape = img.get_data().shape
    return shape


def _check_same_fov(img1, img2):
    """ Return True if img1 and img2 have the same field of view
        (shape and affine), False elsewhere.
    """
    img1 = check_niimgs(img1, accept_3d=True)
    img2 = check_niimgs(img2, accept_3d=True)
    return (_get_shape(img1)[:3] == _get_shape(img2)[:3]
            and np.allclose(img1.get_affine(), img2.get_affine()))


def _repr_niimgs(niimgs):
    """ Pretty printing of niimg or niimgs.
    """
    if isinstance(niimgs, basestring):
        return niimgs
    if isinstance(niimgs, collections.Iterable):
        return '[%s]' % ', '.join(_repr_niimgs(niimg) for niimg in niimgs)
    # Nibabel objects have a 'get_filename'
    try:
        filename = niimgs.get_filename()
        if filename is not None:
            return "%s('%s')" % (niimgs.__class__.__name__,
                                filename)
        else:
            return "%s(\nshape=%s,\naffine=%s\n)" % \
                   (niimgs.__class__.__name__,
                    repr(_get_shape(niimgs)),
                    repr(niimgs.get_affine()))
    except:
        pass
    return repr(niimgs)


def _safe_get_data(img):
    """ Get the data in the image without having a side effect on the
        Nifti1Image object
    """
    if hasattr(img, '_data_cache') and img._data_cache is None:
        # Copy locally the Nift1Image object to avoid the side effect of data
        # loading
        img = copy.deepcopy(img)
    # typically the line below can double memory usage
    # that's why we invoke a forced call to the garbage collector
    gc.collect()
    return img.get_data()


def copy_img(img):
    """Copy an image to a nibabel.Nifti1Image.

    Parameters
    ==========
    img: image
        nibabel.Nifti1Image object to copy.

    Returns
    =======
    img_copy: nibabel.Nifti1Image
        copy of input (data and affine)
    """
    if not is_img(img):
        raise ValueError("input value is not an image")
    return nibabel.Nifti1Image(img.get_data().copy(),
                               img.get_affine().copy())


def short_repr(niimg):
    this_repr = repr(niimg)
    if len(this_repr) > 20:
        # Shorten the repr to have a useful error message
        this_repr = this_repr[:18] + '...'
    return this_repr


def check_niimg(niimg, ensure_3d=False):
    """Check that niimg is a proper niimg. Turn filenames into objects.

    Parameters
    ----------
    niimg: Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. If it is an object, check if get_data()
        and get_affine() methods are present, raise TypeError otherwise.

    ensure_3d: boolean, optional
        If ensure_3d is true, the code checks that the image passed is a
        3D image and raises an error if not

    Returns
    -------
    result: Niimg-like object
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
    if hasattr(niimg, "__iter__"):
        if ensure_3d:
            raise TypeError("A 3D image is expected, but an iterable was"
                " given: %s" % short_repr(niimg))
        if hasattr(niimg, "__len__") and len(niimg) == 0:
            raise TypeError('An empty object - %r - was passed instead of an '
                            'image or a list of images' % niimg)
        return concat_niimgs(niimg)

    if isinstance(niimg, basestring):
        # data is a filename, we load it
        niimg = nibabel.load(niimg)
    elif not is_img(niimg):
        raise TypeError("Data given cannot be converted to a nifti"
                        " image: this object -'%s'- does not expose"
                        " get_data or get_affine methods"
                        % short_repr(niimg))
    if ensure_3d:
        shape = _get_shape(niimg)
        if len(shape) == 3:
            pass
        elif (len(shape) == 4 and shape[3] == 1):
            # "squeeze" the image.
            data = _safe_get_data(niimg)
            affine = niimg.get_affine()
            niimg = nibabel.Nifti1Image(data[:, :, :, 0], affine)
        else:
            raise TypeError("A 3D image is expected, but an image "
                "with a shape of %s was given." % (shape, ))
    return niimg


def _to_4d(data):
    """ Internal function to cast a 3D ndarray to a 4D one by adding a
        new axis at the end
    """
    if len(data.shape) == 4:
        return data
    out = data.view()
    out.shape = data.shape + (1, )
    return out


def concat_niimgs(niimgs, dtype=np.float32, accept_4d=False,
                  auto_resample=False, verbose=0,
                  memory=Memory(cachedir=None), memory_level=0):
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
        Accept 4D images

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

    # get properties from first image
    first_niimg = check_niimg(iter(niimgs).next())
    target_affine = first_niimg.get_affine()
    first_data = first_niimg.get_data()
    target_item_shape = first_niimg.shape[:3]  # skip 4th/time dimension

    # count how many images we have in all (might be list of different 4D's)
    lengths = []
    for index, niimg in enumerate(niimgs):
        this_shape = _get_shape(check_niimg(niimg))
        if len(this_shape) == 3:
            lengths.append(1)
        else:
            if not accept_4d:
                if (isinstance(niimg, basestring)):
                    i_error = "Image " + niimg
                else:
                    i_error = "Image #" + str(index)
                raise ValueError("%s is a 4D shape (shape: %s), but this "
                                 "function accepts only 3D images"
                                 % (i_error, this_shape))
            lengths.append(this_shape[3])

    # Using fortran order makes concatenation much faster than with C order,
    # because the voxels for a given image are grouped together in memory.
    data = np.ndarray(target_item_shape + (sum(lengths), ),
                      order="F", dtype=dtype)

    data[..., :lengths[0]] = _to_4d(first_data)
    cur_4d_index = 0
    for index, (iter_niimg, size) in enumerate(zip(niimgs, lengths)):
        # talk to user
        if (isinstance(iter_niimg, basestring)):
            nii_str = "image " + iter_niimg
        else:
            nii_str = "image #" + str(index)
        if verbose > 0:
            print "Concatenating {0}/{1}: {2}".format(index + 1, sum(lengths),
                                                      nii_str)

        if index == 0:  # we have already loaded the first one
            cur_4d_index += size
            continue

        niimg = check_niimg(iter_niimg)
        if (np.array_equal(niimg.get_affine(), target_affine) and
            target_item_shape == niimg.shape[:3]):
            this_data = niimg.get_data()
        else:
            if not auto_resample:
                raise ValueError("Affine of %s is different"
                                 " from reference affine"
                                 "\nReference affine:\n%r\n"
                                 "Wrong affine:\n%r"
                                 % (nii_str,
                                    target_affine,
                                    niimg.get_affine()))
            if verbose > 0:
                print "...resampled to first nifti!"
            
            from .. import image  # we avoid a circular import
            niimg = cache(image.resample_img, memory, func_memory_level=2,
                          memory_level=memory_level)(
                              niimg,
                              target_affine=target_affine,
                              target_shape=target_item_shape)
            this_data = niimg.get_data()

        data[..., cur_4d_index:cur_4d_index + size] = _to_4d(this_data)
        cur_4d_index += size

    return nibabel.Nifti1Image(data, target_affine)


def check_niimgs(niimgs, accept_3d=False, return_iterator=False):
    """ Check that an object is a list of niimgs and load it if necessary

    Parameters
    ----------
    niimgs: 4D Niimg-like object
        See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg.
        If niimgs is an iterable, checks if data is really 4D. Then,
        considering that it is a list of niimg and load them one by one.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it. If it is an object, check if get_data
        and get_affine methods are present, raise an Exception otherwise.

    accept_3d: boolean
       If True, consider a 3D image as a 4D one with last dimension equals
       to 1.

    return_iterator: boolean
        If False, a single 4D image is returned. When `niimgs` contains 3D
        images they are concatenated together.
        If True, an iterator of 3D images is returned. This reduces the memory
        usage when `niimgs` contains 3D images.

    Returns
    -------
    niimg: 4D nibabel.Nifti1Image or iterator of 3D nibabel.Nifti1Image

    Notes
    -----
    This function is the equivalent to check_niimg() for Niimg-like objects
    with a session level.

    Its application is idempotent.
    """
    # Initialization:
    # If given data is a list, we count the number of levels to check
    # dimensionality and make a consistent error message.
    depth = 0
    first_img = niimgs
    if accept_3d and (isinstance(first_img, basestring)
                      or not isinstance(first_img, collections.Iterable)):
        niimg = check_niimg(niimgs)
        if len(_get_shape(niimg)) == 3:
            niimg = nibabel.Nifti1Image(niimg.get_data()[..., np.newaxis],
                                        niimg.get_affine())
        return niimg

    # Use hasattr() instead of isinstance to workaround a Python 2.6/2.7 bug
    # See http://bugs.python.org/issue7624
    while hasattr(first_img, "__iter__") \
            and not isinstance(first_img, basestring):
        if hasattr(first_img, '__len__') and len(first_img) == 0:
            raise TypeError('An empty object - %r - was passed instead of an '
                            'image or a list of images' % niimgs)
        first_img = iter(first_img).next()
        depth += 1

    # First image is supposed to be a path or a Niimg-like object
    first_img = check_niimg(first_img)

    # Check dimension and depth
    shape = _get_shape(first_img)
    dim = len(shape)

    if (dim + depth) != 4:
        # Detailed error message that tells exactly the user what
        # was provided and what should have been provided.
        raise TypeError("Data must be a 4D Niimg-like object. You provided a "
                        "%s%dD image(s), of shape %s. "
                        "See http://nilearn.github.io/building_blocks/manipulating_mr_images.html#niimg." % (
                        'list of ' * depth, dim, shape))

    # Now, we load data as we know its format
    if dim == 4:
        if return_iterator:
            result = (_index_niimgs(niimgs, i)
                      for i in range(_get_shape(niimgs)[3]))
        else:
            result = check_niimg(niimgs)
    else:
        if return_iterator:
            result = (check_niimg(img) for img in niimgs)
        else:
            result = concat_niimgs(niimgs)
    return result


def _index_niimgs(niimgs, index):
    """Helper function for check_niimgs."""
    return nibabel.Nifti1Image(niimgs.get_data()[:, :, :, index],
                               niimgs.get_affine(),
                               header=niimgs.get_header())
