"""Conversion utilities."""

import glob
import itertools

# Author: Gael Varoquaux, Alexandre Abraham, Philippe Gervais
import warnings
from pathlib import Path

import numpy as np
from joblib import Memory

import nilearn as ni

from .cache_mixin import cache
from .exceptions import DimensionError
from .helpers import stringify_path
from .niimg import _get_data, load_niimg, safe_get_data
from .path_finding import resolve_globbing


def _check_fov(img, affine, shape):
    """Return True if img's field of view correspond to given \
    shape and affine, False elsewhere.
    """
    img = check_niimg(img)
    return img.shape[:3] == shape and np.allclose(img.affine, affine)


def check_same_fov(*args, **kwargs):
    """Return True if provided images have the same field of view (shape and \
    affine) and return False or raise an error elsewhere, depending on the \
    `raise_error` argument.

    This function can take an unlimited number of
    images as arguments or keyword arguments and raise a user-friendly
    ValueError if asked.

    Parameters
    ----------
    args : images
        Images to be checked. Images passed without keywords will be labeled
        as img_#1 in the error message (replace 1 with the appropriate index).

    kwargs : images
        Images to be checked. In case of error, images will be reference by
        their keyword name in the error message.

    raise_error : boolean, optional
        If True, an error will be raised in case of error.

    """
    raise_error = kwargs.pop("raise_error", False)
    for i, arg in enumerate(args):
        kwargs[f"img_#{i}"] = arg
    errors = []
    for (a_name, a_img), (b_name, b_img) in itertools.combinations(
        kwargs.items(), 2
    ):
        if a_img.shape[:3] != b_img.shape[:3]:
            errors.append((a_name, b_name, "shape"))
        if not np.allclose(a_img.affine, b_img.affine):
            errors.append((a_name, b_name, "affine"))
    if errors and raise_error:
        raise ValueError(
            "Following field of view errors were detected:\n"
            + "\n".join(
                [
                    f"- {e[0]} and {e[1]} do not have the same {e[2]}"
                    for e in errors
                ]
            )
        )
    return not errors


def _index_img(img, index):
    from ..image import new_img_like  # avoid circular imports

    """Helper function for check_niimg_4d."""
    return new_img_like(
        img, _get_data(img)[:, :, :, index], img.affine, copy_header=True
    )


def iter_check_niimg(
    niimgs,
    ensure_ndim=None,
    atleast_4d=False,
    target_fov=None,
    dtype=None,
    memory=None,
    memory_level=0,
):
    """Iterate over a list of niimgs and do sanity checks and resampling.

    Parameters
    ----------
    niimgs : list of niimg or glob pattern
        Image to iterate over.

    ensure_ndim : integer, optional
        If specified, an error is raised if the data does not have the
        required dimension.

    atleast_4d : boolean, default=False
        If True, any 3D image is converted to a 4D single scan.

    target_fov : tuple of affine and shape, optional
       If specified, images are resampled to this field of view.

    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    memory : instance of joblib.Memory or string, default=None
        Used to cache the masking process.
        By default, no caching is done.
        If a string is given, it is the path to the caching directory.
        If ``None`` is passed will default to ``Memory(location=None)``.

    memory_level : integer, default=0
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    See Also
    --------
        check_niimg, check_niimg_3d, check_niimg_4d

    """
    if memory is None:
        memory = Memory(location=None)
    # If niimgs is a string, use glob to expand it to the matching filenames.
    niimgs = resolve_globbing(niimgs)

    ref_fov = None
    resample_to_first_img = False
    ndim_minus_one = ensure_ndim - 1 if ensure_ndim is not None else None
    if target_fov is not None and target_fov != "first":
        ref_fov = target_fov
    i = -1
    for i, niimg in enumerate(niimgs):
        try:
            niimg = check_niimg(
                niimg,
                ensure_ndim=ndim_minus_one,
                atleast_4d=atleast_4d,
                dtype=dtype,
            )
            if i == 0:
                ndim_minus_one = len(niimg.shape)
                if ref_fov is None:
                    ref_fov = (niimg.affine, niimg.shape[:3])
                    resample_to_first_img = True

            if not _check_fov(niimg, ref_fov[0], ref_fov[1]):
                if target_fov is None:
                    raise ValueError(
                        f"Field of view of image #{i} is different from "
                        "reference FOV.\n"
                        f"Reference affine:\n{ref_fov[0]!r}\n"
                        f"Image affine:\n{niimg.affine!r}\n"
                        f"Reference shape:\n{ref_fov[1]!r}\n"
                        f"Image shape:\n{niimg.shape!r}\n"
                    )
                from nilearn import image  # we avoid a circular import

                if resample_to_first_img:
                    warnings.warn(
                        "Affine is different across subjects."
                        " Realignement on first subject "
                        "affine forced"
                    )
                niimg = cache(
                    image.resample_img,
                    memory,
                    func_memory_level=2,
                    memory_level=memory_level,
                )(
                    niimg,
                    target_affine=ref_fov[0],
                    target_shape=ref_fov[1],
                    copy_header=True,
                    force_resample=False,  # TODO update to True in 0.13.0
                )
            yield niimg
        except DimensionError as exc:
            # Keep track of the additional dimension in the error
            exc.increment_stack_counter()
            raise
        except TypeError as exc:
            img_name = f" ({niimg}) " if isinstance(niimg, (str, Path)) else ""

            exc.args = (
                f"Error encountered while loading image #{i}{img_name}",
                *exc.args,
            )
            raise

    # Raising an error if input generator is empty.
    if i == -1:
        raise ValueError("Input niimgs list is empty.")


def check_niimg(
    niimg,
    ensure_ndim=None,
    atleast_4d=False,
    dtype=None,
    return_iterator=False,
    wildcards=True,
):
    """Check that niimg is a proper 3D/4D niimg.

    Turn filenames into objects.

    Parameters
    ----------
    niimg : Niimg-like object
        See :ref:`extracting_data`.
        If niimg is a string or pathlib.Path, consider it as a path to
        Nifti image and call nibabel.load on it. The '~' symbol is expanded to
        the user home folder.
        If it is an object, check if the affine attribute present and that
        nilearn.image.get_data returns a result, raise TypeError otherwise.

    ensure_ndim : integer {3, 4}, optional
        Indicate the dimensionality of the expected niimg. An
        error is raised if the niimg is of another dimensionality.

    atleast_4d : boolean, default=False
        Indicates if a 3d image should be turned into a single-scan 4d niimg.

    dtype : {None, dtype, "auto"}, default=None
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous. If None, data will not be converted to a new data type.

    return_iterator : boolean, default=False
        Returns an iterator on the content of the niimg file input.

    wildcards : boolean, default=True
        Use niimg as a regular expression to get a list of matching input
        filenames.
        If multiple files match, the returned list is sorted using an ascending
        order.
        If no file matches the regular expression, a ValueError exception is
        raised.

    Returns
    -------
    result : 3D/4D Niimg-like object
        Result can be nibabel.Nifti1Image or the input, as-is. It is guaranteed
        that the returned object has an affine attribute and that its data can
        be retrieved with nilearn.image.get_data.

    Notes
    -----
    In nilearn, special care has been taken to make image manipulation easy.
    This method is a kind of pre-requisite for any data processing method in
    nilearn because it checks if data have a correct format and loads them if
    necessary.

    Its application is idempotent.

    See Also
    --------
        iter_check_niimg, check_niimg_3d, check_niimg_4d

    """
    from ..image import new_img_like  # avoid circular imports

    niimg = stringify_path(niimg)

    if isinstance(niimg, str):
        if wildcards and ni.EXPAND_PATH_WILDCARDS:
            # Expand user path
            expanded_niimg = str(Path(niimg).expanduser())
            # Ascending sorting
            filenames = sorted(glob.glob(expanded_niimg))

            # processing filenames matching globbing expression
            if len(filenames) >= 1 and glob.has_magic(niimg):
                niimg = filenames  # iterable case
            # niimg is an existing filename
            elif [expanded_niimg] == filenames:
                niimg = filenames[0]
            # No files found by glob
            elif glob.has_magic(niimg):
                # No files matching the glob expression, warn the user
                message = (
                    "No files matching the entered niimg expression: "
                    f"'{niimg}'.\n"
                    "You may have left wildcards usage activated: "
                    "please set the global constant "
                    "'nilearn.EXPAND_PATH_WILDCARDS' to False "
                    "to deactivate this behavior."
                )
                raise ValueError(message)
            else:
                raise ValueError(f"File not found: '{niimg}'")
        elif not Path(niimg).exists():
            raise ValueError(f"File not found: '{niimg}'")

    # in case of an iterable
    if hasattr(niimg, "__iter__") and not isinstance(niimg, str):
        if return_iterator:
            return iter_check_niimg(
                niimg, ensure_ndim=ensure_ndim, dtype=dtype
            )
        return ni.image.concat_imgs(
            niimg, ensure_ndim=ensure_ndim, dtype=dtype
        )

    # Otherwise, it should be a filename or a SpatialImage, we load it
    niimg = load_niimg(niimg, dtype=dtype)

    if ensure_ndim == 3 and len(niimg.shape) == 4 and niimg.shape[3] == 1:
        # "squeeze" the image.
        data = safe_get_data(niimg)
        affine = niimg.affine
        niimg = new_img_like(niimg, data[:, :, :, 0], affine)
    if atleast_4d and len(niimg.shape) == 3:
        data = _get_data(niimg).view()
        data.shape = (*data.shape, 1)
        niimg = new_img_like(niimg, data, niimg.affine)

    if ensure_ndim is not None and len(niimg.shape) != ensure_ndim:
        raise DimensionError(len(niimg.shape), ensure_ndim)

    if return_iterator:
        return (_index_img(niimg, i) for i in range(niimg.shape[3]))

    return niimg


def check_niimg_3d(niimg, dtype=None):
    """Check that niimg is a proper 3D niimg-like object and load it.

    Parameters
    ----------
    niimg : Niimg-like object
        See :ref:`extracting_data`.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it.
        If it is an object, check if the affine attribute present and that
        nilearn.image.get_data returns a result, raise TypeError otherwise.

    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    Returns
    -------
    result : 3D Niimg-like object
        Result can be nibabel.Nifti1Image or the input, as-is. It is guaranteed
        that the returned object has an affine attribute and that its data can
        be retrieved with nilearn.image.get_data.

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
    niimg : 4D Niimg-like object
        See :ref:`extracting_data`.
        If niimgs is an iterable, checks if data is really 4D. Then,
        considering that it is a list of niimg and load them one by one.
        If niimg is a string, consider it as a path to Nifti image and
        call nibabel.load on it.
        If it is an object, check if the affine attribute present and that
        nilearn.image.get_data returns a result, raise TypeError otherwise.

    dtype : {dtype, "auto"}, optional
        Data type toward which the data should be converted. If "auto", the
        data will be converted to int32 if dtype is discrete and float32 if it
        is continuous.

    return_iterator : boolean, default=False
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
    with a run level.

    Its application is idempotent.

    """
    return check_niimg(
        niimg, ensure_ndim=4, return_iterator=return_iterator, dtype=dtype
    )
