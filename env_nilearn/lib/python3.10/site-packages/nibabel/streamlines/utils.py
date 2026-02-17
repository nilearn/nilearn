import itertools

import nibabel


def get_affine_from_reference(ref):
    """Returns the affine defining the reference space.

    Parameters
    ----------
    ref : str or :class:`Nifti1Image` object or ndarray shape (4, 4)
        If str then it's the filename of reference file that will be loaded
        using :func:`nibabel.load` in order to obtain the affine.
        If :class:`Nifti1Image` object then the affine is obtained from it.
        If ndarray shape (4, 4) then it's the affine.

    Returns
    -------
    affine : ndarray (4, 4)
        Transformation matrix mapping voxel space to RAS+mm space.
    """
    if hasattr(ref, 'affine'):
        return ref.affine

    if hasattr(ref, 'shape'):
        if ref.shape != (4, 4):
            msg = '`ref` needs to be a numpy array with shape (4, 4)!'
            raise ValueError(msg)

        return ref

    # Assume `ref` is the name of a neuroimaging file.
    return nibabel.load(ref).affine


def peek_next(iterable):
    """Peek next element of iterable.

    Parameters
    ----------
    iterable
        Iterable to peek the next element from.

    Returns
    -------
    next_item
        Element peeked from `iterable`.
    new_iterable
        Iterable behaving like if the original `iterable` was untouched.
    """
    next_item = next(iterable)
    return next_item, itertools.chain([next_item], iterable)
