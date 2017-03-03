"""
Functions for extracting region-defined signals.

Two ways of defining regions are supported: as labels in a single 3D image,
or as weights in one image per region (maps).
"""
# Author: Philippe Gervais
# License: simplified BSD

import numpy as np
from scipy import linalg, ndimage

from .. import _utils
from .._utils.niimg import _safe_get_data
from .. import masking
from ..image import new_img_like


# FIXME: naming scheme is not really satisfying. Any better idea appreciated.
def img_to_signals_labels(imgs, labels_img, mask_img=None,
                          background_label=0, order="F"):
    """Extract region signals from image.

    This function is applicable to regions defined by labels.

    labels, imgs and mask shapes and affines must fit. This function
    performs no resampling.

    Parameters
    ----------
    imgs: 4D Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        input images.

    labels_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        regions definition as labels. By default, the label zero is used to
        denote an absence of region. Use background_label to change it.

    mask_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Mask to apply to labels before extracting signals. Every point
        outside the mask is considered as background (i.e. no region).

    background_label: number
        number representing background in labels_img.

    order: str
        ordering of output array ("C" or "F"). Defaults to "F".

    Returns
    -------
    signals: numpy.ndarray
        Signals extracted from each region. One output signal is the mean
        of all input signals in a given region. If some regions are entirely
        outside the mask, the corresponding signal is zero.
        Shape is: (scan number, number of regions)

    labels: list or tuple
        corresponding labels for each signal. signal[:, n] was extracted from
        the region with label labels[n].

    See also
    --------
    nilearn.regions.signals_to_img_labels
    nilearn.regions.img_to_signals_maps
    """

    labels_img = _utils.check_niimg_3d(labels_img)

    # TODO: Make a special case for list of strings (load one image at a
    # time).
    imgs = _utils.check_niimg_4d(imgs)
    target_affine = _utils.compat.get_affine(imgs)
    target_shape = imgs.shape[:3]

    # Check shapes and affines.
    if labels_img.shape != target_shape:
        raise ValueError("labels_img and imgs shapes must be identical.")
    if abs(_utils.compat.get_affine(labels_img) - target_affine).max() > 1e-9:
        raise ValueError("labels_img and imgs affines must be identical")

    if mask_img is not None:
        mask_img = _utils.check_niimg_3d(mask_img)
        if mask_img.shape != target_shape:
            raise ValueError("mask_img and imgs shapes must be identical.")
        if abs(_utils.compat.get_affine(mask_img) - target_affine).max() > 1e-9:
            raise ValueError("mask_img and imgs affines must be identical")

    # Perform computation
    labels_data = _safe_get_data(labels_img, ensure_finite=True)
    labels = list(np.unique(labels_data))
    if background_label in labels:
        labels.remove(background_label)

    if mask_img is not None:
        mask_data = _safe_get_data(mask_img, ensure_finite=True)
        labels_data = labels_data.copy()
        labels_data[np.logical_not(mask_data)] = background_label

    data = _safe_get_data(imgs)
    signals = np.ndarray((data.shape[-1], len(labels)), order=order)
    for n, img in enumerate(np.rollaxis(data, -1)):
        signals[n] = np.asarray(ndimage.measurements.mean(img,
                                                          labels=labels_data,
                                                          index=labels))
    # Set to zero signals for missing labels. Workaround for Scipy behaviour
    missing_labels = set(labels) - set(np.unique(labels_data))
    labels_index = dict([(l, n) for n, l in enumerate(labels)])
    for l in missing_labels:
        signals[:, labels_index[l]] = 0
    return signals, labels


def signals_to_img_labels(signals, labels_img, mask_img=None,
                          background_label=0, order="F"):
    """Create image from region signals defined as labels.

    The same region signal is used for each voxel of the corresponding 3D
    volume.

    labels_img, mask_img must have the same shapes and affines.

    Parameters
    ----------
    signals: numpy.ndarray
        2D array with shape: (scan number, number of regions in labels_img)

    labels_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Region definitions using labels.

    mask_img: Niimg-like object, optional
        Boolean array giving voxels to process. integer arrays also accepted,
        In this array, zero means False, non-zero means True.

    background_label: number
        label to use for "no region".

    order: str
        ordering of output array ("C" or "F"). Defaults to "F".

    Returns
    -------
    img: nibabel.Nifti1Image
        Reconstructed image. dtype is that of "signals", affine and shape are
        those of labels_img.

    See also
    --------
    nilearn.regions.img_to_signals_labels
    nilearn.regions.signals_to_img_maps
    """

    labels_img = _utils.check_niimg_3d(labels_img)

    signals = np.asarray(signals)
    target_affine = _utils.compat.get_affine(labels_img)
    target_shape = labels_img.shape[:3]

    if mask_img is not None:
        mask_img = _utils.check_niimg_3d(mask_img)
        if mask_img.shape != target_shape:
            raise ValueError("mask_img and labels_img shapes "
                             "must be identical.")
        if abs(_utils.compat.get_affine(mask_img) - target_affine).max() > 1e-9:
            raise ValueError("mask_img and labels_img affines "
                             "must be identical")

    labels_data = _safe_get_data(labels_img, ensure_finite=True)
    labels = list(np.unique(labels_data))
    if background_label in labels:
        labels.remove(background_label)

    if mask_img is not None:
        mask_data = _safe_get_data(mask_img, ensure_finite=True)
        labels_data = labels_data.copy()
        labels_data[np.logical_not(mask_data)] = background_label

    # nditer is not available in numpy 1.3: using multiple loops.
    # Using these loops still gives a much faster code (6x) than this one:
    ## for n, label in enumerate(labels):
    ##     data[labels_data == label, :] = signals[:, n]
    data = np.zeros(target_shape + (signals.shape[0],),
                    dtype=signals.dtype, order=order)
    labels_dict = dict([(label, n) for n, label in enumerate(labels)])
    # optimized for "data" in F order.
    for k in range(labels_data.shape[2]):
        for j in range(labels_data.shape[1]):
            for i in range(labels_data.shape[0]):
                label = labels_data[i, j, k]
                num = labels_dict.get(label, None)
                if num is not None:
                    data[i, j, k, :] = signals[:, num]

    return new_img_like(labels_img, data, target_affine)


def img_to_signals_maps(imgs, maps_img, mask_img=None):
    """Extract region signals from image.

    This function is applicable to regions defined by maps.

    Parameters
    ----------
    imgs: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Input images.

    maps_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        regions definition as maps (array of weights).
        shape: imgs.shape + (region number, )

    mask_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        mask to apply to regions before extracting signals. Every point
        outside the mask is considered as background (i.e. outside of any
        region).

    order: str
        ordering of output array ("C" or "F"). Defaults to "F".

    Returns
    -------
    region_signals: numpy.ndarray
        Signals extracted from each region.
        Shape is: (scans number, number of regions intersecting mask)

    labels: list
        maps_img[..., labels[n]] is the region that has been used to extract
        signal region_signals[:, n].

    See also
    --------
    nilearn.regions.img_to_signals_labels
    nilearn.regions.signals_to_img_maps
    """

    maps_img = _utils.check_niimg_4d(maps_img)
    imgs = _utils.check_niimg_4d(imgs)
    affine = _utils.compat.get_affine(imgs)
    shape = imgs.shape[:3]

    # Check shapes and affines.
    if maps_img.shape[:3] != shape:
        raise ValueError("maps_img and imgs shapes must be identical.")
    if abs(_utils.compat.get_affine(maps_img) - affine).max() > 1e-9:
        raise ValueError("maps_img and imgs affines must be identical")

    maps_data = _safe_get_data(maps_img, ensure_finite=True)

    if mask_img is not None:
        mask_img = _utils.check_niimg_3d(mask_img)
        if mask_img.shape != shape:
            raise ValueError("mask_img and imgs shapes must be identical.")
        if abs(_utils.compat.get_affine(mask_img) - affine).max() > 1e-9:
            raise ValueError("mask_img and imgs affines must be identical")
        maps_data, maps_mask, labels = \
                   _trim_maps(maps_data,
                              _safe_get_data(mask_img, ensure_finite=True),
                              keep_empty=True)
        maps_mask = _utils.as_ndarray(maps_mask, dtype=np.bool)
    else:
        maps_mask = np.ones(maps_data.shape[:3], dtype=np.bool)
        labels = np.arange(maps_data.shape[-1], dtype=np.int)

    data = _safe_get_data(imgs, ensure_finite=True)
    region_signals = linalg.lstsq(maps_data[maps_mask, :],
                                  data[maps_mask, :])[0].T

    return region_signals, list(labels)


def signals_to_img_maps(region_signals, maps_img, mask_img=None):
    """Create image from region signals defined as maps.

    region_signals, mask_img must have the same shapes and affines.

    Parameters
    ----------
    region_signals: numpy.ndarray
        signals to process, as a 2D array. A signal is a column. There must
        be as many signals as maps.
        In pseudo-code: region_signals.shape[1] == maps_img.shape[-1]

    maps_img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        Region definitions using maps.

    mask_img: Niimg-like object, optional
        See http://nilearn.github.io/manipulating_images/input_output.html
        Boolean array giving voxels to process. integer arrays also accepted,
        zero meaning False.

    Returns
    -------
    img: nibabel.Nifti1Image
        Reconstructed image. affine and shape are those of maps_img.

    See also
    --------
    nilearn.regions.signals_to_img_labels
    nilearn.regions.img_to_signals_maps
    """

    maps_img = _utils.check_niimg_4d(maps_img)
    maps_data = _safe_get_data(maps_img, ensure_finite=True)
    shape = maps_img.shape[:3]
    affine = _utils.compat.get_affine(maps_img)

    if mask_img is not None:
        mask_img = _utils.check_niimg_3d(mask_img)
        if mask_img.shape != shape:
            raise ValueError("mask_img and maps_img shapes must be identical.")
        if abs(_utils.compat.get_affine(mask_img) - affine).max() > 1e-9:
            raise ValueError("mask_img and maps_img affines must be "
                             "identical.")
        maps_data, maps_mask, _ = _trim_maps(
            maps_data, _safe_get_data(mask_img, ensure_finite=True),
            keep_empty=True)
        maps_mask = _utils.as_ndarray(maps_mask, dtype=np.bool)
    else:
        maps_mask = np.ones(maps_data.shape[:3], dtype=np.bool)

    assert(maps_mask.shape == maps_data.shape[:3])

    data = np.dot(region_signals, maps_data[maps_mask, :].T)
    return masking.unmask(data, new_img_like(maps_img, maps_mask, affine))


def _trim_maps(maps, mask, keep_empty=False, order="F"):
    """Crop maps using a mask.

    No consistency check is performed (esp. on affine). Every required check
    must be performed before calling this function.

    Parameters
    ----------
    maps: numpy.ndarray
        Set of maps, defining some regions.

    mask: numpy.ndarray
        Definition of a mask. The shape must match that of a single map.

    keep_empty: bool
        If False, maps that lie completely outside the mask are dropped from
        the output. If True, they are kept, meaning that maps that are
        completely zero can occur in the output.

    order: "F" or "C"
        Ordering of the output maps array (trimmed_maps).

    Returns
    -------
    trimmed_maps: numpy.ndarray
        New set of maps, computed as intersection of each input map and mask.
        Empty maps are discarded if keep_empty is False, thus the number of
        output maps is not necessarily the same as the number of input maps.
        shape: mask.shape + (output maps number,). Data ordering depends
        on the "order" parameter.

    maps_mask: numpy.ndarray
        Union of all output maps supports. One non-zero value in this
        array guarantees that there is at least one output map that is
        non-zero at this voxel.
        shape: mask.shape. Order is always C.

    indices: numpy.ndarray
        indices of regions that have an non-empty intersection with the
        given mask. len(indices) == trimmed_maps.shape[-1]
    """

    maps = maps.copy()
    sums = abs(maps[_utils.as_ndarray(mask, dtype=np.bool),
                    :]).sum(axis=0)

    if keep_empty:
        n_regions = maps.shape[-1]
    else:
        n_regions = (sums > 0).sum()
    trimmed_maps = np.zeros(maps.shape[:3] + (n_regions, ),
                            dtype=maps.dtype, order=order)
    # use int8 instead of np.bool for Nifti1Image
    maps_mask = np.zeros(mask.shape, dtype=np.int8)

    # iterate on maps
    p = 0
    mask = _utils.as_ndarray(mask, dtype=np.bool, order="C")
    for n, m in enumerate(np.rollaxis(maps, -1)):
        if not keep_empty and sums[n] == 0:
            continue
        trimmed_maps[mask, p] = maps[mask, n]
        maps_mask[trimmed_maps[..., p] > 0] = 1
        p += 1

    if keep_empty:
        return trimmed_maps, maps_mask, np.arange(trimmed_maps.shape[-1],
                                                  dtype=np.int)
    else:
        return trimmed_maps, maps_mask, np.where(sums > 0)[0]
