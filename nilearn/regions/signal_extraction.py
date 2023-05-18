"""
Functions for extracting region-defined signals.

Two ways of defining regions are supported: as labels in a single 3D image,
or as weights in one image per region (maps).
"""
# Author: Philippe Gervais
# License: simplified BSD

import warnings

import numpy as np
from scipy import linalg, ndimage

from .. import _utils, masking
from .._utils.niimg import _safe_get_data
from ..image import new_img_like

INF = 1000 * np.finfo(np.float32).eps


def _check_shape_compatibility(img1, img2, dim=None):
    """Check that shapes match for dimensions going from 0 to dim-1.

    Parameters
    ----------
    img1 : Niimg-like object
        See :ref:`extracting_data`.
        Image to extract the data from.

    img2 : Niimg-like object, optional
        See :ref:`extracting_data`.
        Contains map or mask.

    dim : :obj:`int`, optional
        Integer slices a mask for a specific dimension.

    """
    if dim is None:
        img2 = _utils.check_niimg_3d(img2)
        if img1.shape[:3] != img2.shape:
            raise ValueError("Images have incompatible shapes.")
    elif img1.shape[:dim] != img2.shape[:dim]:
        raise ValueError("Images have incompatible shapes.")


def _check_affine_equality(img1, img2):
    """Validate affines of 2 images.

    Parameters
    ----------
    img1 : Niimg-like object
        See :ref:`extracting_data`.
        Image to extract the data from.

    img2 : Niimg-like object, optional
        See :ref:`extracting_data`.
        Contains map or mask.

    """
    if (
        img1.affine.shape != img2.affine.shape
        or abs(img1.affine - img2.affine).max() > INF
    ):
        raise ValueError("Images have different affine matrices.")


def _check_shape_and_affine_compatibility(img1, img2=None, dim=None):
    """Validate shapes and affines of 2 images.

    Check that the provided images:
        - have the same shape
        - have the same affine matrix.

    Parameters
    ----------
    img1 : Niimg-like object
        See :ref:`extracting_data`.
        Image to extract the data from.

    img2 : Niimg-like object, optional
        See :ref:`extracting_data`.
        Contains map or mask.

    dim : :obj:`int`, optional
        Integer slices a mask for a specific dimension.

    Returns
    -------
    non_empty : :obj:`bool`,
        Is only true for non-empty img.

    """
    if img2 is None:
        return False

    _check_shape_compatibility(img1, img2, dim=dim)

    if dim is None:
        img2 = _utils.check_niimg_3d(img2)
    _check_affine_equality(img1, img2)

    return True


def _get_labels_data(
    target_img, labels_img, mask_img=None, background_label=0, dim=None
):
    """Get the label data.

    Ensures that labels, imgs and mask shapes and affines fit,
    then extracts the data from it.

    Parameters
    ----------
    target_img : Niimg-like object
        See :ref:`extracting_data`.
        Image to extract the data from.

    labels_img : Niimg-like object
        See :ref:`extracting_data`.
        Regions definition as labels.
        By default, the label zero is used to denote an absence of region.
        Use background_label to change it.

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to labels before extracting signals.
        Every point outside the mask is considered as background
        (i.e. no region).

    background_label : number, optional
        Number representing background in labels_img. Default=0.

    dim : :obj:`int`, optional
        Integer slices mask for a specific dimension.

    Returns
    -------
    labels : :obj:`list` or :obj:`tuple`
        Corresponding labels for each signal.
        signal[:, n] was extracted from the region with label labels[n].

    labels_data : numpy.ndarray
        Extracted data for each region within the mask.
        Data outside the mask are assigned to the background
        label to restrict signal extraction

    See Also
    --------
    nilearn.regions.signals_to_img_labels
    nilearn.regions.img_to_signals_labels

    """
    _check_shape_and_affine_compatibility(target_img, labels_img)

    labels_data = _safe_get_data(labels_img, ensure_finite=True)

    labels = list(np.unique(labels_data))
    if background_label in labels:
        labels.remove(background_label)

    # Consider only data within the mask
    use_mask = _check_shape_and_affine_compatibility(target_img, mask_img, dim)
    if use_mask:
        mask_img = _utils.check_niimg_3d(mask_img)
        mask_data = _safe_get_data(mask_img, ensure_finite=True)
        labels_data = labels_data.copy()
        labels_data[np.logical_not(mask_data)] = background_label

    return labels, labels_data


def _check_reduction_strategy(strategy: str):
    """Check that the provided strategy is supported.

    Parameters
    ----------
    strategy : :obj:`str`
        The name of a valid function to reduce the region with.
        Must be one of: sum, mean, median, minimum, maximum, variance,
        standard_deviation.

    """
    available_reduction_strategies = {
        "mean",
        "median",
        "sum",
        "minimum",
        "maximum",
        "standard_deviation",
        "variance",
    }

    if strategy not in available_reduction_strategies:
        raise ValueError(
            f"Invalid strategy '{strategy}'. "
            f"Valid strategies are {available_reduction_strategies}."
        )


# FIXME: naming scheme is not really satisfying. Any better idea appreciated.
@_utils.fill_doc
def img_to_signals_labels(
    imgs,
    labels_img,
    mask_img=None,
    background_label=0,
    order="F",
    strategy="mean",
):
    """Extract region signals from image.

    This function is applicable to regions defined by labels.

    labels, imgs and mask shapes and affines must fit. This function
    performs no resampling.

    Parameters
    ----------
    %(imgs)s
        Input images.

    labels_img : Niimg-like object
        See :ref:`extracting_data`.
        Regions definition as labels. By default, the label zero is used to
        denote an absence of region. Use background_label to change it.

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to labels before extracting signals.
        Every point outside the mask is considered
        as background (i.e. no region).

    background_label : number, optional
        Number representing background in labels_img. Default=0.

    order : :obj:`str`, optional
        Ordering of output array ("C" or "F"). Default="F".

    strategy : :obj:`str`, optional
        The name of a valid function to reduce the region with.
        Must be one of: sum, mean, median, minimum, maximum, variance,
        standard_deviation. Default="mean".

    Returns
    -------
    signals : :class:`numpy.ndarray`
        Signals extracted from each region. One output signal is the mean
        of all input signals in a given region. If some regions are entirely
        outside the mask, the corresponding signal is zero.
        Shape is: (scan number, number of regions)

    labels : :obj:`list` or :obj:`tuple`
        Corresponding labels for each signal. signal[:, n] was extracted from
        the region with label labels[n].

    See Also
    --------
    nilearn.regions.signals_to_img_labels
    nilearn.regions.img_to_signals_maps
    nilearn.maskers.NiftiLabelsMasker : Signal extraction on labels images
        e.g. clusters

    """
    labels_img = _utils.check_niimg_3d(labels_img)

    _check_reduction_strategy(strategy)

    # TODO: Make a special case for list of strings
    # (load one image at a time).
    imgs = _utils.check_niimg_4d(imgs)
    labels, labels_data = _get_labels_data(
        imgs, labels_img, mask_img, background_label
    )

    data = _safe_get_data(imgs, ensure_finite=True)
    target_datatype = np.float32 if data.dtype == np.float32 else np.float64
    # Nilearn issue: 2135, PR: 2195 for why this is necessary.
    signals = np.ndarray(
        (data.shape[-1], len(labels)), order=order, dtype=target_datatype
    )
    reduction_function = getattr(ndimage, strategy)
    for n, img in enumerate(np.rollaxis(data, -1)):
        signals[n] = np.asarray(
            reduction_function(img, labels=labels_data, index=labels)
        )
    # Set to zero signals for missing labels. Workaround for Scipy behaviour
    missing_labels = set(labels) - set(np.unique(labels_data))
    labels_index = {l: n for n, l in enumerate(labels)}
    for this_label in missing_labels:
        signals[:, labels_index[this_label]] = 0
    return signals, labels


def signals_to_img_labels(
    signals, labels_img, mask_img=None, background_label=0, order="F"
):
    """Create image from region signals defined as labels.

    The same region signal is used for each :term:`voxel` of the
    corresponding 3D volume.

    labels_img, mask_img must have the same shapes and affines.

    .. versionchanged:: 0.9.2
        Support 1D signals.

    Parameters
    ----------
    signals : :class:`numpy.ndarray`
        1D or 2D array.
        If this is a 1D array, it must have as many elements as there are
        regions in the labels_img.
        If it is 2D, it should have the shape
        (number of scans, number of regions in labels_img).

    labels_img : Niimg-like object
        See :ref:`extracting_data`.
        Region definitions using labels.

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Boolean array giving voxels to process. integer arrays also accepted,
        In this array, zero means False, non-zero means True.

    background_label : number, optional
        Label to use for "no region". Default=0.

    order : :obj:`str`, optional
        Ordering of output array ("C" or "F"). Default="F".

    Returns
    -------
    img : :class:`nibabel.nifti1.Nifti1Image`
        Reconstructed image. dtype is that of "signals", affine and shape are
        those of labels_img.

    See Also
    --------
    nilearn.regions.img_to_signals_labels
    nilearn.regions.signals_to_img_maps
    nilearn.maskers.NiftiLabelsMasker : Signal extraction on labels
        images e.g. clusters

    """
    labels_img = _utils.check_niimg_3d(labels_img)

    labels, labels_data = _get_labels_data(
        labels_img, labels_img, mask_img, background_label
    )

    signals = np.asarray(signals)

    target_shape = labels_img.shape[:3]
    # nditer is not available in numpy 1.3: using multiple loops.
    # Using these loops still gives a much faster code (6x) than this one:
    # for n, label in enumerate(labels):
    #     data[labels_data == label, :] = signals[:, n]
    if signals.ndim == 2:
        target_shape = target_shape + (signals.shape[0],)

    data = np.zeros(target_shape, dtype=signals.dtype, order=order)
    labels_dict = {label: n for n, label in enumerate(labels)}
    # optimized for "data" in F order.
    for k in range(labels_data.shape[2]):
        for j in range(labels_data.shape[1]):
            for i in range(labels_data.shape[0]):
                label = labels_data[i, j, k]
                num = labels_dict.get(label)
                if num is not None:
                    if signals.ndim == 2:
                        data[i, j, k, :] = signals[:, num]
                    else:
                        data[i, j, k] = signals[num]

    return new_img_like(labels_img, data, labels_img.affine)


@_utils.fill_doc
def img_to_signals_maps(
    imgs, maps_img, mask_img=None,
    keep_masked_maps=True
):
    """Extract region signals from image.

    This function is applicable to regions defined by maps.

    Parameters
    ----------
    %(imgs)s
        Input images.

    maps_img : Niimg-like object
        See :ref:`extracting_data`.
        Regions definition as maps (array of weights).
        shape: imgs.shape + (region number, )

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to regions before extracting signals.
        Every point outside the mask is considered
        as background (i.e. outside of any region).

    keep_masked_maps : :obj:`bool`, optional
        If True, masked atlas with invalid maps (maps with no brain coverage
        after applying the mask) will be retained in the output, resulting
        in corresponding time series containing zeros only. If False, the
        invalid maps will be removed from the trimmed atlas, resulting in
        no empty time series in the output.

        .. deprecated:: 0.10.2
            The 'True' option for ``keep_masked_maps`` is deprecated.
            The default value will change to 'False' in 0.13,
            and the ``keep_masked_maps`` parameter will be removed in 0.15.

    Returns
    -------
    region_signals : :class:`numpy.ndarray`
        Signals extracted from each region.
        Shape is: (scans number, number of regions intersecting mask)

    labels : :obj:`list`
        maps_img[..., labels[n]] is the region that has been used to extract
        signal region_signals[:, n].

    See Also
    --------
    nilearn.regions.img_to_signals_labels
    nilearn.regions.signals_to_img_maps
    nilearn.maskers.NiftiMapsMasker : Signal extraction on probabilistic
        maps e.g. ICA

    """
    maps_img = _utils.check_niimg_4d(maps_img)
    imgs = _utils.check_niimg_4d(imgs)

    _check_shape_and_affine_compatibility(imgs, maps_img, 3)

    maps_data = _safe_get_data(maps_img, ensure_finite=True)
    maps_mask = np.ones(maps_data.shape[:3], dtype=bool)
    labels = np.arange(maps_data.shape[-1], dtype=int)

    use_mask = _check_shape_and_affine_compatibility(imgs, mask_img)
    if use_mask:
        mask_img = _utils.check_niimg_3d(mask_img)
        labels_before_mask = set(labels)
        maps_data, maps_mask, labels = _trim_maps(
            maps_data,
            _safe_get_data(mask_img, ensure_finite=True),
            keep_empty=keep_masked_maps,
        )
        maps_mask = _utils.as_ndarray(maps_mask, dtype=bool)
        if keep_masked_maps:
            warnings.warn(
                'Starting in version 0.15, the behavior of "NiftiMapsMasker" '
                'will change when a mask is supplied through the "mask_img" '
                'parameter. Applying "mask_img" before '
                'signal extraction may result in empty region signals in the '
                'output. These empty region signals used to be kept. '
                'In the new behavior, they will be removed from the output.'
                '\n'
                'To explicitly enable/disable the retention of empty '
                'region signals, set the parameter "keep_masked_maps" to '
                'True/False when '
                'initializing the "NiftiMapsMasker" object. '
                'Starting from version 0.13, the default behavior will be '
                'changed to "keep_masked_maps=False". '
                '"keep_masked_maps" parameter will be removed '
                'in version 0.15.',
                DeprecationWarning,
                stacklevel=2
            )
        else:
            labels_after_mask = set(labels)
            labels_diff = labels_before_mask.difference(
                labels_after_mask
            )
            # Raising a warning if any map is removed due to the mask
            if len(labels_diff) > 0:
                warnings.warn(
                    "After applying mask to the maps image, "
                    "the following maps were "
                    f"removed: {labels_diff}. "
                    f"Out of {len(labels_before_mask)} maps, the "
                    "masked map image only contains "
                    f"{len(labels_after_mask)} maps.",
                    stacklevel=2
                )

    data = _safe_get_data(imgs, ensure_finite=True)
    region_signals = linalg.lstsq(maps_data[maps_mask, :], data[maps_mask, :])[
        0
    ].T

    return region_signals, list(labels)


def signals_to_img_maps(region_signals, maps_img, mask_img=None):
    """Create image from region signals defined as maps.

    region_signals, mask_img must have the same shapes and affines.

    Parameters
    ----------
    region_signals : :class:`numpy.ndarray`
        signals to process, as a 2D array. A signal is a column.
        There must be as many signals as maps:

            .. code-block:: python

                region_signals.shape[1] == maps_img.shape[-1]

    maps_img : Niimg-like object
        See :ref:`extracting_data`.
        Region definitions using maps.

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Boolean array giving :term:`voxels<voxel>` to process.
        Integer arrays also accepted, zero meaning False.

    Returns
    -------
    img : :class:`nibabel.nifti1.Nifti1Image`
        Reconstructed image. affine and shape are those of maps_img.

    See Also
    --------
    nilearn.regions.signals_to_img_labels
    nilearn.regions.img_to_signals_maps
    nilearn.maskers.NiftiMapsMasker

    """
    maps_img = _utils.check_niimg_4d(maps_img)
    maps_data = _safe_get_data(maps_img, ensure_finite=True)

    maps_mask = np.ones(maps_data.shape[:3], dtype=bool)

    use_mask = _check_shape_and_affine_compatibility(maps_img, mask_img)
    if use_mask:
        mask_img = _utils.check_niimg_3d(mask_img)
        maps_data, maps_mask, _ = _trim_maps(
            maps_data,
            _safe_get_data(mask_img, ensure_finite=True),
            keep_empty=True,
        )
        maps_mask = _utils.as_ndarray(maps_mask, dtype=bool)
        assert maps_mask.shape == maps_data.shape[:3]

    data = np.dot(region_signals, maps_data[maps_mask, :].T)
    return masking.unmask(
        data, new_img_like(maps_img, maps_mask, maps_img.affine)
    )


def _trim_maps(maps, mask, keep_empty=False, order="F"):
    """Crop maps using a mask.

    No consistency check is performed (esp. on affine). Every required check
    must be performed before calling this function.

    Parameters
    ----------
    maps : :class:`numpy.ndarray`
        Set of maps, defining some regions.

    mask : :class:`numpy.ndarray`
        Definition of a mask. The shape must match that of a single map.

    keep_empty : :obj:`bool`, optional
        If False, maps that lie completely outside the mask are dropped from
        the output. If True, they are kept, meaning that maps that are
        completely zero can occur in the output.
        Default=False.

    order : "F" or "C", optional
        Ordering of the output maps array (trimmed_maps).
        Default="F".

    Returns
    -------
    trimmed_maps : :class:``numpy.ndarray`
        New set of maps, computed as intersection of each input map and mask.
        Empty maps are discarded if keep_empty is False, thus the number of
        output maps is not necessarily the same as the number of input maps.
        shape: mask.shape + (output maps number,). Data ordering depends
        on the "order" parameter.

    maps_mask : :class:`numpy.ndarray`
        Union of all output maps supports. One non-zero value in this
        array guarantees that there is at least one output map that is
        non-zero at this voxel.
        shape: mask.shape. Order is always C.

    indices : :class:`numpy.ndarray`
        Indices of regions that have an non-empty intersection with the
        given mask. len(indices) == trimmed_maps.shape[-1].

    """
    maps = maps.copy()
    sums = abs(maps[_utils.as_ndarray(mask, dtype=bool), :]).sum(axis=0)

    n_regions = maps.shape[-1] if keep_empty else (sums > 0).sum()
    trimmed_maps = np.zeros(
        maps.shape[:3] + (n_regions,), dtype=maps.dtype, order=order
    )
    # use int8 instead of np.bool for Nifti1Image
    maps_mask = np.zeros(mask.shape, dtype=np.int8)

    # iterate on maps
    p = 0
    mask = _utils.as_ndarray(mask, dtype=bool, order="C")
    for n, _ in enumerate(np.rollaxis(maps, -1)):
        if not keep_empty and sums[n] == 0:
            continue
        trimmed_maps[mask, p] = maps[mask, n]
        maps_mask[trimmed_maps[..., p] > 0] = 1
        p += 1

    indices = (
        np.arange(trimmed_maps.shape[-1], dtype=int)
        if keep_empty
        else np.where(sums > 0)[0]
    )

    return trimmed_maps, maps_mask, indices
