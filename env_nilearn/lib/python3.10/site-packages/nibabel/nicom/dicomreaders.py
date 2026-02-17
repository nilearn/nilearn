import glob
from os.path import join as pjoin

import numpy as np

from .. import Nifti1Image
from .dicomwrappers import wrapper_from_data, wrapper_from_file


class DicomReadError(Exception):
    pass


DPCS_TO_TAL = np.diag([-1, -1, 1, 1])


def mosaic_to_nii(dcm_data):
    """Get Nifti file from Siemens

    Parameters
    ----------
    dcm_data : ``dicom.DataSet``
       DICOM header / image as read by ``dicom`` package

    Returns
    -------
    img : ``Nifti1Image``
       Nifti image object
    """
    dcm_w = wrapper_from_data(dcm_data)
    if not dcm_w.is_mosaic:
        raise DicomReadError('data does not appear to be in mosaic format')
    data = dcm_w.get_data()
    aff = np.dot(DPCS_TO_TAL, dcm_w.affine)
    return Nifti1Image(data, aff)


def read_mosaic_dwi_dir(dicom_path, globber='*.dcm', dicom_kwargs=None):
    return read_mosaic_dir(dicom_path, globber, check_is_dwi=True, dicom_kwargs=dicom_kwargs)


def read_mosaic_dir(dicom_path, globber='*.dcm', check_is_dwi=False, dicom_kwargs=None):
    """Read all Siemens mosaic DICOMs in directory, return arrays, params

    Parameters
    ----------
    dicom_path : str
       path containing mosaic DICOM images
    globber : str, optional
       glob to apply within `dicom_path` to select DICOM files.  Default
       is ``*.dcm``
    check_is_dwi : bool, optional
       If True, raises an error if we don't find DWI information in the
       DICOM headers.
    dicom_kwargs : None or dict
       Extra keyword arguments to pass to the pydicom ``dcmread`` function.

    Returns
    -------
    data : 4D array
       data array with last dimension being acquisition. If there were N
       acquisitions, each of shape (X, Y, Z), `data` will be shape (X,
       Y, Z, N)
    affine : (4,4) array
       affine relating 3D voxel space in data to RAS world space
    b_values : (N,) array
       b values for each acquisition.  nan if we did not find diffusion
       information for these images.
    unit_gradients : (N, 3) array
       gradient directions of unit length for each acquisition.  (nan,
       nan, nan) if we did not find diffusion information.
    """
    if dicom_kwargs is None:
        dicom_kwargs = {}
    full_globber = pjoin(dicom_path, globber)
    filenames = sorted(glob.glob(full_globber))
    b_values = []
    gradients = []
    arrays = []
    if len(filenames) == 0:
        raise OSError(f'Found no files with "{full_globber}"')
    for fname in filenames:
        dcm_w = wrapper_from_file(fname, **dicom_kwargs)
        # Because the routine sorts by filename, it only makes sense to use
        # this order for mosaic images.  Slice by slice dicoms need more
        # sensible sorting
        if not dcm_w.is_mosaic:
            raise DicomReadError('data does not appear to be in mosaic format')
        arrays.append(dcm_w.get_data()[..., None])
        q = dcm_w.q_vector
        if q is None:  # probably not diffusion
            if check_is_dwi:
                raise DicomReadError(
                    f'Could not find diffusion information reading file "{fname}";  '
                    'is it possible this is not a _raw_ diffusion directory? '
                    'Could it be a processed dataset like ADC etc?'
                )
            b = np.nan
            g = np.ones((3,)) + np.nan
        else:
            b = dcm_w.b_value
            g = dcm_w.b_vector
        b_values.append(b)
        gradients.append(g)
    affine = np.dot(DPCS_TO_TAL, dcm_w.affine)
    return (np.concatenate(arrays, -1), affine, np.array(b_values), np.array(gradients))


def slices_to_series(wrappers):
    """Sort sequence of slice wrappers into series

    This follows the SPM model fairly closely

    Parameters
    ----------
    wrappers : sequence
       sequence of ``Wrapper`` objects for sorting into volumes

    Returns
    -------
    series : sequence
       sequence of sequences of wrapper objects, where each sequence is
       wrapper objects comprising a series, sorted into slice order
    """
    # first pass
    volume_lists = [wrappers[0:1]]
    for dw in wrappers[1:]:
        for vol_list in volume_lists:
            if dw.is_same_series(vol_list[0]):
                vol_list.append(dw)
                break
        else:  # no match in current volume lists
            volume_lists.append([dw])
    print(f'We appear to have {len(volume_lists)} Series')
    # second pass
    out_vol_lists = []
    for vol_list in volume_lists:
        if len(vol_list) > 1:
            vol_list.sort(key=_slice_sorter)
            zs = [s.slice_indicator for s in vol_list]
            if len(set(zs)) < len(zs):  # not unique zs
                # third pass
                out_vol_lists += _third_pass(vol_list)
                continue
        out_vol_lists.append(vol_list)
    print(f'We have {len(out_vol_lists)} volumes after second pass')
    # final pass check
    for vol_list in out_vol_lists:
        zs = [s.slice_indicator for s in vol_list]
        diffs = np.diff(zs)
        if not np.allclose(diffs, np.mean(diffs)):
            raise DicomReadError('Largeish slice gaps - missing DICOMs?')
    return out_vol_lists


def _slice_sorter(s):
    return s.slice_indicator


def _instance_sorter(s):
    return s.instance_number


def _third_pass(wrappers):
    """What we do when there are not unique zs in a slice set"""
    inos = [s.instance_number for s in wrappers]
    msg_fmt = (
        'Plausibly matching slices, but where some have '
        'the same apparent slice location, and %s; '
        '- slices are probably unsortable'
    )
    if None in inos:
        raise DicomReadError(msg_fmt % 'some or all slices with missing InstanceNumber')
    if len(set(inos)) < len(inos):
        raise DicomReadError(msg_fmt % 'some or all slices with the same InstanceNumber')
    # sort by instance number
    wrappers.sort(key=_instance_sorter)
    # start loop, in which we start a new volume, each time we see a z
    # we've seen already in the current volume
    dw = wrappers[0]
    these_zs = [dw.slice_indicator]
    vol_list = [dw]
    out_vol_lists = [vol_list]
    for dw in wrappers[1:]:
        z = dw.slice_indicator
        if z not in these_zs:
            # same volume
            vol_list.append(dw)
            these_zs.append(z)
            continue
        # new volume
        vol_list.sort(_slice_sorter)
        vol_list = [dw]
        these_zs = [z]
        out_vol_lists.append(vol_list)
    vol_list.sort(_slice_sorter)
    return out_vol_lists
