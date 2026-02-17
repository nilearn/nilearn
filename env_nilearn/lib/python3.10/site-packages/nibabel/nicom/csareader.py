"""CSA header reader from SPM spec"""

import numpy as np

from .structreader import Unpacker
from .utils import find_private_section

# DICOM VR code to Python type
_CONVERTERS = {
    'FL': float,  # float
    'FD': float,  # double
    'DS': float,  # decimal string
    'SS': int,  # signed short
    'US': int,  # unsigned short
    'SL': int,  # signed long
    'UL': int,  # unsigned long
    'IS': int,  # integer string
}

MAX_CSA_ITEMS = 1000


class CSAError(Exception):
    pass


class CSAReadError(CSAError):
    pass


def get_csa_header(dcm_data, csa_type='image'):
    """Get CSA header information from DICOM header

    Return None if the header does not contain CSA information of the
    specified `csa_type`

    Parameters
    ----------
    dcm_data : dicom.Dataset
       DICOM dataset.  Should implement ``__getitem__`` and, if initial check
       for presence of ``dcm_data[(0x29, 0x10)]`` passes, should satisfy
       interface for ``find_private_section``.
    csa_type : {'image', 'series'}, optional
       Type of CSA field to read; default is 'image'

    Returns
    -------
    csa_info : None or dict
       Parsed CSA field of `csa_type` or None, if we cannot find the CSA
       information.
    """
    csa_type = csa_type.lower()
    if csa_type == 'image':
        element_offset = 0x10
    elif csa_type == 'series':
        element_offset = 0x20
    else:
        raise ValueError(f'Invalid CSA header type "{csa_type}"')
    if not (0x29, 0x10) in dcm_data:  # Cannot be Siemens CSA
        return None
    section_start = find_private_section(dcm_data, 0x29, 'SIEMENS CSA HEADER')
    if section_start is None:
        return None
    element_no = section_start + element_offset
    try:
        tag = dcm_data[(0x29, element_no)]
    except KeyError:
        # The element could be missing due to anonymization
        return None
    return read(tag.value)


def read(csa_str):
    """Read CSA header from string `csa_str`

    Parameters
    ----------
    csa_str : str
       byte string containing CSA header information

    Returns
    -------
    header : dict
       header information as dict, where `header` has fields (at least)
       ``type, n_tags, tags``.  ``header['tags']`` is also a dictionary
       with one key, value pair for each tag in the header.
    """
    csa_len = len(csa_str)
    csa_dict = {'tags': {}}
    hdr_id = csa_str[:4]
    up_str = Unpacker(csa_str, endian='<')
    if hdr_id == b'SV10':  # CSA2
        hdr_type = 2
        up_str.ptr = 4  # omit the SV10
        csa_dict['unused0'] = up_str.read(4)
    else:  # CSA1
        hdr_type = 1
    csa_dict['type'] = hdr_type
    csa_dict['n_tags'], csa_dict['check'] = up_str.unpack('2I')
    if not 0 < csa_dict['n_tags'] <= MAX_CSA_ITEMS:
        raise CSAReadError(
            f'Number of tags `t` should be 0 < t <= {MAX_CSA_ITEMS}. '
            f'Instead found {csa_dict["n_tags"]} tags.'
        )
    for tag_no in range(csa_dict['n_tags']):
        name, vm, vr, syngodt, n_items, last3 = up_str.unpack('64si4s3i')
        vr = nt_str(vr)
        name = nt_str(name)
        tag = {
            'n_items': n_items,
            'vm': vm,  # value multiplicity
            'vr': vr,  # value representation
            'syngodt': syngodt,
            'last3': last3,
            'tag_no': tag_no,
        }
        if vm == 0:
            n_values = n_items
        else:
            n_values = vm
        # data converter
        converter = _CONVERTERS.get(vr)
        # CSA1 specific length modifier
        if tag_no == 1:
            tag0_n_items = n_items
        if n_items > MAX_CSA_ITEMS:
            raise CSAReadError(f'Expected <= {MAX_CSA_ITEMS} tags, got {n_items}')
        items = []
        for item_no in range(n_items):
            x0, x1, x2, x3 = up_str.unpack('4i')
            ptr = up_str.ptr
            if hdr_type == 1:  # CSA1 - odd length calculation
                item_len = x0 - tag0_n_items
                if item_len < 0 or (ptr + item_len) > csa_len:
                    if item_no < vm:
                        items.append('')
                    break
            else:  # CSA2
                item_len = x1
                if (ptr + item_len) > csa_len:
                    raise CSAReadError('Item is too long, aborting read')
            if item_no >= n_values:
                assert item_len == 0
                continue
            item = nt_str(up_str.read(item_len))
            if converter:
                # we may have fewer real items than are given in
                # n_items, but we don't know how many - assume that
                # we've reached the end when we hit an empty item
                if item_len == 0:
                    n_values = item_no
                    continue
                item = converter(item)
            items.append(item)
            # go to 4 byte boundary
            plus4 = item_len % 4
            if plus4 != 0:
                up_str.ptr += 4 - plus4
        tag['items'] = items
        csa_dict['tags'][name] = tag
    return csa_dict


def get_scalar(csa_dict, tag_name):
    try:
        items = csa_dict['tags'][tag_name]['items']
    except KeyError:
        return None
    if len(items) == 0:
        return None
    return items[0]


def get_vector(csa_dict, tag_name, n):
    try:
        items = csa_dict['tags'][tag_name]['items']
    except KeyError:
        return None
    if len(items) == 0:
        return None
    if len(items) != n:
        raise ValueError(f'Expecting {n} vector')
    return np.array(items)


def is_mosaic(csa_dict):
    """Return True if the data is of Mosaic type

    Parameters
    ----------
    csa_dict : dict
       dict containing read CSA data

    Returns
    -------
    tf : bool
       True if the `dcm_data` appears to be of Siemens mosaic type,
       False otherwise
    """
    if csa_dict is None:
        return False
    if get_acq_mat_txt(csa_dict) is None:
        return False
    n_o_m = get_n_mosaic(csa_dict)
    return not (n_o_m is None) and n_o_m != 0


def get_n_mosaic(csa_dict):
    return get_scalar(csa_dict, 'NumberOfImagesInMosaic')


def get_acq_mat_txt(csa_dict):
    return get_scalar(csa_dict, 'AcquisitionMatrixText')


def get_slice_normal(csa_dict):
    return get_vector(csa_dict, 'SliceNormalVector', 3)


def get_b_matrix(csa_dict):
    vals = get_vector(csa_dict, 'B_matrix', 6)
    if vals is None:
        return
    # the 6 vector is the upper triangle of the symmetric B matrix
    inds = np.array([0, 1, 2, 1, 3, 4, 2, 4, 5])
    B = np.array(vals)[inds]
    return B.reshape(3, 3)


def get_b_value(csa_dict):
    return get_scalar(csa_dict, 'B_value')


def get_g_vector(csa_dict):
    return get_vector(csa_dict, 'DiffusionGradientDirection', 3)


def get_ice_dims(csa_dict):
    dims = get_scalar(csa_dict, 'ICE_Dims')
    if dims is None:
        return None
    return dims.split('_')


def nt_str(s):
    """Strip string to first null

    Parameters
    ----------
    s : bytes

    Returns
    -------
    sdash : str
       s stripped to first occurrence of null (0)
    """
    zero_pos = s.find(b'\x00')
    if zero_pos == -1:
        return s
    return s[:zero_pos].decode('latin-1')
