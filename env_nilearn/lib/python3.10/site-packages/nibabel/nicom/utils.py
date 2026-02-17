"""Utilities for working with DICOM datasets"""

from enum import Enum


def find_private_section(dcm_data, group_no, creator):
    """Return start element in group `group_no` given creator name `creator`

    Private attribute tags need to announce where they will go by putting a tag
    in the private group (here `group_no`) between elements 1 and 0xFF.  The
    element number of these tags give the start of matching information, in the
    higher tag numbers.

    Parameters
    ----------
    dcm_data : dicom ``dataset``
        Iterating over `dcm_data` produces ``elements`` with attributes
        ``tag``, ``VR``, ``value``
    group_no : int
        Group number in which to search
    creator : str or bytes or regex
        Name of section - e.g. 'SIEMENS CSA HEADER' - or regex to search for
        section name.  Regex used via ``creator.search(element_value)`` where
        ``element_value`` is the value of the data element.

    Returns
    -------
    element_start : int
        Element number at which named section starts.
    """
    if hasattr(creator, 'search'):
        match_func = creator.search
    else:
        if isinstance(creator, bytes):
            creator = creator.decode('latin-1')
        match_func = creator.__eq__
    # Group elements assumed ordered by tag (groupno, elno)
    for element in dcm_data.group_dataset(group_no):
        elno = element.tag.elem
        if elno > 0xFF:
            break
        if element.VR not in ('LO', 'OB'):
            continue
        val = element.value
        if isinstance(val, bytes):
            val = val.decode('latin-1')
        if match_func(val):
            return elno * 0x100
    return None


class Vendor(Enum):
    SIEMENS = 1
    GE = 2
    PHILIPS = 3


vendor_priv_sections = {
    Vendor.SIEMENS: [
        (0x9, 'SIEMENS SYNGO INDEX SERVICE'),
        (0x19, 'SIEMENS MR HEADER'),
        (0x21, 'SIEMENS MR SDR 01'),
        (0x21, 'SIEMENS MR SDS 01'),
        (0x21, 'SIEMENS MR SDI 02'),
        (0x29, 'SIEMENS CSA HEADER'),
        (0x29, 'SIEMENS MEDCOM HEADER2'),
        (0x51, 'SIEMENS MR HEADER'),
    ],
    Vendor.PHILIPS: [
        (0x2001, 'Philips Imaging DD 001'),
        (0x2001, 'Philips Imaging DD 002'),
        (0x2001, 'Philips Imaging DD 129'),
        (0x2005, 'Philips MR Imaging DD 001'),
        (0x2005, 'Philips MR Imaging DD 002'),
        (0x2005, 'Philips MR Imaging DD 003'),
        (0x2005, 'Philips MR Imaging DD 004'),
        (0x2005, 'Philips MR Imaging DD 005'),
        (0x2005, 'Philips MR Imaging DD 006'),
        (0x2005, 'Philips MR Imaging DD 007'),
        (0x2005, 'Philips MR Imaging DD 005'),
        (0x2005, 'Philips MR Imaging DD 006'),
    ],
    Vendor.GE: [
        (0x9, 'GEMS_IDEN_01'),
        (0x19, 'GEMS_ACQU_01'),
        (0x21, 'GEMS_RELA_01'),
        (0x23, 'GEMS_STDY_01'),
        (0x25, 'GEMS_SERS_01'),
        (0x27, 'GEMS_IMAG_01'),
        (0x29, 'GEMS_IMPS_01'),
        (0x43, 'GEMS_PARM_01'),
    ],
}


def vendor_from_private(dcm_data):
    """Try to determine the vendor by looking for specific private tags"""
    for vendor, priv_sections in vendor_priv_sections.items():
        for priv_group, priv_creator in priv_sections:
            if find_private_section(dcm_data, priv_group, priv_creator) != None:
                return vendor
