"""Testing nicom.utils module"""

import re

from nibabel.optpkg import optional_package

from ..utils import find_private_section as fps
from .test_dicomwrappers import DATA, DATA_PHILIPS

pydicom, _, setup_module = optional_package('pydicom')


def test_find_private_section_real():
    # Find section containing named private creator information
    # On real data first
    assert fps(DATA, 0x29, 'SIEMENS CSA HEADER') == 0x1000
    assert fps(DATA, 0x29, b'SIEMENS CSA HEADER') == 0x1000
    assert fps(DATA, 0x29, re.compile('SIEMENS CSA HEADER')) == 0x1000
    assert fps(DATA, 0x29, 'NOT A HEADER') is None
    assert fps(DATA, 0x29, 'SIEMENS MEDCOM HEADER2') == 0x1100
    assert fps(DATA_PHILIPS, 0x29, 'SIEMENS CSA HEADER') == None


def test_find_private_section_fake():
    # Make and test fake datasets
    ds = pydicom.dataset.Dataset({})
    assert fps(ds, 0x11, 'some section') is None
    ds.add_new((0x11, 0x10), 'LO', b'some section')
    assert fps(ds, 0x11, 'some section') == 0x1000
    ds.add_new((0x11, 0x11), 'LO', b'another section')
    ds.add_new((0x11, 0x12), 'LO', b'third section')
    assert fps(ds, 0x11, 'third section') == 0x1200
    # Technically incorrect 'OB' is acceptable for VM (should be 'LO')
    ds.add_new((0x11, 0x12), 'OB', b'third section')
    assert fps(ds, 0x11, 'third section') == 0x1200
    # Anything else not acceptable
    ds.add_new((0x11, 0x12), 'PN', b'third section')
    assert fps(ds, 0x11, 'third section') is None
    # The input (DICOM value) can be a string insteal of bytes
    ds.add_new((0x11, 0x12), 'LO', 'third section')
    assert fps(ds, 0x11, 'third section') == 0x1200
    # Search can be bytes as well as string
    ds.add_new((0x11, 0x12), 'LO', b'third section')
    assert fps(ds, 0x11, b'third section') == 0x1200
    # Search with string or bytes must be exact
    assert fps(ds, 0x11, b'third sectio') is None
    assert fps(ds, 0x11, 'hird sectio') is None
    # The search can be a regexp
    assert fps(ds, 0x11, re.compile(r'third\Wsectio[nN]')) == 0x1200
    # No match -> None
    assert fps(ds, 0x11, re.compile(r'not third\Wsectio[nN]')) is None
    # If there are gaps in the sequence before the one we want, that is OK
    ds.add_new((0x11, 0x13), 'LO', b'near section')
    assert fps(ds, 0x11, 'near section') == 0x1300
    ds.add_new((0x11, 0x15), 'LO', b'far section')
    assert fps(ds, 0x11, 'far section') == 0x1500
    # More than one match - find the first.
    assert fps(ds, 0x11, re.compile('(another|third) section')) == 0x1100
    # The signalling element number must be <= 0xFF
    ds = pydicom.dataset.Dataset({})
    ds.add_new((0x11, 0xFF), 'LO', b'some section')
    assert fps(ds, 0x11, 'some section') == 0xFF00
    ds = pydicom.dataset.Dataset({})
    ds.add_new((0x11, 0x100), 'LO', b'some section')
    assert fps(ds, 0x11, 'some section') is None
