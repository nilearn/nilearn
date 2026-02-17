from unittest import mock

import pytest

from ..pkg_info import cmp_pkg_version

MODULE_SCHEDULE = [
    ('7.0.0', ['nibabel.pydicom_compat']),
    ('5.0.0', ['nibabel.keywordonly', 'nibabel.py3k']),
    ('4.0.0', ['nibabel.trackvis']),
    ('3.0.0', ['nibabel.minc', 'nibabel.checkwarns']),
    # Verify that the test will be quiet if the schedule outlives the modules
    ('1.0.0', ['nibabel.nosuchmod']),
]

OBJECT_SCHEDULE = [
    (
        '8.0.0',
        [
            ('nibabel.casting', 'as_int'),
            ('nibabel.casting', 'int_to_float'),
            ('nibabel.tmpdirs', 'TemporaryDirectory'),
        ],
    ),
    (
        '7.0.0',
        [
            ('nibabel.gifti.gifti', 'GiftiNVPairs'),
        ],
    ),
    (
        '6.0.0',
        [
            ('nibabel.loadsave', 'guessed_image_type'),
            ('nibabel.loadsave', 'read_img_data'),
            ('nibabel.orientations', 'flip_axis'),
            ('nibabel.pydicom_compat', 'dicom_test'),
            ('nibabel.onetime', 'setattr_on_read'),
        ],
    ),
    (
        '5.0.0',
        [
            ('nibabel.gifti.gifti', 'data_tag'),
            ('nibabel.gifti.giftiio', 'read'),
            ('nibabel.gifti.giftiio', 'write'),
            ('nibabel.gifti.parse_gifti_fast', 'Outputter'),
            ('nibabel.gifti.parse_gifti_fast', 'parse_gifti_file'),
            ('nibabel.imageclasses', 'ext_map'),
            ('nibabel.imageclasses', 'class_map'),
            ('nibabel.loadsave', 'which_analyze_type'),
            ('nibabel.volumeutils', 'BinOpener'),
            ('nibabel.volumeutils', 'allopen'),
            ('nibabel.orientations', 'orientation_affine'),
            ('nibabel.spatialimages', 'Header'),
        ],
    ),
    ('4.0.0', [('nibabel.minc1', 'MincFile'), ('nibabel.minc1', 'MincImage')]),
    ('3.0.0', [('nibabel.testing', 'catch_warn_reset')]),
    # Verify that the test will be quiet if the schedule outlives the modules
    ('1.0.0', [('nibabel.nosuchmod', 'anyobj'), ('nibabel.nifti1', 'nosuchobj')]),
]

ATTRIBUTE_SCHEDULE = [
    (
        '7.0.0',
        [
            ('nibabel.gifti.gifti', 'GiftiMetaData', 'from_dict'),
            ('nibabel.gifti.gifti', 'GiftiMetaData', 'metadata'),
            ('nibabel.gifti.gifti', 'GiftiMetaData', 'data'),
        ],
    ),
    (
        '5.0.0',
        [
            ('nibabel.dataobj_images', 'DataobjImage', 'get_data'),
            ('nibabel.freesurfer.mghformat', 'MGHHeader', '_header_data'),
            ('nibabel.gifti.gifti', 'GiftiDataArray', 'from_array'),
            ('nibabel.gifti.gifti', 'GiftiDataArray', 'to_xml_open'),
            ('nibabel.gifti.gifti', 'GiftiDataArray', 'to_xml_close'),
            ('nibabel.gifti.gifti', 'GiftiDataArray', 'get_metadata'),
            ('nibabel.gifti.gifti', 'GiftiImage', 'get_labeltable'),
            ('nibabel.gifti.gifti', 'GiftiImage', 'set_labeltable'),
            ('nibabel.gifti.gifti', 'GiftiImage', 'get_metadata'),
            ('nibabel.gifti.gifti', 'GiftiImage', 'set_metadata'),
            ('nibabel.gifti.gifti', 'GiftiImage', 'getArraysFromIntent'),
            ('nibabel.gifti.gifti', 'GiftiMetaData', 'get_metadata'),
            ('nibabel.gifti.gifti', 'GiftiLabel', 'get_rgba'),
            ('nibabel.nicom.dicomwrappers', 'Wrapper', 'get_affine'),
            ('nibabel.streamlines.array_sequence', 'ArraySequence', 'data'),
            ('nibabel.ecat', 'EcatImage', 'from_filespec'),
            ('nibabel.filebasedimages', 'FileBasedImage', 'get_header'),
            ('nibabel.spatialimages', 'SpatialImage', 'get_affine'),
            ('nibabel.arraywriters', 'ArrayWriter', '_check_nan2zero'),
        ],
    ),
    (
        '4.0.0',
        [
            ('nibabel.dataobj_images', 'DataobjImage', 'get_shape'),
            ('nibabel.filebasedimages', 'FileBasedImage', 'filespec_to_files'),
            ('nibabel.filebasedimages', 'FileBasedImage', 'to_filespec'),
            ('nibabel.filebasedimages', 'FileBasedImage', 'to_files'),
            ('nibabel.filebasedimages', 'FileBasedImage', 'from_files'),
            ('nibabel.arrayproxy', 'ArrayProxy', 'header'),
        ],
    ),
    # Verify that the test will be quiet if the schedule outlives the modules
    (
        '1.0.0',
        [
            ('nibabel.nosuchmod', 'anyobj', 'anyattr'),
            ('nibabel.nifti1', 'nosuchobj', 'anyattr'),
            ('nibabel.nifti1', 'Nifti1Image', 'nosuchattr'),
        ],
    ),
]


def _filter(schedule):
    return [entry for ver, entries in schedule if cmp_pkg_version(ver) < 1 for entry in entries]


def test_module_removal():
    for module in _filter(MODULE_SCHEDULE):
        with pytest.raises(ImportError):
            __import__(module)
            raise AssertionError(f'Time to remove {module}')


def test_object_removal():
    for module_name, obj in _filter(OBJECT_SCHEDULE):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        assert not hasattr(module, obj), f'Time to remove {module_name}.{obj}'


def test_attribute_removal():
    for module_name, cls, attr in _filter(ATTRIBUTE_SCHEDULE):
        try:
            module = __import__(module_name)
        except ImportError:
            continue
        try:
            klass = getattr(module, cls)
        except AttributeError:
            continue
        assert not hasattr(klass, attr), f'Time to remove {module_name}.{cls}.{attr}'


#
# Test the tests, making sure that we will get errors when the time comes
#

_sched = 'nibabel.tests.test_removalschedule.{}_SCHEDULE'.format


@mock.patch(_sched('MODULE'), [('3.0.0', ['nibabel.nifti1'])])
def test_unremoved_module():
    with pytest.raises(AssertionError):
        test_module_removal()


@mock.patch(_sched('OBJECT'), [('3.0.0', [('nibabel.nifti1', 'Nifti1Image')])])
def test_unremoved_object():
    with pytest.raises(AssertionError):
        test_object_removal()


@mock.patch(_sched('ATTRIBUTE'), [('3.0.0', [('nibabel.nifti1', 'Nifti1Image', 'affine')])])
def test_unremoved_attr():
    with pytest.raises(AssertionError):
        test_attribute_removal()
