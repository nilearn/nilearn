"""Testing CIFTI-2 objects"""

import collections
from xml.etree import ElementTree

import numpy as np
import pytest

from nibabel import cifti2 as ci
from nibabel.cifti2.cifti2 import _float_01, _value_if_klass
from nibabel.nifti2 import Nifti2Header
from nibabel.tests.test_dataobj_images import TestDataobjAPI as _TDA
from nibabel.tests.test_image_api import DtypeOverrideMixin, SerializeMixin


def compare_xml_leaf(str1, str2):
    x1 = ElementTree.fromstring(str1)
    x2 = ElementTree.fromstring(str2)
    if len(x1) > 0 or len(x2) > 0:
        raise ValueError

    test = (x1.tag == x2.tag) and (x1.attrib == x2.attrib) and (x1.text == x2.text)
    print((x1.tag, x1.attrib, x1.text))
    print((x2.tag, x2.attrib, x2.text))
    return test


def test_value_if_klass():
    assert _value_if_klass(None, list) is None
    assert _value_if_klass([1], list) == [1]
    with pytest.raises(ValueError):
        _value_if_klass(1, list)


def test_cifti2_metadata():
    md = ci.Cifti2MetaData({'a': 'aval'})
    assert len(md) == 1
    assert list(iter(md)) == ['a']
    assert md['a'] == 'aval'
    assert md.data == {'a': 'aval'}

    with pytest.warns(FutureWarning):
        md = ci.Cifti2MetaData(metadata={'a': 'aval'})
    assert md == {'a': 'aval'}

    with pytest.warns(FutureWarning):
        md = ci.Cifti2MetaData(None)
    assert md == {}

    md = ci.Cifti2MetaData()
    assert len(md) == 0
    assert list(iter(md)) == []
    assert md.data == {}
    with pytest.raises(ValueError):
        md.difference_update(None)

    md['a'] = 'aval'
    assert md['a'] == 'aval'
    assert len(md) == 1
    assert md.data == {'a': 'aval'}

    del md['a']
    assert len(md) == 0

    metadata_test = [('a', 'aval'), ('b', 'bval')]
    md.update(metadata_test)
    assert md.data == dict(metadata_test)

    assert list(iter(md)) == list(iter(collections.OrderedDict(metadata_test)))

    md.update({'a': 'aval', 'b': 'bval'})
    assert md.data == dict(metadata_test)

    md.update({'a': 'aval', 'd': 'dval'})
    assert md.data == dict(metadata_test + [('d', 'dval')])

    md.difference_update({'a': 'aval', 'd': 'dval'})
    assert md.data == dict(metadata_test[1:])

    with pytest.raises(KeyError):
        md.difference_update({'a': 'aval', 'd': 'dval'})
    assert md.to_xml() == b'<MetaData><MD><Name>b</Name><Value>bval</Value></MD></MetaData>'


def test__float_01():
    assert _float_01(0) == 0
    assert _float_01(1) == 1
    assert _float_01('0') == 0
    assert _float_01('0.2') == 0.2
    with pytest.raises(ValueError):
        _float_01(1.1)
    with pytest.raises(ValueError):
        _float_01(-0.1)
    with pytest.raises(ValueError):
        _float_01(2)
    with pytest.raises(ValueError):
        _float_01(-1)
    with pytest.raises(ValueError):
        _float_01('foo')


def test_cifti2_labeltable():
    lt = ci.Cifti2LabelTable()
    assert len(lt) == 0
    with pytest.raises(ci.Cifti2HeaderError):
        lt.to_xml()
    with pytest.raises(ci.Cifti2HeaderError):
        lt._to_xml_element()

    label = ci.Cifti2Label(label='Test', key=0)
    lt[0] = label
    assert len(lt) == 1
    assert dict(lt) == {label.key: label}

    lt.clear()
    lt.append(label)
    assert len(lt) == 1
    assert dict(lt) == {label.key: label}

    lt.clear()
    test_tuple = (label.label, label.red, label.green, label.blue, label.alpha)
    lt[label.key] = test_tuple
    assert len(lt) == 1
    v = lt[label.key]
    assert (v.label, v.red, v.green, v.blue, v.alpha) == test_tuple

    with pytest.raises(ValueError):
        lt[1] = label

    with pytest.raises(ValueError):
        lt[0] = test_tuple[:-1]

    with pytest.raises(ValueError):
        lt[0] = ('foo', 1.1, 0, 0, 1)

    with pytest.raises(ValueError):
        lt[0] = ('foo', 1.0, -1, 0, 1)

    with pytest.raises(ValueError):
        lt[0] = ('foo', 1.0, 0, -0.1, 1)


def test_cifti2_label():
    lb = ci.Cifti2Label()
    lb.label = 'Test'
    lb.key = 0
    assert lb.rgba == (0, 0, 0, 0)
    assert compare_xml_leaf(
        lb.to_xml().decode('utf-8'),
        "<Label Key='0' Red='0' Green='0' Blue='0' Alpha='0'>Test</Label>",
    )

    lb.red = 0
    lb.green = 0.1
    lb.blue = 0.2
    lb.alpha = 0.3
    assert lb.rgba == (0, 0.1, 0.2, 0.3)

    assert compare_xml_leaf(
        lb.to_xml().decode('utf-8'),
        "<Label Key='0' Red='0' Green='0.1' Blue='0.2' Alpha='0.3'>Test</Label>",
    )

    lb.red = 10
    with pytest.raises(ci.Cifti2HeaderError):
        lb.to_xml()
    lb.red = 0

    lb.key = 'a'
    with pytest.raises(ci.Cifti2HeaderError):
        lb.to_xml()
    lb.key = 0


def test_cifti2_parcel():
    pl = ci.Cifti2Parcel()
    with pytest.raises(ci.Cifti2HeaderError):
        pl.to_xml()

    with pytest.raises(TypeError):
        pl.append_cifti_vertices(None)

    with pytest.raises(ValueError):
        ci.Cifti2Parcel(vertices=[1, 2, 3])

    pl = ci.Cifti2Parcel(
        name='region',
        voxel_indices_ijk=ci.Cifti2VoxelIndicesIJK([[1, 2, 3]]),
        vertices=[ci.Cifti2Vertices([0, 1, 2])],
    )
    pl.pop_cifti2_vertices(0)

    assert len(pl.vertices) == 0
    assert (
        pl.to_xml() == b'<Parcel Name="region"><VoxelIndicesIJK>1 2 3</VoxelIndicesIJK></Parcel>'
    )


def test_cifti2_vertices():
    vs = ci.Cifti2Vertices()
    with pytest.raises(ci.Cifti2HeaderError):
        vs.to_xml()

    vs.brain_structure = 'CIFTI_STRUCTURE_OTHER'

    assert vs.to_xml() == b'<Vertices BrainStructure="CIFTI_STRUCTURE_OTHER" />'

    assert len(vs) == 0
    vs.extend(np.array([0, 1, 2]))
    assert len(vs) == 3
    with pytest.raises(ValueError):
        vs[1] = 'a'
    with pytest.raises(ValueError):
        vs.insert(1, 'a')

    assert vs.to_xml() == b'<Vertices BrainStructure="CIFTI_STRUCTURE_OTHER">0 1 2</Vertices>'

    vs[0] = 10
    assert vs[0] == 10
    assert len(vs) == 3
    vs = ci.Cifti2Vertices(vertices=[0, 1, 2])
    assert len(vs) == 3


def test_cifti2_transformationmatrixvoxelindicesijktoxyz():
    tr = ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ()
    with pytest.raises(ci.Cifti2HeaderError):
        tr.to_xml()


def test_cifti2_surface():
    s = ci.Cifti2Surface()
    with pytest.raises(ci.Cifti2HeaderError):
        s.to_xml()


def test_cifti2_volume():
    vo = ci.Cifti2Volume()
    with pytest.raises(ci.Cifti2HeaderError):
        vo.to_xml()


def test_cifti2_vertexindices():
    vi = ci.Cifti2VertexIndices()
    assert len(vi) == 0
    with pytest.raises(ci.Cifti2HeaderError):
        vi.to_xml()
    vi.extend(np.array([0, 1, 2]))
    assert len(vi) == 3
    assert vi.to_xml() == b'<VertexIndices>0 1 2</VertexIndices>'

    with pytest.raises(ValueError):
        vi[0] = 'a'

    vi[0] = 10
    assert vi[0] == 10
    assert len(vi) == 3


def test_cifti2_voxelindicesijk():
    vi = ci.Cifti2VoxelIndicesIJK()
    with pytest.raises(ci.Cifti2HeaderError):
        vi.to_xml()

    vi = ci.Cifti2VoxelIndicesIJK()
    assert len(vi) == 0

    with pytest.raises(ci.Cifti2HeaderError):
        vi.to_xml()
    vi.extend(np.array([[0, 1, 2]]))

    assert len(vi) == 1
    assert vi[0] == [0, 1, 2]
    vi.append([3, 4, 5])
    assert len(vi) == 2
    vi.append([6, 7, 8])
    assert len(vi) == 3
    del vi[-1]
    assert len(vi) == 2

    assert vi[1] == [3, 4, 5]
    vi[1] = [3, 4, 6]
    assert vi[1] == [3, 4, 6]
    with pytest.raises(ValueError):
        vi['a'] = [1, 2, 3]

    with pytest.raises(TypeError):
        vi[[1, 2]] = [1, 2, 3]

    with pytest.raises(ValueError):
        vi[1] = [2, 3]

    assert vi[1, 1] == 4

    with pytest.raises(ValueError):
        vi[[1, 1]] = 'a'

    assert vi[0, 1:] == [1, 2]
    vi[0, 1] = 10
    assert vi[0, 1] == 10
    vi[0, 1] = 1

    # test for vi[:, 0] and other slices
    with pytest.raises(NotImplementedError):
        vi[:, 0]
    with pytest.raises(NotImplementedError):
        vi[:, 0] = 0
    with pytest.raises(NotImplementedError):
        # Don't know how to use remove with slice
        del vi[:, 0]
    with pytest.raises(ValueError):
        vi[0, 0, 0]

    with pytest.raises(ValueError):
        vi[0, 0, 0] = 0

    assert vi.to_xml().decode('utf-8') == '<VoxelIndicesIJK>0 1 2\n3 4 6</VoxelIndicesIJK>'

    with pytest.raises(TypeError):
        ci.Cifti2VoxelIndicesIJK([0, 1])

    vi = ci.Cifti2VoxelIndicesIJK([[1, 2, 3]])
    assert len(vi) == 1


def test_matrixindicesmap():
    mim = ci.Cifti2MatrixIndicesMap(0, 'CIFTI_INDEX_TYPE_LABELS')
    volume = ci.Cifti2Volume()
    volume2 = ci.Cifti2Volume()
    parcel = ci.Cifti2Parcel()

    assert mim.volume is None
    mim.extend((volume, parcel))

    assert mim.volume == volume
    with pytest.raises(ci.Cifti2HeaderError):
        mim.insert(0, volume)

    with pytest.raises(ci.Cifti2HeaderError):
        mim[1] = volume

    mim[0] = volume2
    assert mim.volume == volume2

    del mim.volume
    assert mim.volume is None
    with pytest.raises(ValueError):
        del mim.volume

    mim.volume = volume
    assert mim.volume == volume
    mim.volume = volume2
    assert mim.volume == volume2

    with pytest.raises(ValueError):
        mim.volume = parcel


def test_matrix():
    m = ci.Cifti2Matrix()

    with pytest.raises(ValueError):
        m.metadata = ci.Cifti2Parcel()

    with pytest.raises(TypeError):
        m[0] = ci.Cifti2Parcel()

    with pytest.raises(TypeError):
        m.insert(0, ci.Cifti2Parcel())

    mim_none = ci.Cifti2MatrixIndicesMap(None, 'CIFTI_INDEX_TYPE_LABELS')
    mim_0 = ci.Cifti2MatrixIndicesMap(0, 'CIFTI_INDEX_TYPE_LABELS')
    mim_1 = ci.Cifti2MatrixIndicesMap(1, 'CIFTI_INDEX_TYPE_LABELS')
    mim_01 = ci.Cifti2MatrixIndicesMap([0, 1], 'CIFTI_INDEX_TYPE_LABELS')

    with pytest.raises(ci.Cifti2HeaderError):
        m.insert(0, mim_none)

    assert m.mapped_indices == []

    h = ci.Cifti2Header(matrix=m)
    assert m.mapped_indices == []
    m.insert(0, mim_0)
    assert h.mapped_indices == [0]
    assert h.number_of_mapped_indices == 1
    with pytest.raises(ci.Cifti2HeaderError):
        m.insert(0, mim_0)

    with pytest.raises(ci.Cifti2HeaderError):
        m.insert(0, mim_01)

    m[0] = mim_1
    assert list(m.mapped_indices) == [1]
    m.insert(0, mim_0)
    assert sorted(m.mapped_indices) == [0, 1]
    assert h.number_of_mapped_indices == 2
    assert h.get_index_map(0) == mim_0
    assert h.get_index_map(1) == mim_1
    with pytest.raises(ci.Cifti2HeaderError):
        h.get_index_map(2)


def test_underscoring():
    # Pairs taken from inflection tests
    # https://github.com/jpvanhal/inflection/blob/663982e/test_inflection.py#L113-L125
    pairs = (
        ('Product', 'product'),
        ('SpecialGuest', 'special_guest'),
        ('ApplicationController', 'application_controller'),
        ('Area51Controller', 'area51_controller'),
        ('HTMLTidy', 'html_tidy'),
        ('HTMLTidyGenerator', 'html_tidy_generator'),
        ('FreeBSD', 'free_bsd'),
        ('HTML', 'html'),
    )

    for camel, underscored in pairs:
        assert ci.cifti2._underscore(camel) == underscored


class TestCifti2ImageAPI(_TDA, SerializeMixin, DtypeOverrideMixin):
    """Basic validation for Cifti2Image instances"""

    # A callable returning an image from ``image_maker(data, header)``
    image_maker = ci.Cifti2Image
    # A callable returning a header from ``header_maker()``
    header_maker = ci.Cifti2Header
    # A callable returning a nifti header
    ni_header_maker = Nifti2Header
    example_shapes = ((2,), (2, 3), (2, 3, 4))
    standard_extension = '.nii'
    storable_dtypes = (
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
        np.float32,
        np.float64,
    )

    def make_imaker(self, arr, header=None, ni_header=None):
        for idx, sz in enumerate(arr.shape):
            maps = [ci.Cifti2NamedMap(str(value)) for value in range(sz)]
            mim = ci.Cifti2MatrixIndicesMap((idx,), 'CIFTI_INDEX_TYPE_SCALARS', maps=maps)
            header.matrix.append(mim)
        return lambda: self.image_maker(arr.copy(), header, ni_header)
