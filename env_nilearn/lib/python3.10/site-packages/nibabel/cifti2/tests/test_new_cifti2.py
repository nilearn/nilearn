"""Tests the generation of new CIFTI-2 files from scratch

Contains a series of functions to create and check each of the 5 CIFTI index
types (i.e. BRAIN_MODELS, PARCELS, SCALARS, LABELS, and SERIES).

These functions are used in the tests to generate most CIFTI file types from
scratch.
"""

import numpy as np
import pytest

import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.tmpdirs import InTemporaryDirectory

from ...testing import (
    assert_array_equal,
    clear_and_catch_warnings,
    error_warnings,
    suppress_warnings,
)

affine = [
    [-1.5, 0, 0, 90],
    [0, 1.5, 0, -85],
    [0, 0, 1.5, -71],
    [0, 0, 0, 1.0],
]

dimensions = (120, 83, 78)

number_of_vertices = 30000

brain_models = [
    (
        'CIFTI_STRUCTURE_THALAMUS_LEFT',
        [
            [60, 60, 60],
            [61, 59, 60],
            [61, 60, 59],
            [80, 90, 92],
        ],
    ),
    ('CIFTI_STRUCTURE_CORTEX_LEFT', [0, 1000, 1301, 19972, 27312]),
    ('CIFTI_STRUCTURE_CORTEX_RIGHT', [207]),
]


def create_geometry_map(applies_to_matrix_dimension):
    voxels = ci.Cifti2VoxelIndicesIJK(brain_models[0][1])
    left_thalamus = ci.Cifti2BrainModel(
        index_offset=0,
        index_count=4,
        model_type='CIFTI_MODEL_TYPE_VOXELS',
        brain_structure=brain_models[0][0],
        voxel_indices_ijk=voxels,
    )

    vertices = ci.Cifti2VertexIndices(np.array(brain_models[1][1]))
    left_cortex = ci.Cifti2BrainModel(
        index_offset=4,
        index_count=5,
        model_type='CIFTI_MODEL_TYPE_SURFACE',
        brain_structure=brain_models[1][0],
        vertex_indices=vertices,
    )
    left_cortex.surface_number_of_vertices = number_of_vertices

    vertices = ci.Cifti2VertexIndices(np.array(brain_models[2][1]))
    right_cortex = ci.Cifti2BrainModel(
        index_offset=9,
        index_count=1,
        model_type='CIFTI_MODEL_TYPE_SURFACE',
        brain_structure=brain_models[2][0],
        vertex_indices=vertices,
    )
    right_cortex.surface_number_of_vertices = number_of_vertices

    volume = ci.Cifti2Volume(
        dimensions, ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, affine)
    )
    return ci.Cifti2MatrixIndicesMap(
        applies_to_matrix_dimension,
        'CIFTI_INDEX_TYPE_BRAIN_MODELS',
        maps=[left_thalamus, left_cortex, right_cortex, volume],
    )


def check_geometry_map(mapping):
    assert mapping.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_BRAIN_MODELS'
    assert len(list(mapping.brain_models)) == 3
    left_thalamus, left_cortex, right_cortex = mapping.brain_models

    assert left_thalamus.index_offset == 0
    assert left_thalamus.index_count == 4
    assert left_thalamus.model_type == 'CIFTI_MODEL_TYPE_VOXELS'
    assert left_thalamus.brain_structure == brain_models[0][0]
    assert left_thalamus.vertex_indices is None
    assert left_thalamus.surface_number_of_vertices is None
    assert left_thalamus.voxel_indices_ijk._indices == brain_models[0][1]

    assert left_cortex.index_offset == 4
    assert left_cortex.index_count == 5
    assert left_cortex.model_type == 'CIFTI_MODEL_TYPE_SURFACE'
    assert left_cortex.brain_structure == brain_models[1][0]
    assert left_cortex.voxel_indices_ijk is None
    assert left_cortex.vertex_indices._indices == brain_models[1][1]
    assert left_cortex.surface_number_of_vertices == number_of_vertices

    assert right_cortex.index_offset == 9
    assert right_cortex.index_count == 1
    assert right_cortex.model_type == 'CIFTI_MODEL_TYPE_SURFACE'
    assert right_cortex.brain_structure == brain_models[2][0]
    assert right_cortex.voxel_indices_ijk is None
    assert right_cortex.vertex_indices._indices == brain_models[2][1]
    assert right_cortex.surface_number_of_vertices == number_of_vertices

    assert mapping.volume.volume_dimensions == dimensions
    assert (mapping.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix == affine).all()


parcels = [
    (
        'volume_parcel',
        (
            [
                [60, 60, 60],
                [61, 59, 60],
                [61, 60, 59],
                [80, 90, 92],
            ],
        ),
    ),
    (
        'surface_parcel',
        (
            ('CIFTI_STRUCTURE_CORTEX_LEFT', [0, 1000, 1301, 19972, 27312]),
            ('CIFTI_STRUCTURE_CORTEX_RIGHT', [0, 100, 381]),
        ),
    ),
    (
        'mixed_parcel',
        (
            [
                [71, 81, 39],
                [53, 21, 91],
            ],
            ('CIFTI_STRUCTURE_CORTEX_LEFT', [71, 88, 999]),
        ),
    ),
    ('single_element', ([[71, 81, 39]], ('CIFTI_STRUCTURE_CORTEX_LEFT', [40]))),
]


def create_parcel_map(applies_to_matrix_dimension):
    mapping = ci.Cifti2MatrixIndicesMap(applies_to_matrix_dimension, 'CIFTI_INDEX_TYPE_PARCELS')
    for name, elements in parcels:
        surfaces = []
        volume = None
        for element in elements:
            if isinstance(element[0], str):
                surfaces.append(ci.Cifti2Vertices(element[0], element[1]))
            else:
                volume = ci.Cifti2VoxelIndicesIJK(element)
        mapping.append(ci.Cifti2Parcel(name, volume, surfaces))

    mapping.extend(
        [
            ci.Cifti2Surface(f'CIFTI_STRUCTURE_CORTEX_{orientation}', number_of_vertices)
            for orientation in ['LEFT', 'RIGHT']
        ]
    )
    mapping.volume = ci.Cifti2Volume(
        dimensions, ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ(-3, affine)
    )
    return mapping


def check_parcel_map(mapping):
    assert mapping.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_PARCELS'
    assert len(list(mapping.parcels)) == len(parcels)
    for (name, elements), parcel in zip(parcels, mapping.parcels):
        assert parcel.name == name
        idx_surface = 0
        for element in elements:
            if isinstance(element[0], str):
                surface = parcel.vertices[idx_surface]
                assert surface.brain_structure == element[0]
                assert surface._vertices == element[1]
                idx_surface += 1
            else:
                assert parcel.voxel_indices_ijk._indices == element

    for surface, orientation in zip(mapping.surfaces, ('LEFT', 'RIGHT')):
        assert surface.brain_structure == f'CIFTI_STRUCTURE_CORTEX_{orientation}'
        assert surface.surface_number_of_vertices == number_of_vertices

    assert mapping.volume.volume_dimensions == dimensions
    assert (mapping.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix == affine).all()


scalars = [('first_name', {'meta_key': 'some_metadata'}), ('another name', {})]


def create_scalar_map(applies_to_matrix_dimension):
    maps = [ci.Cifti2NamedMap(name, ci.Cifti2MetaData(meta)) for name, meta in scalars]
    return ci.Cifti2MatrixIndicesMap(
        applies_to_matrix_dimension, 'CIFTI_INDEX_TYPE_SCALARS', maps=maps
    )


def check_scalar_map(mapping):
    assert mapping.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_SCALARS'
    assert len(list(mapping.named_maps)) == 2

    for expected, named_map in zip(scalars, mapping.named_maps):
        assert named_map.map_name == expected[0]
        if len(expected[1]) == 0:
            assert named_map.metadata is None
        else:
            assert named_map.metadata == expected[1]


labels = [
    (
        'first_name',
        {'meta_key': 'some_metadata'},
        {
            0: ('label0', (0.1, 0.3, 0.2, 0.5)),
            1: ('new_label', (0.5, 0.3, 0.1, 0.4)),
        },
    ),
    (
        'another name',
        {},
        {
            0: ('???', (0, 0, 0, 0)),
            1: ('great region', (0.4, 0.1, 0.23, 0.15)),
        },
    ),
]


def create_label_map(applies_to_matrix_dimension):
    maps = []
    for name, meta, label in labels:
        label_table = ci.Cifti2LabelTable()
        for key, (tag, rgba) in label.items():
            label_table[key] = ci.Cifti2Label(key, tag, *rgba)
        maps.append(ci.Cifti2NamedMap(name, ci.Cifti2MetaData(meta), label_table))
    return ci.Cifti2MatrixIndicesMap(
        applies_to_matrix_dimension, 'CIFTI_INDEX_TYPE_LABELS', maps=maps
    )


def check_label_map(mapping):
    assert mapping.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_LABELS'
    assert len(list(mapping.named_maps)) == 2

    for expected, named_map in zip(scalars, mapping.named_maps):
        assert named_map.map_name == expected[0]
        if len(expected[1]) == 0:
            assert named_map.metadata is None
        else:
            assert named_map.metadata == expected[1]


def create_series_map(applies_to_matrix_dimension):
    return ci.Cifti2MatrixIndicesMap(
        applies_to_matrix_dimension,
        'CIFTI_INDEX_TYPE_SERIES',
        number_of_series_points=13,
        series_exponent=-3,
        series_start=18.2,
        series_step=10.5,
        series_unit='SECOND',
    )


def check_series_map(mapping):
    assert mapping.indices_map_to_data_type == 'CIFTI_INDEX_TYPE_SERIES'
    assert mapping.number_of_series_points == 13
    assert mapping.series_exponent == -3
    assert mapping.series_start == 18.2
    assert mapping.series_step == 10.5
    assert mapping.series_unit == 'SECOND'


def test_dtseries():
    series_map = create_series_map((0,))
    geometry_map = create_geometry_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((series_map, geometry_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(13, 10)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SERIES')

    with InTemporaryDirectory():
        ci.save(img, 'test.dtseries.nii')
        img2 = nib.load('test.dtseries.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnDenseSeries'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_series_map(img2.header.matrix.get_index_map(0))
        check_geometry_map(img2.header.matrix.get_index_map(1))
        del img2


def test_dscalar():
    scalar_map = create_scalar_map((0,))
    geometry_map = create_geometry_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((scalar_map, geometry_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 10)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_SCALARS')

    with InTemporaryDirectory():
        ci.save(img, 'test.dscalar.nii')
        img2 = nib.load('test.dscalar.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnDenseScalar'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_scalar_map(img2.header.matrix.get_index_map(0))
        check_geometry_map(img2.header.matrix.get_index_map(1))
        del img2


def test_dlabel():
    label_map = create_label_map((0,))
    geometry_map = create_geometry_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((label_map, geometry_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 10)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_LABELS')

    with InTemporaryDirectory():
        ci.save(img, 'test.dlabel.nii')
        img2 = nib.load('test.dlabel.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnDenseLabel'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_label_map(img2.header.matrix.get_index_map(0))
        check_geometry_map(img2.header.matrix.get_index_map(1))
        del img2


def test_dconn():
    mapping = create_geometry_map((0, 1))
    matrix = ci.Cifti2Matrix()
    matrix.append(mapping)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(10, 10)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE')

    with InTemporaryDirectory():
        ci.save(img, 'test.dconn.nii')
        img2 = nib.load('test.dconn.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnDense'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        assert img2.header.matrix.get_index_map(0) == img2.header.matrix.get_index_map(1)
        check_geometry_map(img2.header.matrix.get_index_map(0))
        del img2


def test_ptseries():
    series_map = create_series_map((0,))
    parcel_map = create_parcel_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((series_map, parcel_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(13, 4)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SERIES')

    with InTemporaryDirectory():
        ci.save(img, 'test.ptseries.nii')
        img2 = nib.load('test.ptseries.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnParcelSries'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_series_map(img2.header.matrix.get_index_map(0))
        check_parcel_map(img2.header.matrix.get_index_map(1))
        del img2


def test_pscalar():
    scalar_map = create_scalar_map((0,))
    parcel_map = create_parcel_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((scalar_map, parcel_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 4)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_SCALAR')

    with InTemporaryDirectory():
        ci.save(img, 'test.pscalar.nii')
        img2 = nib.load('test.pscalar.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnParcelScalr'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_scalar_map(img2.header.matrix.get_index_map(0))
        check_parcel_map(img2.header.matrix.get_index_map(1))
        del img2


def test_pdconn():
    geometry_map = create_geometry_map((0,))
    parcel_map = create_parcel_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((geometry_map, parcel_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(10, 4)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_DENSE')

    with InTemporaryDirectory():
        ci.save(img, 'test.pdconn.nii')
        img2 = ci.load('test.pdconn.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnParcelDense'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_geometry_map(img2.header.matrix.get_index_map(0))
        check_parcel_map(img2.header.matrix.get_index_map(1))
        del img2


def test_dpconn():
    parcel_map = create_parcel_map((0,))
    geometry_map = create_geometry_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((parcel_map, geometry_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(4, 10)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_DENSE_PARCELLATED')

    with InTemporaryDirectory():
        ci.save(img, 'test.dpconn.nii')
        img2 = ci.load('test.dpconn.nii')
        assert img2.nifti_header.get_intent()[0] == 'ConnDenseParcel'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_parcel_map(img2.header.matrix.get_index_map(0))
        check_geometry_map(img2.header.matrix.get_index_map(1))
        del img2


def test_plabel():
    label_map = create_label_map((0,))
    parcel_map = create_parcel_map((1,))
    matrix = ci.Cifti2Matrix()
    matrix.extend((label_map, parcel_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(2, 4)
    img = ci.Cifti2Image(data, hdr)

    with InTemporaryDirectory():
        ci.save(img, 'test.plabel.nii')
        img2 = ci.load('test.plabel.nii')
        assert img.nifti_header.get_intent()[0] == 'ConnUnknown'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        check_label_map(img2.header.matrix.get_index_map(0))
        check_parcel_map(img2.header.matrix.get_index_map(1))
        del img2


def test_pconn():
    mapping = create_parcel_map((0, 1))
    matrix = ci.Cifti2Matrix()
    matrix.append(mapping)
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(4, 4)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED')

    with InTemporaryDirectory():
        ci.save(img, 'test.pconn.nii')
        img2 = ci.load('test.pconn.nii')
        assert img.nifti_header.get_intent()[0] == 'ConnParcels'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        assert img2.header.matrix.get_index_map(0) == img2.header.matrix.get_index_map(1)
        check_parcel_map(img2.header.matrix.get_index_map(0))
        del img2


def test_pconnseries():
    parcel_map = create_parcel_map((0, 1))
    series_map = create_series_map((2,))

    matrix = ci.Cifti2Matrix()
    matrix.extend((parcel_map, series_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(4, 4, 13)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_PARCELLATED_SERIES')

    with InTemporaryDirectory():
        ci.save(img, 'test.pconnseries.nii')
        img2 = ci.load('test.pconnseries.nii')
        assert img.nifti_header.get_intent()[0] == 'ConnPPSr'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        assert img2.header.matrix.get_index_map(0) == img2.header.matrix.get_index_map(1)
        check_parcel_map(img2.header.matrix.get_index_map(0))
        check_series_map(img2.header.matrix.get_index_map(2))
        del img2


def test_pconnscalar():
    parcel_map = create_parcel_map((0, 1))
    scalar_map = create_scalar_map((2,))

    matrix = ci.Cifti2Matrix()
    matrix.extend((parcel_map, scalar_map))
    hdr = ci.Cifti2Header(matrix)
    data = np.random.randn(4, 4, 2)
    img = ci.Cifti2Image(data, hdr)
    img.nifti_header.set_intent('NIFTI_INTENT_CONNECTIVITY_PARCELLATED_PARCELLATED_SCALAR')

    with InTemporaryDirectory():
        ci.save(img, 'test.pconnscalar.nii')
        img2 = ci.load('test.pconnscalar.nii')
        assert img.nifti_header.get_intent()[0] == 'ConnPPSc'
        assert isinstance(img2, ci.Cifti2Image)
        assert_array_equal(img2.get_fdata(), data)
        assert img2.header.matrix.get_index_map(0) == img2.header.matrix.get_index_map(1)

        check_parcel_map(img2.header.matrix.get_index_map(0))
        check_scalar_map(img2.header.matrix.get_index_map(2))
        del img2


def test_wrong_shape():
    scalar_map = create_scalar_map((0,))
    brain_model_map = create_geometry_map((1,))

    matrix = ci.Cifti2Matrix()
    matrix.extend((scalar_map, brain_model_map))
    hdr = ci.Cifti2Header(matrix)

    # correct shape is (2, 10)
    for data in (
        np.random.randn(1, 11),
        np.random.randn(2, 10, 1),
        np.random.randn(1, 2, 10),
        np.random.randn(3, 10),
        np.random.randn(2, 9),
    ):
        with clear_and_catch_warnings():
            with error_warnings():
                with pytest.raises(UserWarning):
                    ci.Cifti2Image(data, hdr)
        with suppress_warnings():
            img = ci.Cifti2Image(data, hdr)

        with pytest.raises(ValueError):
            img.to_file_map()
