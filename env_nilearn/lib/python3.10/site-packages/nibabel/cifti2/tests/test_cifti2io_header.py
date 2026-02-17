# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##

from os.path import dirname
from os.path import join as pjoin

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from packaging.version import Version

import nibabel as nib
from nibabel import cifti2 as ci
from nibabel.cifti2.parse_cifti2 import _Cifti2AsNiftiHeader
from nibabel.tests import test_nifti2 as tn2
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data
from nibabel.tmpdirs import InTemporaryDirectory

NIBABEL_TEST_DATA = pjoin(dirname(nib.__file__), 'tests', 'data')
NIFTI2_DATA = pjoin(NIBABEL_TEST_DATA, 'example_nifti2.nii.gz')

CIFTI2_DATA = pjoin(get_nibabel_data(), 'nitest-cifti2')

DATA_FILE1 = pjoin(CIFTI2_DATA, '')
DATA_FILE2 = pjoin(CIFTI2_DATA, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.dscalar.nii')
DATA_FILE3 = pjoin(CIFTI2_DATA, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.dtseries.nii')
DATA_FILE4 = pjoin(CIFTI2_DATA, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.ptseries.nii')
DATA_FILE5 = pjoin(CIFTI2_DATA, 'Conte69.parcellations_VGD11b.32k_fs_LR.dlabel.nii')
DATA_FILE6 = pjoin(CIFTI2_DATA, 'ones.dscalar.nii')
datafiles = [DATA_FILE2, DATA_FILE3, DATA_FILE4, DATA_FILE5, DATA_FILE6]


def test_space_separated_affine():
    ci.Cifti2Image.from_filename(pjoin(NIBABEL_TEST_DATA, 'row_major.dconn.nii'))


def test_read_nifti2():
    # Error trying to read a CIFTI-2 image from a NIfTI2-only image.
    filemap = ci.Cifti2Image.make_file_map()
    for k in filemap:
        filemap[k].fileobj = open(NIFTI2_DATA)
    with pytest.raises(ValueError):
        ci.Cifti2Image.from_file_map(filemap)


@needs_nibabel_data('nitest-cifti2')
def test_read_internal():
    img2 = ci.load(DATA_FILE6)
    assert isinstance(img2.header, ci.Cifti2Header)
    assert img2.shape == (1, 91282)


@needs_nibabel_data('nitest-cifti2')
def test_read_and_proxies():
    img2 = nib.load(DATA_FILE6)
    assert isinstance(img2.header, ci.Cifti2Header)
    assert img2.shape == (1, 91282)
    # While we cannot reshape arrayproxies, all images are in-memory
    assert not img2.in_memory
    data = img2.get_fdata()
    assert data is not img2.dataobj
    # Uncaching has no effect, images are always array images
    img2.uncache()
    assert data is not img2.get_fdata()


@needs_nibabel_data('nitest-cifti2')
def test_version():
    for dat in datafiles:
        img = nib.load(dat)
        assert Version(img.header.version) == Version('2')


@needs_nibabel_data('nitest-cifti2')
def test_readwritedata():
    with InTemporaryDirectory():
        for name in datafiles:
            img = ci.load(name)
            ci.save(img, 'test.nii')
            img2 = ci.load('test.nii')
            assert len(img.header.matrix) == len(img2.header.matrix)
            # Order should be preserved in load/save
            for mim1, mim2 in zip(img.header.matrix, img2.header.matrix):
                named_maps1 = [m_ for m_ in mim1 if isinstance(m_, ci.Cifti2NamedMap)]
                named_maps2 = [m_ for m_ in mim2 if isinstance(m_, ci.Cifti2NamedMap)]
                assert len(named_maps1) == len(named_maps2)
                for map1, map2 in zip(named_maps1, named_maps2):
                    assert map1.map_name == map2.map_name
                    if map1.label_table is None:
                        assert map2.label_table is None
                    else:
                        assert len(map1.label_table) == len(map2.label_table)

            assert_array_almost_equal(img.dataobj, img2.dataobj)


@needs_nibabel_data('nitest-cifti2')
def test_nibabel_readwritedata():
    with InTemporaryDirectory():
        for name in datafiles:
            img = nib.load(name)
            nib.save(img, 'test.nii')
            img2 = nib.load('test.nii')
            assert len(img.header.matrix) == len(img2.header.matrix)
            # Order should be preserved in load/save
            for mim1, mim2 in zip(img.header.matrix, img2.header.matrix):
                named_maps1 = [m_ for m_ in mim1 if isinstance(m_, ci.Cifti2NamedMap)]
                named_maps2 = [m_ for m_ in mim2 if isinstance(m_, ci.Cifti2NamedMap)]
                assert len(named_maps1) == len(named_maps2)
                for map1, map2 in zip(named_maps1, named_maps2):
                    assert map1.map_name == map2.map_name
                    if map1.label_table is None:
                        assert map2.label_table is None
                    else:
                        assert len(map1.label_table) == len(map2.label_table)
            assert_array_almost_equal(img.dataobj, img2.dataobj)


@needs_nibabel_data('nitest-cifti2')
def test_cifti2types():
    """Check that we instantiate Cifti2 classes correctly, and that our
    test files exercise all classes"""
    counter = {
        ci.Cifti2LabelTable: 0,
        ci.Cifti2Label: 0,
        ci.Cifti2NamedMap: 0,
        ci.Cifti2Surface: 0,
        ci.Cifti2VoxelIndicesIJK: 0,
        ci.Cifti2Vertices: 0,
        ci.Cifti2Parcel: 0,
        ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ: 0,
        ci.Cifti2Volume: 0,
        ci.Cifti2VertexIndices: 0,
        ci.Cifti2BrainModel: 0,
        ci.Cifti2MatrixIndicesMap: 0,
    }

    for name in datafiles:
        hdr = ci.load(name).header
        # Matrix and MetaData aren't conditional, so don't bother counting
        assert isinstance(hdr.matrix, ci.Cifti2Matrix)
        assert isinstance(hdr.matrix.metadata, ci.Cifti2MetaData)
        for mim in hdr.matrix:
            assert isinstance(mim, ci.Cifti2MatrixIndicesMap)
            counter[ci.Cifti2MatrixIndicesMap] += 1
            for map_ in mim:
                print(map_)
                if isinstance(map_, ci.Cifti2BrainModel):
                    counter[ci.Cifti2BrainModel] += 1
                    if isinstance(map_.vertex_indices, ci.Cifti2VertexIndices):
                        counter[ci.Cifti2VertexIndices] += 1
                    if isinstance(map_.voxel_indices_ijk, ci.Cifti2VoxelIndicesIJK):
                        counter[ci.Cifti2VoxelIndicesIJK] += 1
                elif isinstance(map_, ci.Cifti2NamedMap):
                    counter[ci.Cifti2NamedMap] += 1
                    assert isinstance(map_.metadata, ci.Cifti2MetaData)
                    if isinstance(map_.label_table, ci.Cifti2LabelTable):
                        counter[ci.Cifti2LabelTable] += 1
                        for label in map_.label_table:
                            assert isinstance(map_.label_table[label], ci.Cifti2Label)
                            counter[ci.Cifti2Label] += 1
                elif isinstance(map_, ci.Cifti2Parcel):
                    counter[ci.Cifti2Parcel] += 1
                    if isinstance(map_.voxel_indices_ijk, ci.Cifti2VoxelIndicesIJK):
                        counter[ci.Cifti2VoxelIndicesIJK] += 1
                    assert isinstance(map_.vertices, list)
                    for vtcs in map_.vertices:
                        assert isinstance(vtcs, ci.Cifti2Vertices)
                        counter[ci.Cifti2Vertices] += 1
                elif isinstance(map_, ci.Cifti2Surface):
                    counter[ci.Cifti2Surface] += 1
                elif isinstance(map_, ci.Cifti2Volume):
                    counter[ci.Cifti2Volume] += 1
                    if isinstance(
                        map_.transformation_matrix_voxel_indices_ijk_to_xyz,
                        ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ,
                    ):
                        counter[ci.Cifti2TransformationMatrixVoxelIndicesIJKtoXYZ] += 1

            assert list(mim.named_maps) == [m_ for m_ in mim if isinstance(m_, ci.Cifti2NamedMap)]
            assert list(mim.surfaces) == [m_ for m_ in mim if isinstance(m_, ci.Cifti2Surface)]
            assert list(mim.parcels) == [m_ for m_ in mim if isinstance(m_, ci.Cifti2Parcel)]
            assert list(mim.brain_models) == [
                m_ for m_ in mim if isinstance(m_, ci.Cifti2BrainModel)
            ]
            assert ([mim.volume] if mim.volume else []) == [
                m_ for m_ in mim if isinstance(m_, ci.Cifti2Volume)
            ]

    for klass, count in counter.items():
        assert count > 0, 'No exercise of ' + klass.__name__


@needs_nibabel_data('nitest-cifti2')
def test_read_geometry():
    img = ci.Cifti2Image.from_filename(DATA_FILE6)
    geometry_mapping = img.header.matrix.get_index_map(1)

    # For every brain model in ones.dscalar.nii defines:
    # brain structure name, number of grayordinates, first vertex or voxel, last vertex or voxel
    expected_geometry = [
        ('CIFTI_STRUCTURE_CORTEX_LEFT', 29696, 0, 32491),
        ('CIFTI_STRUCTURE_CORTEX_RIGHT', 29716, 0, 32491),
        ('CIFTI_STRUCTURE_ACCUMBENS_LEFT', 135, [49, 66, 28], [48, 72, 35]),
        ('CIFTI_STRUCTURE_ACCUMBENS_RIGHT', 140, [40, 66, 29], [43, 66, 36]),
        ('CIFTI_STRUCTURE_AMYGDALA_LEFT', 315, [55, 61, 21], [56, 58, 31]),
        ('CIFTI_STRUCTURE_AMYGDALA_RIGHT', 332, [34, 62, 20], [36, 61, 31]),
        ('CIFTI_STRUCTURE_BRAIN_STEM', 3472, [42, 41, 0], [46, 50, 36]),
        ('CIFTI_STRUCTURE_CAUDATE_LEFT', 728, [50, 72, 32], [53, 60, 49]),
        ('CIFTI_STRUCTURE_CAUDATE_RIGHT', 755, [40, 68, 33], [37, 62, 49]),
        ('CIFTI_STRUCTURE_CEREBELLUM_LEFT', 8709, [49, 35, 4], [46, 37, 37]),
        ('CIFTI_STRUCTURE_CEREBELLUM_RIGHT', 9144, [38, 35, 4], [44, 38, 36]),
        ('CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_LEFT', 706, [52, 53, 26], [56, 49, 35]),
        ('CIFTI_STRUCTURE_DIENCEPHALON_VENTRAL_RIGHT', 712, [39, 54, 26], [35, 49, 36]),
        ('CIFTI_STRUCTURE_HIPPOCAMPUS_LEFT', 764, [55, 60, 21], [54, 44, 39]),
        ('CIFTI_STRUCTURE_HIPPOCAMPUS_RIGHT', 795, [33, 60, 21], [38, 45, 39]),
        ('CIFTI_STRUCTURE_PALLIDUM_LEFT', 297, [56, 59, 32], [55, 61, 39]),
        ('CIFTI_STRUCTURE_PALLIDUM_RIGHT', 260, [36, 62, 32], [35, 62, 39]),
        ('CIFTI_STRUCTURE_PUTAMEN_LEFT', 1060, [51, 66, 28], [58, 64, 43]),
        ('CIFTI_STRUCTURE_PUTAMEN_RIGHT', 1010, [34, 66, 29], [31, 62, 43]),
        ('CIFTI_STRUCTURE_THALAMUS_LEFT', 1288, [55, 47, 33], [52, 53, 46]),
        ('CIFTI_STRUCTURE_THALAMUS_RIGHT', 1248, [32, 47, 34], [38, 55, 46]),
    ]
    current_index = 0
    for from_file, expected in zip(geometry_mapping.brain_models, expected_geometry):
        assert from_file.model_type in ('CIFTI_MODEL_TYPE_SURFACE', 'CIFTI_MODEL_TYPE_VOXELS')
        assert from_file.brain_structure == expected[0]
        assert from_file.index_offset == current_index
        assert from_file.index_count == expected[1]
        current_index += from_file.index_count

        if from_file.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
            assert from_file.voxel_indices_ijk is None
            assert len(from_file.vertex_indices) == expected[1]
            assert from_file.vertex_indices[0] == expected[2]
            assert from_file.vertex_indices[-1] == expected[3]
            assert from_file.surface_number_of_vertices == 32492
        else:
            assert from_file.vertex_indices is None
            assert from_file.surface_number_of_vertices is None
            assert len(from_file.voxel_indices_ijk) == expected[1]
            assert from_file.voxel_indices_ijk[0] == expected[2]
            assert from_file.voxel_indices_ijk[-1] == expected[3]
    assert current_index == img.shape[1]

    expected_affine = [
        [-2, 0, 0, 90],
        [0, 2, 0, -126],
        [0, 0, 2, -72],
        [0, 0, 0, 1],
    ]
    expected_dimensions = (91, 109, 91)
    assert np.array_equal(
        geometry_mapping.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix,
        expected_affine,
    )
    assert geometry_mapping.volume.volume_dimensions == expected_dimensions


@needs_nibabel_data('nitest-cifti2')
def test_read_parcels():
    img = ci.Cifti2Image.from_filename(DATA_FILE4)
    parcel_mapping = img.header.matrix.get_index_map(1)

    expected_parcels = [
        ('MEDIAL.WALL', ((719, 20, 28550), (810, 21, 28631))),
        ('BA2_FRB08', ((516, 6757, 17888), (461, 6757, 17887))),
        ('BA1_FRB08', ((211, 5029, 17974), (214, 3433, 17934))),
        ('BA3b_FRB08', ((444, 3436, 18065), (397, 3436, 18065))),
        ('BA4p_FRB08', ((344, 3445, 18164), (371, 3443, 18175))),
        ('BA3a_FRB08', ((290, 3441, 18140), (289, 3440, 18140))),
        ('BA4a_FRB08', ((471, 3446, 18181), (455, 3446, 19759))),
        ('BA6_FRB08', ((1457, 2, 30951), (1400, 2, 30951))),
        ('BA17_V1_FRB08', ((629, 23155, 25785), (635, 23155, 25759))),
        ('BA45_FRB08', ((245, 10100, 18774), (214, 10103, 18907))),
        ('BA44_FRB08', ((226, 10118, 19240), (273, 10119, 19270))),
        ('hOc5_MT_FRB08', ((104, 15019, 23329), (80, 15023, 23376))),
        ('BA18_V2_FRB08', ((702, 95, 25902), (651, 98, 25903))),
        ('V3A_SHM07', ((82, 4, 25050), (82, 4, 25050))),
        ('V3B_SHM07', ((121, 13398, 23303), (121, 13398, 23303))),
        ('LO1_KPO10', ((54, 15007, 23543), (54, 15007, 23543))),
        ('LO2_KPO10', ((79, 15013, 23636), (79, 15013, 23636))),
        ('PITd_KPO10', ((53, 15018, 23769), (65, 15018, 23769))),
        ('PITv_KPO10', ((72, 23480, 23974), (72, 23480, 23974))),
        ('OP1_BSW08', ((470, 8421, 18790), (470, 8421, 18790))),
        ('OP2_BSW08', ((67, 10, 31060), (67, 10, 31060))),
        ('OP3_BSW08', ((119, 10137, 18652), (119, 10137, 18652))),
        ('OP4_BSW08', ((191, 16613, 19429), (192, 16613, 19429))),
        ('IPS1_SHM07', ((54, 11775, 14496), (54, 11775, 14496))),
        ('IPS2_SHM07', ((71, 11771, 14587), (71, 11771, 14587))),
        ('IPS3_SHM07', ((114, 11764, 14783), (114, 11764, 14783))),
        ('IPS4_SHM07', ((101, 11891, 12653), (101, 11891, 12653))),
        ('V7_SHM07', ((140, 11779, 14002), (140, 11779, 14002))),
        ('V4v_SHM07', ((81, 23815, 24557), (90, 23815, 24557))),
        ('V3d_KPO10', ((90, 23143, 25192), (115, 23143, 25192))),
        ('14c_OFP03', ((22, 19851, 21311), (22, 19851, 21311))),
        ('13a_OFP03', ((20, 20963, 21154), (20, 20963, 21154))),
        ('47s_OFP03', ((211, 10182, 20343), (211, 10182, 20343))),
        ('14r_OFP03', ((54, 21187, 21324), (54, 21187, 21324))),
        ('13m_OFP03', ((103, 20721, 21075), (103, 20721, 21075))),
        ('13l_OFP03', ((101, 20466, 20789), (101, 20466, 20789))),
        ('32pl_OFP03', ((14, 19847, 21409), (14, 19847, 21409))),
        ('25_OFP03', ((8, 19844, 27750), (8, 19844, 27750))),
        ('47m_OFP03', ((200, 10174, 20522), (200, 10174, 20522))),
        ('47l_OFP03', ((142, 10164, 19969), (160, 10164, 19969))),
        ('Iai_OFP03', ((153, 10188, 20199), (153, 10188, 20199))),
        ('10r_OFP03', ((138, 19811, 28267), (138, 19811, 28267))),
        ('11m_OFP03', ((92, 20850, 21165), (92, 20850, 21165))),
        ('11l_OFP03', ((200, 20275, 21029), (200, 20275, 21029))),
        ('47r_OFP03', ((259, 10094, 20535), (259, 10094, 20535))),
        ('10m_OFP03', ((102, 19825, 21411), (102, 19825, 21411))),
        ('Iam_OFP03', ((15, 20346, 20608), (15, 20346, 20608))),
        ('Ial_OFP03', ((89, 10194, 11128), (89, 10194, 11128))),
        ('24_OFP03', ((39, 19830, 28279), (36, 19830, 28279))),
        ('Iapm_OFP03', ((7, 20200, 20299), (7, 20200, 20299))),
        ('10p_OFP03', ((480, 19780, 28640), (480, 19780, 28640))),
        ('V6_PHG06', ((72, 12233, 12869), (72, 12233, 12869))),
        ('ER_FRB08', ((103, 21514, 26470), (103, 21514, 26470))),
        ('13b_OFP03', ((60, 21042, 21194), (71, 21040, 21216))),
    ]

    assert img.shape[1] == len(expected_parcels)
    assert len(list(parcel_mapping.parcels)) == len(expected_parcels)

    for (name, expected_surfaces), parcel in zip(expected_parcels, parcel_mapping.parcels):
        assert parcel.name == name
        assert len(parcel.vertices) == 2
        for vertices, orientation, (length, first_element, last_element) in zip(
            parcel.vertices, ('LEFT', 'RIGHT'), expected_surfaces
        ):
            assert len(vertices) == length
            assert vertices[0] == first_element
            assert vertices[-1] == last_element
            assert vertices.brain_structure == f'CIFTI_STRUCTURE_CORTEX_{orientation}'


@needs_nibabel_data('nitest-cifti2')
def test_read_scalar():
    img = ci.Cifti2Image.from_filename(DATA_FILE2)
    scalar_mapping = img.header.matrix.get_index_map(0)

    expected_names = ('MyelinMap_BC_decurv', 'corrThickness')
    assert img.shape[0] == len(expected_names)
    assert len(list(scalar_mapping.named_maps)) == len(expected_names)

    expected_meta = [('PaletteColorMapping', '<PaletteColorMapping Version="1">\n   <ScaleMo')]
    for scalar, name in zip(scalar_mapping.named_maps, expected_names):
        assert scalar.map_name == name

        assert len(scalar.metadata) == len(expected_meta)
        print(expected_meta[0], scalar.metadata.data.keys())
        for key, value in expected_meta:
            assert key in scalar.metadata.data.keys()
            assert scalar.metadata[key][: len(value)] == value

        assert scalar.label_table is None, '.dscalar file should not define a label table'


@needs_nibabel_data('nitest-cifti2')
def test_read_series():
    img = ci.Cifti2Image.from_filename(DATA_FILE4)
    series_mapping = img.header.matrix.get_index_map(0)
    assert series_mapping.series_start == 0.0
    assert series_mapping.series_step == 1.0
    assert series_mapping.series_unit == 'SECOND'
    assert series_mapping.series_exponent == 0.0
    assert series_mapping.number_of_series_points == img.shape[0]


@needs_nibabel_data('nitest-cifti2')
def test_read_labels():
    img = ci.Cifti2Image.from_filename(DATA_FILE5)
    label_mapping = img.header.matrix.get_index_map(0)

    expected_names = [
        'Composite Parcellation-lh (FRB08_OFP03_retinotopic)',
        'Brodmann lh (from colin.R via pals_R-to-fs_LR)',
        'MEDIAL WALL lh (fs_LR)',
    ]
    assert img.shape[0] == len(expected_names)
    assert len(list(label_mapping.named_maps)) == len(expected_names)

    some_expected_labels = {
        0: ('???', (0.667, 0.667, 0.667, 0.0)),
        1: ('MEDIAL.WALL', (0.075, 0.075, 0.075, 1.0)),
        2: ('BA2_FRB08', (0.467, 0.459, 0.055, 1.0)),
        3: ('BA1_FRB08', (0.475, 0.722, 0.859, 1.0)),
        4: ('BA3b_FRB08', (0.855, 0.902, 0.286, 1.0)),
        5: ('BA4p_FRB08', (0.902, 0.573, 0.122, 1.0)),
        89: ('36_B05', (0.467, 0.0, 0.129, 1.0)),
        90: ('35_B05', (0.467, 0.067, 0.067, 1.0)),
        91: ('28_B05', (0.467, 0.337, 0.271, 1.0)),
        92: ('29_B05', (0.267, 0.0, 0.529, 1.0)),
        93: ('26_B05', (0.757, 0.2, 0.227, 1.0)),
        94: ('33_B05', (0.239, 0.082, 0.373, 1.0)),
        95: ('13b_OFP03', (1.0, 1.0, 0.0, 1.0)),
    }

    for named_map, name in zip(label_mapping.named_maps, expected_names):
        assert named_map.map_name == name
        assert len(named_map.metadata) == 0
        assert len(named_map.label_table) == 96
        for index, (label, rgba) in some_expected_labels.items():
            assert named_map.label_table[index].label == label
            assert named_map.label_table[index].rgba == rgba


class TestCifti2SingleHeader(tn2.TestNifti2SingleHeader):
    header_class = _Cifti2AsNiftiHeader
    _pixdim_message = 'pixdim[1,2,3] should be zero or positive'

    def test_pixdim_checks(self):
        hdr_t = self.header_class()
        for i in (1, 2, 3):
            hdr = hdr_t.copy()
            hdr['pixdim'][i] = -1
            assert self._dxer(hdr) == self._pixdim_message

    def test_nifti_qfac_checks(self):
        # Test qfac is 1 or -1 or 0
        hdr = self.header_class()
        # 1, 0, -1 all OK
        hdr['pixdim'][0] = 1
        self.log_chk(hdr, 0)
        hdr['pixdim'][0] = 0
        self.log_chk(hdr, 0)
        hdr['pixdim'][0] = -1
        self.log_chk(hdr, 0)
        # Anything else is not
        hdr['pixdim'][0] = 2
        fhdr, message, raiser = self.log_chk(hdr, 20)
        assert fhdr['pixdim'][0] == 1
        assert message == 'pixdim[0] (qfac) should be 1 (default) or 0 or -1; setting qfac to 1'

    def test_pixdim_log_checks(self):
        # pixdim can be zero or positive
        HC = self.header_class
        hdr = HC()
        hdr['pixdim'][1] = -2  # severity 35
        fhdr, message, raiser = self.log_chk(hdr, 35)
        assert fhdr['pixdim'][1] == 2
        assert message == self._pixdim_message + '; setting to abs of pixdim values'

        pytest.raises(*raiser)

        hdr = HC()
        hdr['pixdim'][1:4] = 0  # No error or warning
        fhdr, message, raiser = self.log_chk(hdr, 0)
        assert raiser == ()
