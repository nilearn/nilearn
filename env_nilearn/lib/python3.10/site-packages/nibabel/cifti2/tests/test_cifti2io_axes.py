import os
import tempfile

import numpy as np

import nibabel as nib
from nibabel.cifti2 import cifti2, cifti2_axes
from nibabel.tests.nibabel_data import get_nibabel_data, needs_nibabel_data

test_directory = os.path.join(get_nibabel_data(), 'nitest-cifti2')

hcp_labels = [
    'CortexLeft',
    'CortexRight',
    'AccumbensLeft',
    'AccumbensRight',
    'AmygdalaLeft',
    'AmygdalaRight',
    'brain_stem',
    'CaudateLeft',
    'CaudateRight',
    'CerebellumLeft',
    'CerebellumRight',
    'Diencephalon_ventral_left',
    'Diencephalon_ventral_right',
    'HippocampusLeft',
    'HippocampusRight',
    'PallidumLeft',
    'PallidumRight',
    'PutamenLeft',
    'PutamenRight',
    'ThalamusLeft',
    'ThalamusRight',
]

hcp_n_elements = [
    29696,
    29716,
    135,
    140,
    315,
    332,
    3472,
    728,
    755,
    8709,
    9144,
    706,
    712,
    764,
    795,
    297,
    260,
    1060,
    1010,
    1288,
    1248,
]

hcp_affine = np.array(
    [[-2.0, 0.0, 0.0, 90.0], [0.0, 2.0, 0.0, -126.0], [0.0, 0.0, 2.0, -72.0], [0.0, 0.0, 0.0, 1.0]]
)


def check_hcp_grayordinates(brain_model):
    """Checks that a BrainModelAxis matches the expected 32k HCP grayordinates"""
    assert isinstance(brain_model, cifti2_axes.BrainModelAxis)
    structures = list(brain_model.iter_structures())
    assert len(structures) == len(hcp_labels)
    idx_start = 0
    for idx, (name, _, bm), label, nel in zip(
        range(len(structures)), structures, hcp_labels, hcp_n_elements
    ):
        if idx < 2:
            assert name in bm.nvertices.keys()
            assert (bm.voxel == -1).all()
            assert (bm.vertex != -1).any()
            assert bm.nvertices[name] == 32492
        else:
            assert name not in bm.nvertices.keys()
            assert (bm.voxel != -1).any()
            assert (bm.vertex == -1).all()
            assert (bm.affine == hcp_affine).all()
            assert bm.volume_shape == (91, 109, 91)
        assert name == cifti2_axes.BrainModelAxis.to_cifti_brain_structure_name(label)
        assert len(bm) == nel
        assert (bm.name == brain_model.name[idx_start : idx_start + nel]).all()
        assert (bm.voxel == brain_model.voxel[idx_start : idx_start + nel]).all()
        assert (bm.vertex == brain_model.vertex[idx_start : idx_start + nel]).all()
        idx_start += nel
    assert idx_start == len(brain_model)

    assert (brain_model.vertex[:5] == np.arange(5)).all()
    assert structures[0][2].vertex[-1] == 32491
    assert structures[1][2].vertex[0] == 0
    assert structures[1][2].vertex[-1] == 32491
    assert structures[-1][2].name[-1] == brain_model.name[-1]
    assert (structures[-1][2].voxel[-1] == brain_model.voxel[-1]).all()
    assert structures[-1][2].vertex[-1] == brain_model.vertex[-1]
    assert (brain_model.voxel[-1] == [38, 55, 46]).all()
    assert (brain_model.voxel[70000] == [56, 22, 19]).all()


def check_Conte69(brain_model):
    """Checks that the BrainModelAxis matches the expected Conte69 surface coordinates"""
    assert isinstance(brain_model, cifti2_axes.BrainModelAxis)
    structures = list(brain_model.iter_structures())
    assert len(structures) == 2
    assert structures[0][0] == 'CIFTI_STRUCTURE_CORTEX_LEFT'
    assert structures[0][2].surface_mask.all()
    assert structures[1][0] == 'CIFTI_STRUCTURE_CORTEX_RIGHT'
    assert structures[1][2].surface_mask.all()
    assert (brain_model.voxel == -1).all()

    assert (brain_model.vertex[:5] == np.arange(5)).all()
    assert structures[0][2].vertex[-1] == 32491
    assert structures[1][2].vertex[0] == 0
    assert structures[1][2].vertex[-1] == 32491


def check_rewrite(arr, axes, extension='.nii'):
    """
    Checks whether writing the Cifti2 array to disc and reading it back in gives the same object

    Parameters
    ----------
    arr : array
        N-dimensional array of data
    axes : Sequence[cifti2_axes.Axis]
        sequence of length N with the meaning of the rows/columns along each dimension
    extension : str
        custom extension to use
    """
    (fd, name) = tempfile.mkstemp(extension)
    cifti2.Cifti2Image(arr, header=axes).to_filename(name)
    img = nib.load(name)
    arr2 = img.get_fdata()
    assert np.allclose(arr, arr2)
    for idx in range(len(img.shape)):
        assert axes[idx] == img.header.get_axis(idx)
    return img


@needs_nibabel_data('nitest-cifti2')
def test_read_ones():
    img = nib.load(os.path.join(test_directory, 'ones.dscalar.nii'))
    arr = img.get_fdata()
    axes = [img.header.get_axis(dim) for dim in range(2)]
    assert (arr == 1).all()
    assert isinstance(axes[0], cifti2_axes.ScalarAxis)
    assert len(axes[0]) == 1
    assert axes[0].name[0] == 'ones'
    assert axes[0].meta[0] == {}
    check_hcp_grayordinates(axes[1])
    img = check_rewrite(arr, axes)
    check_hcp_grayordinates(img.header.get_axis(1))


@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_dscalar():
    img = nib.load(
        os.path.join(test_directory, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.dscalar.nii')
    )
    arr = img.get_fdata()
    axes = [img.header.get_axis(dim) for dim in range(2)]
    assert isinstance(axes[0], cifti2_axes.ScalarAxis)
    assert len(axes[0]) == 2
    assert axes[0].name[0] == 'MyelinMap_BC_decurv'
    assert axes[0].name[1] == 'corrThickness'
    assert axes[0].meta[0] == {
        'PaletteColorMapping': '<PaletteColorMapping Version="1">\n   <ScaleMode>MODE_AUTO_SCALE_PERCENTAGE</ScaleMode>\n   <AutoScalePercentageValues>98.000000 2.000000 2.000000 98.000000</AutoScalePercentageValues>\n   <UserScaleValues>-100.000000 0.000000 0.000000 100.000000</UserScaleValues>\n   <PaletteName>ROY-BIG-BL</PaletteName>\n   <InterpolatePalette>true</InterpolatePalette>\n   <DisplayPositiveData>true</DisplayPositiveData>\n   <DisplayZeroData>false</DisplayZeroData>\n   <DisplayNegativeData>true</DisplayNegativeData>\n   <ThresholdTest>THRESHOLD_TEST_SHOW_OUTSIDE</ThresholdTest>\n   <ThresholdType>THRESHOLD_TYPE_OFF</ThresholdType>\n   <ThresholdFailureInGreen>false</ThresholdFailureInGreen>\n   <ThresholdNormalValues>-1.000000 1.000000</ThresholdNormalValues>\n   <ThresholdMappedValues>-1.000000 1.000000</ThresholdMappedValues>\n   <ThresholdMappedAvgAreaValues>-1.000000 1.000000</ThresholdMappedAvgAreaValues>\n   <ThresholdDataName></ThresholdDataName>\n   <ThresholdRangeMode>PALETTE_THRESHOLD_RANGE_MODE_MAP</ThresholdRangeMode>\n</PaletteColorMapping>'
    }
    check_Conte69(axes[1])
    check_rewrite(arr, axes)


@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_dtseries():
    img = nib.load(
        os.path.join(test_directory, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.dtseries.nii')
    )
    arr = img.get_fdata()
    axes = [img.header.get_axis(dim) for dim in range(2)]
    assert isinstance(axes[0], cifti2_axes.SeriesAxis)
    assert len(axes[0]) == 2
    assert axes[0].start == 0
    assert axes[0].step == 1
    assert axes[0].size == arr.shape[0]
    assert (axes[0].time == [0, 1]).all()
    check_Conte69(axes[1])
    check_rewrite(arr, axes)


@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_dlabel():
    img = nib.load(
        os.path.join(test_directory, 'Conte69.parcellations_VGD11b.32k_fs_LR.dlabel.nii')
    )
    arr = img.get_fdata()
    axes = [img.header.get_axis(dim) for dim in range(2)]
    assert isinstance(axes[0], cifti2_axes.LabelAxis)
    assert len(axes[0]) == 3
    assert (
        axes[0].name
        == [
            'Composite Parcellation-lh (FRB08_OFP03_retinotopic)',
            'Brodmann lh (from colin.R via pals_R-to-fs_LR)',
            'MEDIAL WALL lh (fs_LR)',
        ]
    ).all()
    assert axes[0].label[1][70] == ('19_B05', (1.0, 0.867, 0.467, 1.0))
    assert (axes[0].meta == [{}] * 3).all()
    check_Conte69(axes[1])
    check_rewrite(arr, axes)


@needs_nibabel_data('nitest-cifti2')
def test_read_conte69_ptseries():
    img = nib.load(
        os.path.join(test_directory, 'Conte69.MyelinAndCorrThickness.32k_fs_LR.ptseries.nii')
    )
    arr = img.get_fdata()
    axes = [img.header.get_axis(dim) for dim in range(2)]
    assert isinstance(axes[0], cifti2_axes.SeriesAxis)
    assert len(axes[0]) == 2
    assert axes[0].start == 0
    assert axes[0].step == 1
    assert axes[0].size == arr.shape[0]
    assert (axes[0].time == [0, 1]).all()

    assert len(axes[1]) == 54
    voxels, vertices = axes[1]['ER_FRB08']
    assert voxels.shape == (0, 3)
    assert len(vertices) == 2
    assert vertices['CIFTI_STRUCTURE_CORTEX_LEFT'].shape == (206 // 2,)
    assert vertices['CIFTI_STRUCTURE_CORTEX_RIGHT'].shape == (206 // 2,)
    check_rewrite(arr, axes)
