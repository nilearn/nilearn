from copy import deepcopy

import numpy as np
import pytest

import nibabel.cifti2.cifti2_axes as axes

from .test_cifti2io_axes import check_rewrite

rand_affine = np.random.randn(4, 4)
vol_shape = (5, 10, 3)
use_label = {0: ('something', (0.2, 0.4, 0.1, 0.5)), 1: ('even better', (0.3, 0.8, 0.43, 0.9))}


def get_brain_models():
    """
    Generates a set of practice BrainModelAxis axes

    Yields
    ------
    BrainModelAxis axis
    """
    mask = np.zeros(vol_shape)
    mask[0, 1, 2] = 1
    mask[0, 4, 2] = True
    mask[0, 4, 0] = True
    yield axes.BrainModelAxis.from_mask(mask, 'ThalamusRight', rand_affine)
    mask[0, 0, 0] = True
    yield axes.BrainModelAxis.from_mask(mask, affine=rand_affine)

    yield axes.BrainModelAxis.from_surface([0, 5, 10], 15, 'CortexLeft')
    yield axes.BrainModelAxis.from_surface([0, 5, 10, 13], 15)

    surface_mask = np.zeros(15, dtype='bool')
    surface_mask[[2, 9, 14]] = True
    yield axes.BrainModelAxis.from_mask(surface_mask, name='CortexRight')


def get_parcels():
    """
    Generates a practice Parcel axis out of all practice brain models

    Returns
    -------
    Parcel axis
    """
    bml = list(get_brain_models())
    return axes.ParcelsAxis.from_brain_models(
        [('mixed', bml[0] + bml[2]), ('volume', bml[1]), ('surface', bml[3])]
    )


def get_scalar():
    """
    Generates a practice ScalarAxis axis with names ('one', 'two', 'three')

    Returns
    -------
    ScalarAxis axis
    """
    return axes.ScalarAxis(['one', 'two', 'three'])


def get_label():
    """
    Generates a practice LabelAxis axis with names ('one', 'two', 'three') and two labels

    Returns
    -------
    LabelAxis axis
    """
    return axes.LabelAxis(['one', 'two', 'three'], use_label)


def get_series():
    """
    Generates a set of 4 practice SeriesAxis axes with different starting times/lengths/time steps and units

    Yields
    ------
    SeriesAxis axis
    """
    yield axes.SeriesAxis(3, 10, 4)
    yield axes.SeriesAxis(8, 10, 3)
    yield axes.SeriesAxis(3, 2, 4)
    yield axes.SeriesAxis(5, 10, 5, 'HERTZ')


def get_axes():
    """
    Iterates through all of the practice axes defined in the functions above

    Yields
    ------
    Cifti2 axis
    """
    yield get_parcels()
    yield get_scalar()
    yield get_label()
    yield from get_brain_models()
    yield from get_series()


def test_brain_models():
    """
    Tests the introspection and creation of CIFTI-2 BrainModelAxis axes
    """
    bml = list(get_brain_models())
    assert len(bml[0]) == 3
    assert (bml[0].vertex == -1).all()
    assert (bml[0].voxel == [[0, 1, 2], [0, 4, 0], [0, 4, 2]]).all()
    assert bml[0][1][0] == 'CIFTI_MODEL_TYPE_VOXELS'
    assert (bml[0][1][1] == [0, 4, 0]).all()
    assert bml[0][1][2] == axes.BrainModelAxis.to_cifti_brain_structure_name('thalamus_right')
    assert len(bml[1]) == 4
    assert (bml[1].vertex == -1).all()
    assert (bml[1].voxel == [[0, 0, 0], [0, 1, 2], [0, 4, 0], [0, 4, 2]]).all()
    assert len(bml[2]) == 3
    assert (bml[2].voxel == -1).all()
    assert (bml[2].vertex == [0, 5, 10]).all()
    assert bml[2][1] == ('CIFTI_MODEL_TYPE_SURFACE', 5, 'CIFTI_STRUCTURE_CORTEX_LEFT')
    assert len(bml[3]) == 4
    assert (bml[3].voxel == -1).all()
    assert (bml[3].vertex == [0, 5, 10, 13]).all()
    assert bml[4][1] == ('CIFTI_MODEL_TYPE_SURFACE', 9, 'CIFTI_STRUCTURE_CORTEX_RIGHT')
    assert len(bml[4]) == 3
    assert (bml[4].voxel == -1).all()
    assert (bml[4].vertex == [2, 9, 14]).all()

    for bm, label, is_surface in zip(
        bml,
        ['ThalamusRight', 'Other', 'cortex_left', 'Other'],
        (False, False, True, True),
    ):
        assert np.all(bm.surface_mask == ~bm.volume_mask)
        structures = list(bm.iter_structures())
        assert len(structures) == 1
        name = structures[0][0]
        assert name == axes.BrainModelAxis.to_cifti_brain_structure_name(label)
        if is_surface:
            assert bm.nvertices[name] == 15
        else:
            assert name not in bm.nvertices
            assert (bm.affine == rand_affine).all()
            assert bm.volume_shape == vol_shape

    bmt = bml[0] + bml[1] + bml[2]
    assert len(bmt) == 10
    structures = list(bmt.iter_structures())
    assert len(structures) == 3
    for bm, (name, _, bm_split) in zip(bml[:3], structures):
        assert bm == bm_split
        assert (bm_split.name == name).all()
        assert bm == bmt[bmt.name == bm.name[0]]
        assert bm == bmt[np.where(bmt.name == bm.name[0])]

    bmt = bmt + bml[2]
    assert len(bmt) == 13
    structures = list(bmt.iter_structures())
    assert len(structures) == 3
    assert len(structures[-1][2]) == 6

    # break brain model
    bmt.affine = np.eye(4)
    with pytest.raises(ValueError):
        bmt.affine = np.eye(3)
    with pytest.raises(ValueError):
        bmt.affine = np.eye(4).flatten()

    bmt.volume_shape = (5, 3, 1)
    with pytest.raises(ValueError):
        bmt.volume_shape = (5.0, 3, 1)
    with pytest.raises(ValueError):
        bmt.volume_shape = (5, 3, 1, 4)

    with pytest.raises(IndexError):
        bmt['thalamus_left']

    # Test the constructor
    bm_vox = axes.BrainModelAxis(
        'thalamus_left',
        voxel=np.ones((5, 3), dtype=int),
        affine=np.eye(4),
        volume_shape=(2, 3, 4),
    )
    assert np.all(bm_vox.name == ['CIFTI_STRUCTURE_THALAMUS_LEFT'] * 5)
    assert np.array_equal(bm_vox.vertex, np.full(5, -1))
    assert np.array_equal(bm_vox.voxel, np.full((5, 3), 1))
    with pytest.raises(ValueError):
        # no volume shape
        axes.BrainModelAxis(
            'thalamus_left',
            voxel=np.ones((5, 3), dtype=int),
            affine=np.eye(4),
        )
    with pytest.raises(ValueError):
        # no affine
        axes.BrainModelAxis(
            'thalamus_left',
            voxel=np.ones((5, 3), dtype=int),
            volume_shape=(2, 3, 4),
        )
    with pytest.raises(ValueError):
        # incorrect name
        axes.BrainModelAxis(
            'random_name',
            voxel=np.ones((5, 3), dtype=int),
            affine=np.eye(4),
            volume_shape=(2, 3, 4),
        )
    with pytest.raises(ValueError):
        # negative voxel indices
        axes.BrainModelAxis(
            'thalamus_left',
            voxel=-np.ones((5, 3), dtype=int),
            affine=np.eye(4),
            volume_shape=(2, 3, 4),
        )
    with pytest.raises(ValueError):
        # no voxels or vertices
        axes.BrainModelAxis(
            'thalamus_left',
            affine=np.eye(4),
            volume_shape=(2, 3, 4),
        )
    with pytest.raises(ValueError):
        # incorrect voxel shape
        axes.BrainModelAxis(
            'thalamus_left',
            voxel=np.ones((5, 2), dtype=int),
            affine=np.eye(4),
            volume_shape=(2, 3, 4),
        )

    bm_vertex = axes.BrainModelAxis(
        'cortex_left',
        vertex=np.ones(5, dtype=int),
        nvertices={'cortex_left': 20},
    )
    assert np.array_equal(bm_vertex.name, ['CIFTI_STRUCTURE_CORTEX_LEFT'] * 5)
    assert np.array_equal(bm_vertex.vertex, np.full(5, 1))
    assert np.array_equal(bm_vertex.voxel, np.full((5, 3), -1))
    with pytest.raises(ValueError):
        axes.BrainModelAxis('cortex_left', vertex=np.ones(5, dtype=int))
    with pytest.raises(ValueError):
        axes.BrainModelAxis(
            'cortex_left',
            vertex=np.ones(5, dtype=int),
            nvertices={'cortex_right': 20},
        )
    with pytest.raises(ValueError):
        axes.BrainModelAxis(
            'cortex_left',
            vertex=-np.ones(5, dtype=int),
            nvertices={'cortex_left': 20},
        )

    # test from_mask errors
    with pytest.raises(ValueError):
        # affine should be 4x4 matrix
        axes.BrainModelAxis.from_mask(np.arange(5) > 2, affine=np.ones(5))
    with pytest.raises(ValueError):
        # only 1D or 3D masks accepted
        axes.BrainModelAxis.from_mask(np.ones((5, 3)))

    # tests error in adding together or combining as ParcelsAxis
    bm_vox = axes.BrainModelAxis(
        'thalamus_left',
        voxel=np.ones((5, 3), dtype=int),
        affine=np.eye(4),
        volume_shape=(2, 3, 4),
    )
    bm_vox + bm_vox
    assert (bm_vertex + bm_vox)[: bm_vertex.size] == bm_vertex
    assert (bm_vox + bm_vertex)[: bm_vox.size] == bm_vox
    for bm_added in (bm_vox + bm_vertex, bm_vertex + bm_vox):
        assert bm_added.nvertices == bm_vertex.nvertices
        assert np.all(bm_added.affine == bm_vox.affine)
        assert bm_added.volume_shape == bm_vox.volume_shape

    axes.ParcelsAxis.from_brain_models([('a', bm_vox), ('b', bm_vox)])
    with pytest.raises(Exception):
        bm_vox + get_label()

    bm_other_shape = axes.BrainModelAxis(
        'thalamus_left', voxel=np.ones((5, 3), dtype=int), affine=np.eye(4), volume_shape=(4, 3, 4)
    )
    with pytest.raises(ValueError):
        bm_vox + bm_other_shape
    with pytest.raises(ValueError):
        axes.ParcelsAxis.from_brain_models([('a', bm_vox), ('b', bm_other_shape)])
    bm_other_affine = axes.BrainModelAxis(
        'thalamus_left',
        voxel=np.ones((5, 3), dtype=int),
        affine=np.eye(4) * 2,
        volume_shape=(2, 3, 4),
    )
    with pytest.raises(ValueError):
        bm_vox + bm_other_affine
    with pytest.raises(ValueError):
        axes.ParcelsAxis.from_brain_models([('a', bm_vox), ('b', bm_other_affine)])

    bm_vertex = axes.BrainModelAxis(
        'cortex_left', vertex=np.ones(5, dtype=int), nvertices={'cortex_left': 20}
    )
    bm_other_number = axes.BrainModelAxis(
        'cortex_left', vertex=np.ones(5, dtype=int), nvertices={'cortex_left': 30}
    )
    with pytest.raises(ValueError):
        bm_vertex + bm_other_number
    with pytest.raises(ValueError):
        axes.ParcelsAxis.from_brain_models([('a', bm_vertex), ('b', bm_other_number)])

    # test equalities
    bm_vox = axes.BrainModelAxis(
        'thalamus_left',
        voxel=np.ones((5, 3), dtype=int),
        affine=np.eye(4),
        volume_shape=(2, 3, 4),
    )
    bm_other = deepcopy(bm_vox)
    assert bm_vox == bm_other
    bm_other.voxel[1, 0] = 0
    assert bm_vox != bm_other

    bm_other = deepcopy(bm_vox)
    bm_other.vertex[1] = 10
    assert bm_vox == bm_other, 'vertices are ignored in volumetric BrainModelAxis'

    bm_other = deepcopy(bm_vox)
    bm_other.name[1] = 'BRAIN_STRUCTURE_OTHER'
    assert bm_vox != bm_other

    bm_other = deepcopy(bm_vox)
    bm_other.affine[0, 0] = 10
    assert bm_vox != bm_other

    bm_other = deepcopy(bm_vox)
    bm_other.affine = None
    assert bm_vox != bm_other
    assert bm_other != bm_vox

    bm_other = deepcopy(bm_vox)
    bm_other.volume_shape = (10, 3, 4)
    assert bm_vox != bm_other

    bm_vertex = axes.BrainModelAxis(
        'cortex_left', vertex=np.ones(5, dtype=int), nvertices={'cortex_left': 20}
    )
    bm_other = deepcopy(bm_vertex)
    assert bm_vertex == bm_other
    bm_other.voxel[1, 0] = 0
    assert bm_vertex == bm_other, 'voxels are ignored in surface BrainModelAxis'

    bm_other = deepcopy(bm_vertex)
    bm_other.vertex[1] = 10
    assert bm_vertex != bm_other

    bm_other = deepcopy(bm_vertex)
    bm_other.name[1] = 'BRAIN_STRUCTURE_CORTEX_RIGHT'
    assert bm_vertex != bm_other

    bm_other = deepcopy(bm_vertex)
    bm_other.nvertices['BRAIN_STRUCTURE_CORTEX_LEFT'] = 50
    assert bm_vertex != bm_other

    bm_other = deepcopy(bm_vertex)
    bm_other.nvertices['BRAIN_STRUCTURE_CORTEX_RIGHT'] = 20
    assert bm_vertex != bm_other

    assert bm_vox != get_parcels()
    assert bm_vertex != get_parcels()


def test_parcels():
    """
    Test the introspection and creation of CIFTI-2 Parcel axes
    """
    prc = get_parcels()
    assert isinstance(prc, axes.ParcelsAxis)
    assert prc[0] == ('mixed',) + prc['mixed']
    assert prc['mixed'][0].shape == (3, 3)
    assert len(prc['mixed'][1]) == 1
    assert prc['mixed'][1]['CIFTI_STRUCTURE_CORTEX_LEFT'].shape == (3,)

    assert prc[1] == ('volume',) + prc['volume']
    assert prc['volume'][0].shape == (4, 3)
    assert len(prc['volume'][1]) == 0

    assert prc[2] == ('surface',) + prc['surface']
    assert prc['surface'][0].shape == (0, 3)
    assert len(prc['surface'][1]) == 1
    assert prc['surface'][1]['CIFTI_STRUCTURE_OTHER'].shape == (4,)

    prc2 = prc + prc
    assert len(prc2) == 6
    assert (prc2.affine == prc.affine).all()
    assert prc2.nvertices == prc.nvertices
    assert prc2.volume_shape == prc.volume_shape
    assert prc2[:3] == prc
    assert prc2[3:] == prc

    assert prc2[3:]['mixed'][0].shape == (3, 3)
    assert len(prc2[3:]['mixed'][1]) == 1
    assert prc2[3:]['mixed'][1]['CIFTI_STRUCTURE_CORTEX_LEFT'].shape == (3,)

    with pytest.raises(IndexError):
        prc['non_existent']

    prc['surface']
    with pytest.raises(IndexError):
        # parcel exists twice
        prc2['surface']

    # break parcels
    prc.affine = np.eye(4)
    with pytest.raises(ValueError):
        prc.affine = np.eye(3)
    with pytest.raises(ValueError):
        prc.affine = np.eye(4).flatten()

    prc.volume_shape = (5, 3, 1)
    with pytest.raises(ValueError):
        prc.volume_shape = (5.0, 3, 1)
    with pytest.raises(ValueError):
        prc.volume_shape = (5, 3, 1, 4)

    # break adding of parcels
    with pytest.raises(Exception):
        prc + get_label()

    prc = get_parcels()
    other_prc = get_parcels()
    prc + other_prc

    other_prc = get_parcels()
    other_prc.affine = np.eye(4) * 2
    with pytest.raises(ValueError):
        prc + other_prc

    other_prc = get_parcels()
    other_prc.volume_shape = (20, 3, 4)
    with pytest.raises(ValueError):
        prc + other_prc

    # test parcel equalities
    prc = get_parcels()
    assert prc != get_scalar()

    prc_other = deepcopy(prc)
    assert prc == prc_other
    assert prc != prc_other[:2]
    assert prc == prc_other[:]
    prc_other.affine[0, 0] = 10
    assert prc != prc_other

    prc_other = deepcopy(prc)
    prc_other.affine = None
    assert prc != prc_other
    assert prc_other != prc
    assert (prc + prc_other).affine is not None
    assert (prc_other + prc).affine is not None

    prc_other = deepcopy(prc)
    prc_other.volume_shape = (10, 3, 4)
    assert prc != prc_other
    with pytest.raises(ValueError):
        prc + prc_other

    prc_other = deepcopy(prc)
    prc_other.nvertices['CIFTI_STRUCTURE_CORTEX_LEFT'] = 80
    assert prc != prc_other
    with pytest.raises(ValueError):
        prc + prc_other

    prc_other = deepcopy(prc)
    prc_other.voxels[0] = np.ones((2, 3), dtype='i4')
    assert prc != prc_other

    prc_other = deepcopy(prc)
    prc_other.voxels[0] = prc_other.voxels * 2
    assert prc != prc_other

    prc_other = deepcopy(prc)
    prc_other.vertices[0]['CIFTI_STRUCTURE_CORTEX_LEFT'] = np.ones((8,), dtype='i4')
    assert prc != prc_other

    prc_other = deepcopy(prc)
    prc_other.vertices[0]['CIFTI_STRUCTURE_CORTEX_LEFT'] *= 2
    assert prc != prc_other

    prc_other = deepcopy(prc)
    prc_other.name[0] = 'new_name'
    assert prc != prc_other

    # test direct initialisation
    test_parcel = axes.ParcelsAxis(
        voxels=[np.ones((3, 2), dtype=int)],
        vertices=[{}],
        name=['single_voxel'],
        affine=np.eye(4),
        volume_shape=(2, 3, 4),
    )
    assert len(test_parcel) == 1

    # test direct initialisation with multiple parcels
    test_parcel = axes.ParcelsAxis(
        voxels=[np.ones((3, 2), dtype=int), np.zeros((3, 2), dtype=int)],
        vertices=[{}, {}],
        name=['first_parcel', 'second_parcel'],
        affine=np.eye(4),
        volume_shape=(2, 3, 4),
    )
    assert len(test_parcel) == 2

    # test direct initialisation with ragged voxel/vertices array
    test_parcel = axes.ParcelsAxis(
        voxels=[np.ones((3, 2), dtype=int), np.zeros((5, 2), dtype=int)],
        vertices=[{}, {}],
        name=['first_parcel', 'second_parcel'],
        affine=np.eye(4),
        volume_shape=(2, 3, 4),
    )
    assert len(test_parcel) == 2

    with pytest.raises(ValueError):
        axes.ParcelsAxis(
            voxels=[np.ones((3, 2), dtype=int)],
            vertices=[{}],
            name=[['single_voxel']],  # wrong shape name array
            affine=np.eye(4),
            volume_shape=(2, 3, 4),
        )


def test_scalar():
    """
    Test the introspection and creation of CIFTI-2 ScalarAxis axes
    """
    sc = get_scalar()
    assert len(sc) == 3
    assert isinstance(sc, axes.ScalarAxis)
    assert (sc.name == ['one', 'two', 'three']).all()
    assert (sc.meta == [{}] * 3).all()
    assert sc[1] == ('two', {})
    sc2 = sc + sc
    assert len(sc2) == 6
    assert (sc2.name == ['one', 'two', 'three', 'one', 'two', 'three']).all()
    assert (sc2.meta == [{}] * 6).all()
    assert sc2[:3] == sc
    assert sc2[3:] == sc

    sc.meta[1]['a'] = 3
    assert 'a' not in sc.meta

    # test equalities
    assert sc != get_label()
    with pytest.raises(Exception):
        sc + get_label()

    sc_other = deepcopy(sc)
    assert sc == sc_other
    assert sc != sc_other[:2]
    assert sc == sc_other[:]
    sc_other.name[0] = 'new_name'
    assert sc != sc_other

    sc_other = deepcopy(sc)
    sc_other.meta[0]['new_key'] = 'new_entry'
    assert sc != sc_other
    sc.meta[0]['new_key'] = 'new_entry'
    assert sc == sc_other

    # test constructor
    assert axes.ScalarAxis(['scalar_name'], [{}]) == axes.ScalarAxis(['scalar_name'])

    with pytest.raises(ValueError):
        axes.ScalarAxis([['scalar_name']])  # wrong shape

    with pytest.raises(ValueError):
        axes.ScalarAxis(['scalar_name'], [{}, {}])  # wrong size


def test_label():
    """
    Test the introspection and creation of CIFTI-2 ScalarAxis axes
    """
    lab = get_label()
    assert len(lab) == 3
    assert isinstance(lab, axes.LabelAxis)
    assert (lab.name == ['one', 'two', 'three']).all()
    assert (lab.meta == [{}] * 3).all()
    assert (lab.label == [use_label] * 3).all()
    assert lab[1] == ('two', use_label, {})
    lab2 = lab + lab
    assert len(lab2) == 6
    assert (lab2.name == ['one', 'two', 'three', 'one', 'two', 'three']).all()
    assert (lab2.meta == [{}] * 6).all()
    assert (lab2.label == [use_label] * 6).all()
    assert lab2[:3] == lab
    assert lab2[3:] == lab

    # test equalities
    lab = get_label()
    assert lab != get_scalar()
    with pytest.raises(Exception):
        lab + get_scalar()

    other_lab = deepcopy(lab)
    assert lab != other_lab[:2]
    assert lab == other_lab[:]
    other_lab.name[0] = 'new_name'
    assert lab != other_lab

    other_lab = deepcopy(lab)
    other_lab.meta[0]['new_key'] = 'new_item'
    assert 'new_key' not in other_lab.meta[1]
    assert lab != other_lab
    lab.meta[0]['new_key'] = 'new_item'
    assert lab == other_lab

    other_lab = deepcopy(lab)
    other_lab.label[0][20] = ('new_label', (0, 0, 0, 1))
    assert lab != other_lab
    assert 20 not in other_lab.label[1]
    lab.label[0][20] = ('new_label', (0, 0, 0, 1))
    assert lab == other_lab

    # test constructor
    assert axes.LabelAxis(['scalar_name'], [{}], [{}]) == axes.LabelAxis(['scalar_name'], [{}])

    with pytest.raises(ValueError):
        axes.LabelAxis([['scalar_name']], [{}])  # wrong shape

    with pytest.raises(ValueError):
        axes.LabelAxis(['scalar_name'], [{}, {}])  # wrong size


def test_series():
    """
    Test the introspection and creation of CIFTI-2 SeriesAxis axes
    """
    sr = list(get_series())
    assert sr[0].unit == 'SECOND'
    assert sr[1].unit == 'SECOND'
    assert sr[2].unit == 'SECOND'
    assert sr[3].unit == 'HERTZ'
    sr[0].unit = 'hertz'
    assert sr[0].unit == 'HERTZ'
    with pytest.raises(ValueError):
        sr[0].unit = 'non_existent'

    sr = list(get_series())
    assert (sr[0].time == np.arange(4) * 10 + 3).all()
    assert (sr[1].time == np.arange(3) * 10 + 8).all()
    assert (sr[2].time == np.arange(4) * 2 + 3).all()
    assert ((sr[0] + sr[1]).time == np.arange(7) * 10 + 3).all()
    assert ((sr[1] + sr[0]).time == np.arange(7) * 10 + 8).all()
    assert ((sr[1] + sr[0] + sr[0]).time == np.arange(11) * 10 + 8).all()
    assert sr[1][2] == 28
    assert sr[1][-2] == sr[1].time[-2]

    with pytest.raises(ValueError):
        sr[0] + sr[2]
    with pytest.raises(ValueError):
        sr[2] + sr[1]
    with pytest.raises(ValueError):
        sr[0] + sr[3]
    with pytest.raises(ValueError):
        sr[3] + sr[1]
    with pytest.raises(ValueError):
        sr[3] + sr[2]

    # test slicing
    assert (sr[0][1:3].time == sr[0].time[1:3]).all()
    assert (sr[0][1:].time == sr[0].time[1:]).all()
    assert (sr[0][:-2].time == sr[0].time[:-2]).all()
    assert (sr[0][1:-1].time == sr[0].time[1:-1]).all()
    assert (sr[0][1:-1:2].time == sr[0].time[1:-1:2]).all()
    assert (sr[0][::2].time == sr[0].time[::2]).all()
    assert (sr[0][:10:2].time == sr[0].time[::2]).all()
    assert (sr[0][10:].time == sr[0].time[10:]).all()
    assert (sr[0][10:12].time == sr[0].time[10:12]).all()
    assert (sr[0][10::-1].time == sr[0].time[10::-1]).all()
    assert (sr[0][3:1:-1].time == sr[0].time[3:1:-1]).all()
    assert (sr[0][1:3:-1].time == sr[0].time[1:3:-1]).all()

    with pytest.raises(IndexError):
        assert sr[0][[0, 1]]
    with pytest.raises(IndexError):
        assert sr[0][20]
    with pytest.raises(IndexError):
        assert sr[0][-20]

    # test_equalities
    sr = next(get_series())
    with pytest.raises(Exception):
        sr + get_scalar()
    assert sr != sr[:2]
    assert sr == sr[:]

    for key, value in (
        ('start', 20),
        ('step', 7),
        ('size', 14),
        ('unit', 'HERTZ'),
    ):
        sr_other = deepcopy(sr)
        assert sr == sr_other
        setattr(sr_other, key, value)
        assert sr != sr_other


def test_writing():
    """
    Tests the writing and reading back in of custom created CIFTI-2 axes
    """
    for ax1 in get_axes():
        for ax2 in get_axes():
            arr = np.random.randn(len(ax1), len(ax2))
            check_rewrite(arr, (ax1, ax2))


def test_common_interface():
    """
    Tests the common interface for all custom created CIFTI-2 axes
    """
    for axis1, axis2 in zip(get_axes(), get_axes()):
        assert axis1 == axis2
        concatenated = axis1 + axis2
        assert axis1 != concatenated
        assert axis1 == concatenated[: axis1.size]
        if isinstance(axis1, axes.SeriesAxis):
            assert axis2 != concatenated[axis1.size :]
        else:
            assert axis2 == concatenated[axis1.size :]

        assert len(axis1) == axis1.size
