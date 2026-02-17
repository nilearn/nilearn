import copy
import operator
import unittest
import warnings
from collections import defaultdict

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from ...testing import assert_arrays_equal, clear_and_catch_warnings
from .. import tractogram as module_tractogram
from ..tractogram import (
    LazyDict,
    LazyTractogram,
    PerArrayDict,
    PerArraySequenceDict,
    Tractogram,
    TractogramItem,
    is_data_dict,
    is_lazy_dict,
)

DATA = {}


def make_fake_streamline(
    nb_points, data_per_point_shapes={}, data_for_streamline_shapes={}, rng=None
):
    """Make a single streamline according to provided requirements."""
    if rng is None:
        rng = np.random.RandomState()

    streamline = rng.randn(nb_points, 3).astype('f4')

    data_per_point = {}
    for k, shape in data_per_point_shapes.items():
        data_per_point[k] = rng.randn(*((nb_points,) + shape)).astype('f4')

    data_for_streamline = {}
    for k, shape in data_for_streamline.items():
        data_for_streamline[k] = rng.randn(*shape).astype('f4')

    return streamline, data_per_point, data_for_streamline


def make_fake_tractogram(
    list_nb_points, data_per_point_shapes={}, data_for_streamline_shapes={}, rng=None
):
    """Make multiple streamlines according to provided requirements."""
    all_streamlines = []
    all_data_per_point = defaultdict(list)
    all_data_per_streamline = defaultdict(list)
    for nb_points in list_nb_points:
        data = make_fake_streamline(
            nb_points, data_per_point_shapes, data_for_streamline_shapes, rng
        )
        streamline, data_per_point, data_for_streamline = data

        all_streamlines.append(streamline)
        for k, v in data_per_point.items():
            all_data_per_point[k].append(v)

        for k, v in data_for_streamline.items():
            all_data_per_streamline[k].append(v)

    return all_streamlines, all_data_per_point, all_data_per_streamline


def make_dummy_streamline(nb_points):
    """Make the streamlines that have been used to create test data files."""
    if nb_points == 1:
        streamline = np.arange(1 * 3, dtype='f4').reshape((1, 3))
        data_per_point = {
            'fa': np.array([[0.2]], dtype='f4'),
            'colors': np.array([(1, 0, 0)] * 1, dtype='f4'),
        }
        data_for_streamline = {
            'mean_curvature': np.array([1.11], dtype='f4'),
            'mean_torsion': np.array([1.22], dtype='f4'),
            'mean_colors': np.array([1, 0, 0], dtype='f4'),
            'clusters_labels': np.array([0, 1], dtype='i4'),
        }

    elif nb_points == 2:
        streamline = np.arange(2 * 3, dtype='f4').reshape((2, 3))
        data_per_point = {
            'fa': np.array([[0.3], [0.4]], dtype='f4'),
            'colors': np.array([(0, 1, 0)] * 2, dtype='f4'),
        }
        data_for_streamline = {
            'mean_curvature': np.array([2.11], dtype='f4'),
            'mean_torsion': np.array([2.22], dtype='f4'),
            'mean_colors': np.array([0, 1, 0], dtype='f4'),
            'clusters_labels': np.array([2, 3, 4], dtype='i4'),
        }

    elif nb_points == 5:
        streamline = np.arange(5 * 3, dtype='f4').reshape((5, 3))
        data_per_point = {
            'fa': np.array([[0.5], [0.6], [0.6], [0.7], [0.8]], dtype='f4'),
            'colors': np.array([(0, 0, 1)] * 5, dtype='f4'),
        }
        data_for_streamline = {
            'mean_curvature': np.array([3.11], dtype='f4'),
            'mean_torsion': np.array([3.22], dtype='f4'),
            'mean_colors': np.array([0, 0, 1], dtype='f4'),
            'clusters_labels': np.array([5, 6, 7, 8], dtype='i4'),
        }

    return streamline, data_per_point, data_for_streamline


def setup_module():
    global DATA
    DATA['rng'] = np.random.RandomState(1234)

    DATA['streamlines'] = []
    DATA['fa'] = []
    DATA['colors'] = []
    DATA['mean_curvature'] = []
    DATA['mean_torsion'] = []
    DATA['mean_colors'] = []
    DATA['clusters_labels'] = []
    for nb_points in [1, 2, 5]:
        data = make_dummy_streamline(nb_points)
        streamline, data_per_point, data_for_streamline = data
        DATA['streamlines'].append(streamline)
        DATA['fa'].append(data_per_point['fa'])
        DATA['colors'].append(data_per_point['colors'])
        DATA['mean_curvature'].append(data_for_streamline['mean_curvature'])
        DATA['mean_torsion'].append(data_for_streamline['mean_torsion'])
        DATA['mean_colors'].append(data_for_streamline['mean_colors'])
        DATA['clusters_labels'].append(data_for_streamline['clusters_labels'])

    DATA['data_per_point'] = {'colors': DATA['colors'], 'fa': DATA['fa']}
    DATA['data_per_streamline'] = {
        'mean_curvature': DATA['mean_curvature'],
        'mean_torsion': DATA['mean_torsion'],
        'mean_colors': DATA['mean_colors'],
        'clusters_labels': DATA['clusters_labels'],
    }

    DATA['empty_tractogram'] = Tractogram(affine_to_rasmm=np.eye(4))
    DATA['simple_tractogram'] = Tractogram(DATA['streamlines'], affine_to_rasmm=np.eye(4))
    DATA['tractogram'] = Tractogram(
        DATA['streamlines'],
        DATA['data_per_streamline'],
        DATA['data_per_point'],
        affine_to_rasmm=np.eye(4),
    )

    DATA['streamlines_func'] = lambda: (e for e in DATA['streamlines'])
    DATA['data_per_point_func'] = {
        'colors': lambda: (e for e in DATA['colors']),
        'fa': lambda: (e for e in DATA['fa']),
    }
    DATA['data_per_streamline_func'] = {
        'mean_curvature': lambda: (e for e in DATA['mean_curvature']),
        'mean_torsion': lambda: (e for e in DATA['mean_torsion']),
        'mean_colors': lambda: (e for e in DATA['mean_colors']),
        'clusters_labels': lambda: (e for e in DATA['clusters_labels']),
    }

    DATA['lazy_tractogram'] = LazyTractogram(
        DATA['streamlines_func'],
        DATA['data_per_streamline_func'],
        DATA['data_per_point_func'],
        affine_to_rasmm=np.eye(4),
    )


def check_tractogram_item(tractogram_item, streamline, data_for_streamline={}, data_for_points={}):
    assert_array_equal(tractogram_item.streamline, streamline)

    assert len(tractogram_item.data_for_streamline) == len(data_for_streamline)
    for key in data_for_streamline.keys():
        assert_array_equal(tractogram_item.data_for_streamline[key], data_for_streamline[key])

    assert len(tractogram_item.data_for_points) == len(data_for_points)
    for key in data_for_points.keys():
        assert_arrays_equal(tractogram_item.data_for_points[key], data_for_points[key])


def assert_tractogram_item_equal(t1, t2):
    check_tractogram_item(t1, t2.streamline, t2.data_for_streamline, t2.data_for_points)


def check_tractogram(tractogram, streamlines=[], data_per_streamline={}, data_per_point={}):
    streamlines = list(streamlines)
    assert len(tractogram) == len(streamlines)
    assert_arrays_equal(tractogram.streamlines, streamlines)
    [t for t in tractogram]  # Force iteration through tractogram.

    assert len(tractogram.data_per_streamline) == len(data_per_streamline)
    for key in data_per_streamline.keys():
        assert_arrays_equal(tractogram.data_per_streamline[key], data_per_streamline[key])

    assert len(tractogram.data_per_point) == len(data_per_point)
    for key in data_per_point.keys():
        assert_arrays_equal(tractogram.data_per_point[key], data_per_point[key])


def assert_tractogram_equal(t1, t2):
    check_tractogram(t1, t2.streamlines, t2.data_per_streamline, t2.data_per_point)


def extender(a, b):
    a.extend(b)
    return a


class TestPerArrayDict(unittest.TestCase):
    def test_per_array_dict_creation(self):
        # Create a PerArrayDict object using another
        # PerArrayDict object.
        nb_streamlines = len(DATA['tractogram'])
        data_per_streamline = DATA['tractogram'].data_per_streamline
        data_dict = PerArrayDict(nb_streamlines, data_per_streamline)
        assert data_dict.keys() == data_per_streamline.keys()
        for k in data_dict.keys():
            if isinstance(data_dict[k], np.ndarray) and np.all(
                data_dict[k].shape[0] == data_dict[k].shape
            ):
                assert_array_equal(data_dict[k], data_per_streamline[k])

        del data_dict['mean_curvature']
        assert len(data_dict) == len(data_per_streamline) - 1

        # Create a PerArrayDict object using an existing dict object.
        data_per_streamline = DATA['data_per_streamline']
        data_dict = PerArrayDict(nb_streamlines, data_per_streamline)
        assert data_dict.keys() == data_per_streamline.keys()
        for k in data_dict.keys():
            if isinstance(data_dict[k], np.ndarray) and np.all(
                data_dict[k].shape[0] == data_dict[k].shape
            ):
                assert_array_equal(data_dict[k], data_per_streamline[k])

        del data_dict['mean_curvature']
        assert len(data_dict) == len(data_per_streamline) - 1

        # Create a PerArrayDict object using keyword arguments.
        data_per_streamline = DATA['data_per_streamline']
        data_dict = PerArrayDict(nb_streamlines, **data_per_streamline)
        assert data_dict.keys() == data_per_streamline.keys()
        for k in data_dict.keys():
            if isinstance(data_dict[k], np.ndarray) and np.all(
                data_dict[k].shape[0] == data_dict[k].shape
            ):
                assert_array_equal(data_dict[k], data_per_streamline[k])

        del data_dict['mean_curvature']
        assert len(data_dict) == len(data_per_streamline) - 1

    def test_getitem(self):
        sdict = PerArrayDict(len(DATA['tractogram']), DATA['data_per_streamline'])

        with pytest.raises(KeyError):
            sdict['invalid']

        # Test slicing and advanced indexing.
        for k, v in DATA['tractogram'].data_per_streamline.items():
            assert k in sdict
            assert_arrays_equal(sdict[k], v)
            assert_arrays_equal(sdict[::2][k], v[::2])
            assert_arrays_equal(sdict[::-1][k], v[::-1])
            assert_arrays_equal(sdict[-1][k], v[-1])
            assert_arrays_equal(sdict[[0, -1]][k], v[[0, -1]])

    def test_extend(self):
        sdict = PerArrayDict(len(DATA['tractogram']), DATA['data_per_streamline'])

        new_data = {
            'mean_curvature': 2 * np.array(DATA['mean_curvature']),
            'mean_torsion': 3 * np.array(DATA['mean_torsion']),
            'mean_colors': 4 * np.array(DATA['mean_colors']),
            'clusters_labels': 5 * np.array(DATA['clusters_labels'], dtype=object),
        }
        sdict2 = PerArrayDict(len(DATA['tractogram']), new_data)

        sdict.extend(sdict2)
        assert len(sdict) == len(sdict2)
        for k in DATA['tractogram'].data_per_streamline:
            assert_arrays_equal(
                sdict[k][: len(DATA['tractogram'])], DATA['tractogram'].data_per_streamline[k]
            )
            assert_arrays_equal(sdict[k][len(DATA['tractogram']) :], new_data[k])

        # Extending with an empty PerArrayDict should change nothing.
        sdict_orig = copy.deepcopy(sdict)
        sdict.extend(PerArrayDict())
        for k in sdict_orig.keys():
            assert_arrays_equal(sdict[k], sdict_orig[k])

        # Test incompatible PerArrayDicts.
        # Other dict has more entries.
        new_data = {
            'mean_curvature': 2 * np.array(DATA['mean_curvature']),
            'mean_torsion': 3 * np.array(DATA['mean_torsion']),
            'mean_colors': 4 * np.array(DATA['mean_colors']),
            'clusters_labels': 5 * np.array(DATA['clusters_labels'], dtype=object),
            'other': 6 * np.array(DATA['mean_colors']),
        }
        sdict2 = PerArrayDict(len(DATA['tractogram']), new_data)

        with pytest.raises(ValueError):
            sdict.extend(sdict2)
        # Other dict has not the same entries (key mistmached).
        new_data = {
            'mean_curvature': 2 * np.array(DATA['mean_curvature']),
            'mean_torsion': 3 * np.array(DATA['mean_torsion']),
            'other': 4 * np.array(DATA['mean_colors']),
        }
        sdict2 = PerArrayDict(len(DATA['tractogram']), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)

        # Other dict has the right number of entries but wrong shape.
        new_data = {
            'mean_curvature': 2 * np.array(DATA['mean_curvature']),
            'mean_torsion': 3 * np.array(DATA['mean_torsion']),
            'mean_colors': 4 * np.array(DATA['mean_torsion']),
            'clusters_labels': 5 * np.array(DATA['clusters_labels'], dtype=object),
        }
        sdict2 = PerArrayDict(len(DATA['tractogram']), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)


class TestPerArraySequenceDict(unittest.TestCase):
    def test_per_array_sequence_dict_creation(self):
        # Create a PerArraySequenceDict object using another
        # PerArraySequenceDict object.
        total_nb_rows = DATA['tractogram'].streamlines.total_nb_rows
        data_per_point = DATA['tractogram'].data_per_point
        data_dict = PerArraySequenceDict(total_nb_rows, data_per_point)
        assert data_dict.keys() == data_per_point.keys()
        for k in data_dict.keys():
            assert_arrays_equal(data_dict[k], data_per_point[k])

        del data_dict['fa']
        assert len(data_dict) == len(data_per_point) - 1

        # Create a PerArraySequenceDict object using an existing dict object.
        data_per_point = DATA['data_per_point']
        data_dict = PerArraySequenceDict(total_nb_rows, data_per_point)
        assert data_dict.keys() == data_per_point.keys()
        for k in data_dict.keys():
            assert_arrays_equal(data_dict[k], data_per_point[k])

        del data_dict['fa']
        assert len(data_dict) == len(data_per_point) - 1

        # Create a PerArraySequenceDict object using keyword arguments.
        data_per_point = DATA['data_per_point']
        data_dict = PerArraySequenceDict(total_nb_rows, **data_per_point)
        assert data_dict.keys() == data_per_point.keys()
        for k in data_dict.keys():
            assert_arrays_equal(data_dict[k], data_per_point[k])

        del data_dict['fa']
        assert len(data_dict) == len(data_per_point) - 1

    def test_getitem(self):
        total_nb_rows = DATA['tractogram'].streamlines.total_nb_rows
        sdict = PerArraySequenceDict(total_nb_rows, DATA['data_per_point'])

        with pytest.raises(KeyError):
            sdict['invalid']

        # Test slicing and advanced indexing.
        for k, v in DATA['tractogram'].data_per_point.items():
            assert k in sdict
            assert_arrays_equal(sdict[k], v)
            assert_arrays_equal(sdict[::2][k], v[::2])
            assert_arrays_equal(sdict[::-1][k], v[::-1])
            assert_arrays_equal(sdict[-1][k], v[-1])
            assert_arrays_equal(sdict[[0, -1]][k], v[[0, -1]])

    def test_extend(self):
        total_nb_rows = DATA['tractogram'].streamlines.total_nb_rows
        sdict = PerArraySequenceDict(total_nb_rows, DATA['data_per_point'])

        # Test compatible PerArraySequenceDicts.
        list_nb_points = [2, 7, 4]
        data_per_point_shapes = {
            'colors': DATA['colors'][0].shape[1:],
            'fa': DATA['fa'][0].shape[1:],
        }
        _, new_data, _ = make_fake_tractogram(
            list_nb_points, data_per_point_shapes, rng=DATA['rng']
        )
        sdict2 = PerArraySequenceDict(np.sum(list_nb_points), new_data)

        sdict.extend(sdict2)
        assert len(sdict) == len(sdict2)
        for k in DATA['tractogram'].data_per_point:
            assert_arrays_equal(
                sdict[k][: len(DATA['tractogram'])], DATA['tractogram'].data_per_point[k]
            )
            assert_arrays_equal(sdict[k][len(DATA['tractogram']) :], new_data[k])

        # Extending with an empty PerArraySequenceDicts should change nothing.
        sdict_orig = copy.deepcopy(sdict)
        sdict.extend(PerArraySequenceDict())
        for k in sdict_orig.keys():
            assert_arrays_equal(sdict[k], sdict_orig[k])

        # Test incompatible PerArraySequenceDicts.
        # Other dict has more entries.
        data_per_point_shapes = {
            'colors': DATA['colors'][0].shape[1:],
            'fa': DATA['fa'][0].shape[1:],
            'other': (7,),
        }
        _, new_data, _ = make_fake_tractogram(
            list_nb_points, data_per_point_shapes, rng=DATA['rng']
        )
        sdict2 = PerArraySequenceDict(np.sum(list_nb_points), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)

        # Other dict has not the same entries (key mistmached).
        data_per_point_shapes = {
            'colors': DATA['colors'][0].shape[1:],
            'other': DATA['fa'][0].shape[1:],
        }
        _, new_data, _ = make_fake_tractogram(
            list_nb_points, data_per_point_shapes, rng=DATA['rng']
        )
        sdict2 = PerArraySequenceDict(np.sum(list_nb_points), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)

        # Other dict has the right number of entries but wrong shape.
        data_per_point_shapes = {
            'colors': DATA['colors'][0].shape[1:],
            'fa': DATA['fa'][0].shape[1:] + (3,),
        }
        _, new_data, _ = make_fake_tractogram(
            list_nb_points, data_per_point_shapes, rng=DATA['rng']
        )
        sdict2 = PerArraySequenceDict(np.sum(list_nb_points), new_data)
        with pytest.raises(ValueError):
            sdict.extend(sdict2)


class TestLazyDict(unittest.TestCase):
    def test_lazydict_creation(self):
        # Different ways of creating LazyDict
        lazy_dicts = []
        lazy_dicts += [LazyDict(DATA['data_per_streamline_func'])]
        lazy_dicts += [LazyDict(**DATA['data_per_streamline_func'])]

        expected_keys = DATA['data_per_streamline_func'].keys()
        for data_dict in lazy_dicts:
            assert is_lazy_dict(data_dict)
            assert data_dict.keys() == expected_keys
            for k in data_dict.keys():
                if isinstance(data_dict[k], np.ndarray) and np.all(
                    data_dict[k].shape[0] == data_dict[k].shape
                ):
                    assert_array_equal(list(data_dict[k]), list(DATA['data_per_streamline'][k]))

            assert len(data_dict) == len(DATA['data_per_streamline_func'])


class TestTractogramItem(unittest.TestCase):
    def test_creating_tractogram_item(self):
        rng = np.random.RandomState(42)
        streamline = rng.rand(rng.randint(10, 50), 3)
        colors = rng.rand(len(streamline), 3)
        mean_curvature = 1.11
        mean_color = np.array([0, 1, 0], dtype='f4')

        data_for_streamline = {'mean_curvature': mean_curvature, 'mean_color': mean_color}

        data_for_points = {'colors': colors}

        # Create a tractogram item with a streamline, data.
        t = TractogramItem(streamline, data_for_streamline, data_for_points)
        assert len(t) == len(streamline)
        assert_array_equal(t.streamline, streamline)
        assert_array_equal(list(t), streamline)
        assert_array_equal(t.data_for_streamline['mean_curvature'], mean_curvature)
        assert_array_equal(t.data_for_streamline['mean_color'], mean_color)
        assert_array_equal(t.data_for_points['colors'], colors)


class TestTractogram(unittest.TestCase):
    def test_tractogram_creation(self):
        # Create an empty tractogram.
        tractogram = Tractogram()
        check_tractogram(tractogram)
        assert tractogram.affine_to_rasmm is None

        # Create a tractogram with only streamlines
        tractogram = Tractogram(streamlines=DATA['streamlines'])
        check_tractogram(tractogram, DATA['streamlines'])

        # Create a tractogram with a given affine_to_rasmm.
        affine = np.diag([1, 2, 3, 1])
        tractogram = Tractogram(affine_to_rasmm=affine)
        assert_array_equal(tractogram.affine_to_rasmm, affine)

        # Create a tractogram with streamlines and other data.
        tractogram = Tractogram(
            DATA['streamlines'], DATA['data_per_streamline'], DATA['data_per_point']
        )

        check_tractogram(
            tractogram, DATA['streamlines'], DATA['data_per_streamline'], DATA['data_per_point']
        )

        assert is_data_dict(tractogram.data_per_streamline)
        assert is_data_dict(tractogram.data_per_point)

        # Create a tractogram from another tractogram attributes.
        tractogram2 = Tractogram(
            tractogram.streamlines, tractogram.data_per_streamline, tractogram.data_per_point
        )

        assert_tractogram_equal(tractogram2, tractogram)

        # Create a tractogram from a LazyTractogram object.
        tractogram = LazyTractogram(
            DATA['streamlines_func'], DATA['data_per_streamline_func'], DATA['data_per_point_func']
        )

        tractogram2 = Tractogram(
            tractogram.streamlines, tractogram.data_per_streamline, tractogram.data_per_point
        )

        # Inconsistent number of scalars between streamlines
        wrong_data = [[(1, 0, 0)] * 1, [(0, 1, 0), (0, 1)], [(0, 0, 1)] * 5]

        data_per_point = {'wrong_data': wrong_data}
        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_point=data_per_point)

        # Inconsistent number of scalars between streamlines
        wrong_data = [[(1, 0, 0)] * 1, [(0, 1)] * 2, [(0, 0, 1)] * 5]

        data_per_point = {'wrong_data': wrong_data}
        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_point=data_per_point)

    def test_setting_affine_to_rasmm(self):
        tractogram = DATA['tractogram'].copy()
        affine = np.diag(range(4))

        # Test assigning None.
        tractogram.affine_to_rasmm = None
        assert tractogram.affine_to_rasmm is None

        # Test assigning a valid ndarray (should make a copy).
        tractogram.affine_to_rasmm = affine
        assert tractogram.affine_to_rasmm is not affine

        # Test assigning a list of lists.
        tractogram.affine_to_rasmm = affine.tolist()
        assert_array_equal(tractogram.affine_to_rasmm, affine)

        # Test assigning a ndarray with wrong shape.
        with pytest.raises(ValueError):
            tractogram.affine_to_rasmm = affine[::2]

    def test_tractogram_getitem(self):
        # Retrieve TractogramItem by their index.
        for i, t in enumerate(DATA['tractogram']):
            assert_tractogram_item_equal(DATA['tractogram'][i], t)

        # Get one TractogramItem out of two.
        tractogram_view = DATA['simple_tractogram'][::2]
        check_tractogram(tractogram_view, DATA['streamlines'][::2])

        # Use slicing.
        r_tractogram = DATA['tractogram'][::-1]
        check_tractogram(
            r_tractogram,
            DATA['streamlines'][::-1],
            DATA['tractogram'].data_per_streamline[::-1],
            DATA['tractogram'].data_per_point[::-1],
        )

        # Make sure slicing conserves the affine_to_rasmm property.
        tractogram = DATA['tractogram'].copy()
        tractogram.affine_to_rasmm = DATA['rng'].rand(4, 4)
        tractogram_view = tractogram[::2]
        assert_array_equal(tractogram_view.affine_to_rasmm, tractogram.affine_to_rasmm)

    def test_tractogram_add_new_data(self):
        # Tractogram with only streamlines
        t = DATA['simple_tractogram'].copy()
        t.data_per_point['fa'] = DATA['fa']
        t.data_per_point['colors'] = DATA['colors']
        t.data_per_streamline['mean_curvature'] = DATA['mean_curvature']
        t.data_per_streamline['mean_torsion'] = DATA['mean_torsion']
        t.data_per_streamline['mean_colors'] = DATA['mean_colors']
        t.data_per_streamline['clusters_labels'] = DATA['clusters_labels']
        assert_tractogram_equal(t, DATA['tractogram'])

        # Retrieve tractogram by their index.
        for i, item in enumerate(t):
            assert_tractogram_item_equal(t[i], item)

        # Use slicing.
        r_tractogram = t[::-1]
        check_tractogram(
            r_tractogram, t.streamlines[::-1], t.data_per_streamline[::-1], t.data_per_point[::-1]
        )

        # Add new data to a tractogram for which its `streamlines` is a view.
        t = Tractogram(DATA['streamlines'] * 2, affine_to_rasmm=np.eye(4))
        t = t[: len(DATA['streamlines'])]  # Create a view of `streamlines`
        t.data_per_point['fa'] = DATA['fa']
        t.data_per_point['colors'] = DATA['colors']
        t.data_per_streamline['mean_curvature'] = DATA['mean_curvature']
        t.data_per_streamline['mean_torsion'] = DATA['mean_torsion']
        t.data_per_streamline['mean_colors'] = DATA['mean_colors']
        t.data_per_streamline['clusters_labels'] = DATA['clusters_labels']
        assert_tractogram_equal(t, DATA['tractogram'])

    def test_tractogram_copy(self):
        # Create a copy of a tractogram.
        tractogram = DATA['tractogram'].copy()

        # Check we copied the data and not simply created new references.
        assert tractogram is not DATA['tractogram']
        assert tractogram.streamlines is not DATA['tractogram'].streamlines
        assert tractogram.data_per_streamline is not DATA['tractogram'].data_per_streamline
        assert tractogram.data_per_point is not DATA['tractogram'].data_per_point

        for key in tractogram.data_per_streamline:
            assert (
                tractogram.data_per_streamline[key]
                is not DATA['tractogram'].data_per_streamline[key]
            )

        for key in tractogram.data_per_point:
            assert tractogram.data_per_point[key] is not DATA['tractogram'].data_per_point[key]

        # Check the values of the data are the same.
        assert_tractogram_equal(tractogram, DATA['tractogram'])

    def test_creating_invalid_tractogram(self):
        # Not enough data_per_point for all the points of all streamlines.
        scalars = [
            [(1, 0, 0)] * 1,
            [(0, 1, 0)] * 2,
            [(0, 0, 1)] * 3,
        ]  # Last streamlines has 5 points.

        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_point={'scalars': scalars})

        # Not enough data_per_streamline for all streamlines.
        properties = [np.array([1.11, 1.22], dtype='f4'), np.array([3.11, 3.22], dtype='f4')]

        with pytest.raises(ValueError):
            Tractogram(
                streamlines=DATA['streamlines'], data_per_streamline={'properties': properties}
            )

        # Inconsistent dimension for a data_per_point.
        scalars = [[(1, 0, 0)] * 1, [(0, 1)] * 2, [(0, 0, 1)] * 5]

        with pytest.raises(ValueError):
            Tractogram(streamlines=DATA['streamlines'], data_per_point={'scalars': scalars})

        # Too many dimension for a data_per_streamline.
        properties = [
            np.array([[1.11], [1.22]], dtype='f4'),
            np.array([[2.11], [2.22]], dtype='f4'),
            np.array([[3.11], [3.22]], dtype='f4'),
        ]

        with pytest.raises(ValueError):
            Tractogram(
                streamlines=DATA['streamlines'], data_per_streamline={'properties': properties}
            )

    def test_tractogram_apply_affine(self):
        tractogram = DATA['tractogram'].copy()
        affine = np.eye(4)
        scaling = np.array((1, 2, 3), dtype=float)
        affine[range(3), range(3)] = scaling

        # Apply the affine to the streamline in a lazy manner.
        transformed_tractogram = tractogram.apply_affine(affine, lazy=True)
        assert type(transformed_tractogram) is LazyTractogram
        check_tractogram(
            transformed_tractogram,
            streamlines=[s * scaling for s in DATA['streamlines']],
            data_per_streamline=DATA['data_per_streamline'],
            data_per_point=DATA['data_per_point'],
        )
        assert_array_equal(
            transformed_tractogram.affine_to_rasmm, np.dot(np.eye(4), np.linalg.inv(affine))
        )
        # Make sure streamlines of the original tractogram have not been
        # modified.
        assert_arrays_equal(tractogram.streamlines, DATA['streamlines'])

        # Apply the affine to the streamlines in-place.
        transformed_tractogram = tractogram.apply_affine(affine)
        assert transformed_tractogram is tractogram
        check_tractogram(
            tractogram,
            streamlines=[s * scaling for s in DATA['streamlines']],
            data_per_streamline=DATA['data_per_streamline'],
            data_per_point=DATA['data_per_point'],
        )

        # Apply affine again and check the affine_to_rasmm.
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(
            transformed_tractogram.affine_to_rasmm,
            np.dot(np.eye(4), np.dot(np.linalg.inv(affine), np.linalg.inv(affine))),
        )

        # Applying the affine to a tractogram that has been indexed or sliced
        # shouldn't affect the remaining streamlines.
        tractogram = DATA['tractogram'].copy()
        transformed_tractogram = tractogram[::2].apply_affine(affine)
        assert transformed_tractogram is not tractogram
        check_tractogram(
            tractogram[::2],
            streamlines=[s * scaling for s in DATA['streamlines'][::2]],
            data_per_streamline=DATA['tractogram'].data_per_streamline[::2],
            data_per_point=DATA['tractogram'].data_per_point[::2],
        )

        # Remaining streamlines should match the original ones.
        check_tractogram(
            tractogram[1::2],
            streamlines=DATA['streamlines'][1::2],
            data_per_streamline=DATA['tractogram'].data_per_streamline[1::2],
            data_per_point=DATA['tractogram'].data_per_point[1::2],
        )

        # Check that applying an affine and its inverse give us back the
        # original streamlines.
        tractogram = DATA['tractogram'].copy()
        affine = np.random.RandomState(1234).randn(4, 4)
        affine[-1] = [0, 0, 0, 1]  # Remove perspective projection.

        tractogram.apply_affine(affine)
        tractogram.apply_affine(np.linalg.inv(affine))
        assert_array_almost_equal(tractogram.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Test applying the identity transformation.
        tractogram = DATA['tractogram'].copy()
        tractogram.apply_affine(np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Test removing affine_to_rasmm
        tractogram = DATA['tractogram'].copy()
        tractogram.affine_to_rasmm = None
        tractogram.apply_affine(affine)
        assert tractogram.affine_to_rasmm is None

    def test_tractogram_to_world(self):
        tractogram = DATA['tractogram'].copy()
        affine = np.random.RandomState(1234).randn(4, 4)
        affine[-1] = [0, 0, 0, 1]  # Remove perspective projection.

        # Apply the affine to the streamlines, then bring them back
        # to world space in a lazy manner.
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram.affine_to_rasmm, np.linalg.inv(affine))

        tractogram_world = transformed_tractogram.to_world(lazy=True)
        assert type(tractogram_world) is LazyTractogram
        assert_array_almost_equal(tractogram_world.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Bring them back streamlines to world space in a in-place manner.
        tractogram_world = transformed_tractogram.to_world()
        assert tractogram_world is tractogram
        assert_array_almost_equal(tractogram.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Calling to_world twice should do nothing.
        tractogram_world2 = transformed_tractogram.to_world()
        assert tractogram_world2 is tractogram
        assert_array_almost_equal(tractogram.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Calling to_world when affine_to_rasmm is None should fail.
        tractogram = DATA['tractogram'].copy()
        tractogram.affine_to_rasmm = None
        with pytest.raises(ValueError):
            tractogram.to_world()

    def test_tractogram_extend(self):
        # Load tractogram that contains some metadata.
        t = DATA['tractogram'].copy()

        for op, in_place in ((operator.add, False), (operator.iadd, True), (extender, True)):
            first_arg = t.copy()
            new_t = op(first_arg, t)
            assert (new_t is first_arg) == in_place
            assert_tractogram_equal(new_t[: len(t)], DATA['tractogram'])
            assert_tractogram_equal(new_t[len(t) :], DATA['tractogram'])

        # Test extending an empty Tractogram.
        t = Tractogram()
        t += DATA['tractogram']
        assert_tractogram_equal(t, DATA['tractogram'])

        # and the other way around.
        t = DATA['tractogram'].copy()
        t += Tractogram()
        assert_tractogram_equal(t, DATA['tractogram'])


class TestLazyTractogram(unittest.TestCase):
    def test_lazy_tractogram_creation(self):
        # To create tractogram from arrays use `Tractogram`.
        with pytest.raises(TypeError):
            LazyTractogram(streamlines=DATA['streamlines'])

        # Streamlines and other data as generators
        streamlines = (x for x in DATA['streamlines'])
        data_per_point = {'colors': (x for x in DATA['colors'])}
        data_per_streamline = {
            'torsion': (x for x in DATA['mean_torsion']),
            'colors': (x for x in DATA['mean_colors']),
        }

        # Creating LazyTractogram with generators is not allowed as
        # generators get exhausted and are not reusable unlike generator
        # function.
        with pytest.raises(TypeError):
            LazyTractogram(streamlines=streamlines)
        with pytest.raises(TypeError):
            LazyTractogram(data_per_point={'none': None})
        with pytest.raises(TypeError):
            LazyTractogram(data_per_streamline=data_per_streamline)
        with pytest.raises(TypeError):
            LazyTractogram(streamlines=DATA['streamlines'], data_per_point=data_per_point)

        # Empty `LazyTractogram`
        tractogram = LazyTractogram()
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            check_tractogram(tractogram)
        assert tractogram.affine_to_rasmm is None

        # Create tractogram with streamlines and other data
        tractogram = LazyTractogram(
            DATA['streamlines_func'], DATA['data_per_streamline_func'], DATA['data_per_point_func']
        )

        assert is_lazy_dict(tractogram.data_per_streamline)
        assert is_lazy_dict(tractogram.data_per_point)

        [t for t in tractogram]  # Force iteration through tractogram.
        assert len(tractogram) == len(DATA['streamlines'])

        # Generator functions get re-called and creates new iterators.
        for i in range(2):
            assert_tractogram_equal(tractogram, DATA['tractogram'])

    def test_lazy_tractogram_from_data_func(self):
        # Create an empty `LazyTractogram` yielding nothing.
        tractogram = LazyTractogram.from_data_func(lambda: iter([]))
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            check_tractogram(tractogram)

        # Create `LazyTractogram` from a generator function yielding
        # TractogramItem.
        data = [
            DATA['streamlines'],
            DATA['fa'],
            DATA['colors'],
            DATA['mean_curvature'],
            DATA['mean_torsion'],
            DATA['mean_colors'],
            DATA['clusters_labels'],
        ]

        def _data_gen():
            for d in zip(*data):
                data_for_points = {'fa': d[1], 'colors': d[2]}
                data_for_streamline = {
                    'mean_curvature': d[3],
                    'mean_torsion': d[4],
                    'mean_colors': d[5],
                    'clusters_labels': d[6],
                }
                yield TractogramItem(d[0], data_for_streamline, data_for_points)

        tractogram = LazyTractogram.from_data_func(_data_gen)
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            assert_tractogram_equal(tractogram, DATA['tractogram'])

        # Creating a LazyTractogram from not a corouting should raise an error.
        with pytest.raises(TypeError):
            LazyTractogram.from_data_func(_data_gen())

    def test_lazy_tractogram_getitem(self):
        with pytest.raises(NotImplementedError):
            DATA['lazy_tractogram'][0]

    def test_lazy_tractogram_extend(self):
        t = DATA['lazy_tractogram'].copy()
        new_t = DATA['lazy_tractogram'].copy()

        for op in (operator.add, operator.iadd, extender):
            with pytest.raises(NotImplementedError):
                op(new_t, t)

    def test_lazy_tractogram_len(self):
        modules = [module_tractogram]  # Modules for which to catch warnings.
        with clear_and_catch_warnings(record=True, modules=modules) as w:
            warnings.simplefilter('always')  # Always trigger warnings.

            # Calling `len` will create new generators each time.
            tractogram = LazyTractogram(DATA['streamlines_func'])
            assert tractogram._nb_streamlines is None

            # This should produce a warning message.
            assert len(tractogram) == len(DATA['streamlines'])
            assert tractogram._nb_streamlines == len(DATA['streamlines'])
            assert len(w) == 1

            tractogram = LazyTractogram(DATA['streamlines_func'])

            # New instances should still produce a warning message.
            assert len(tractogram) == len(DATA['streamlines'])
            assert len(w) == 2
            assert issubclass(w[-1].category, Warning) is True

            # Calling again 'len' again should *not* produce a warning.
            assert len(tractogram) == len(DATA['streamlines'])
            assert len(w) == 2

        with clear_and_catch_warnings(record=True, modules=modules) as w:
            # Once we iterated through the tractogram, we know the length.

            tractogram = LazyTractogram(DATA['streamlines_func'])

            assert tractogram._nb_streamlines is None
            [t for t in tractogram]  # Force iteration through tractogram.
            assert tractogram._nb_streamlines == len(DATA['streamlines'])
            # This should *not* produce a warning.
            assert len(tractogram) == len(DATA['streamlines'])
            assert len(w) == 0

    def test_lazy_tractogram_apply_affine(self):
        affine = np.eye(4)
        scaling = np.array((1, 2, 3), dtype=float)
        affine[range(3), range(3)] = scaling

        tractogram = DATA['lazy_tractogram'].copy()

        transformed_tractogram = tractogram.apply_affine(affine)
        assert transformed_tractogram is not tractogram
        assert_array_equal(tractogram._affine_to_apply, np.eye(4))
        assert_array_equal(tractogram.affine_to_rasmm, np.eye(4))
        assert_array_equal(transformed_tractogram._affine_to_apply, affine)
        assert_array_equal(
            transformed_tractogram.affine_to_rasmm, np.dot(np.eye(4), np.linalg.inv(affine))
        )
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            check_tractogram(
                transformed_tractogram,
                streamlines=[s * scaling for s in DATA['streamlines']],
                data_per_streamline=DATA['data_per_streamline'],
                data_per_point=DATA['data_per_point'],
            )

        # Apply affine again and check the affine_to_rasmm.
        transformed_tractogram = transformed_tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram._affine_to_apply, np.dot(affine, affine))
        assert_array_equal(
            transformed_tractogram.affine_to_rasmm,
            np.dot(np.eye(4), np.dot(np.linalg.inv(affine), np.linalg.inv(affine))),
        )

        # Calling to_world when affine_to_rasmm is None should fail.
        tractogram = DATA['lazy_tractogram'].copy()
        tractogram.affine_to_rasmm = None
        with pytest.raises(ValueError):
            tractogram.to_world()

        # But calling apply_affine when affine_to_rasmm is None should work.
        tractogram = DATA['lazy_tractogram'].copy()
        tractogram.affine_to_rasmm = None
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram._affine_to_apply, affine)
        assert transformed_tractogram.affine_to_rasmm is None
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            check_tractogram(
                transformed_tractogram,
                streamlines=[s * scaling for s in DATA['streamlines']],
                data_per_streamline=DATA['data_per_streamline'],
                data_per_point=DATA['data_per_point'],
            )

        # Calling apply_affine with lazy=False should fail for LazyTractogram.
        tractogram = DATA['lazy_tractogram'].copy()
        with pytest.raises(ValueError):
            tractogram.apply_affine(affine=np.eye(4), lazy=False)

    def test_tractogram_to_world(self):
        tractogram = DATA['lazy_tractogram'].copy()
        affine = np.random.RandomState(1234).randn(4, 4)
        affine[-1] = [0, 0, 0, 1]  # Remove perspective projection.

        # Apply the affine to the streamlines, then bring them back
        # to world space in a lazy manner.
        transformed_tractogram = tractogram.apply_affine(affine)
        assert_array_equal(transformed_tractogram.affine_to_rasmm, np.linalg.inv(affine))

        tractogram_world = transformed_tractogram.to_world()
        assert tractogram_world is not transformed_tractogram
        assert_array_almost_equal(tractogram_world.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Calling to_world twice should do nothing.
        tractogram_world = tractogram_world.to_world()
        assert_array_almost_equal(tractogram_world.affine_to_rasmm, np.eye(4))
        for s1, s2 in zip(tractogram_world.streamlines, DATA['streamlines']):
            assert_array_almost_equal(s1, s2)

        # Calling to_world when affine_to_rasmm is None should fail.
        tractogram = DATA['lazy_tractogram'].copy()
        tractogram.affine_to_rasmm = None
        with pytest.raises(ValueError):
            tractogram.to_world()

    def test_lazy_tractogram_copy(self):
        # Create a copy of the lazy tractogram.
        tractogram = DATA['lazy_tractogram'].copy()

        # Check we copied the data and not simply created new references.
        assert tractogram is not DATA['lazy_tractogram']

        # When copying LazyTractogram, the generator function yielding
        # streamlines should stay the same.
        assert tractogram._streamlines is DATA['lazy_tractogram']._streamlines

        # Copying LazyTractogram, creates new internal LazyDict objects,
        # but generator functions contained in it should stay the same.
        assert tractogram._data_per_streamline is not DATA['lazy_tractogram']._data_per_streamline
        assert tractogram._data_per_point is not DATA['lazy_tractogram']._data_per_point

        for key in tractogram.data_per_streamline:
            data = tractogram.data_per_streamline.store[key]
            expected = DATA['lazy_tractogram'].data_per_streamline.store[key]
            assert data is expected

        for key in tractogram.data_per_point:
            data = tractogram.data_per_point.store[key]
            expected = DATA['lazy_tractogram'].data_per_point.store[key]
            assert data is expected

        # The affine should be a copy.
        assert tractogram._affine_to_apply is not DATA['lazy_tractogram']._affine_to_apply
        assert_array_equal(tractogram._affine_to_apply, DATA['lazy_tractogram']._affine_to_apply)

        # Check the data are the equivalent.
        with pytest.warns(Warning, match='Number of streamlines will be determined manually'):
            assert_tractogram_equal(tractogram, DATA['tractogram'])
