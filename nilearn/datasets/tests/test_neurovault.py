"""
Test the neurovault module.
"""
# Author: Jerome Dockes
# License: simplified BSD

import os
import warnings
import stat
try:
    from os.path import samefile
except ImportError:
    # os.path.samefile not available on windows
    samefile = None
import tempfile
import shutil
import json
import re

import numpy as np
from nose import SkipTest
from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)
try:
    from nose.tools import assert_warns
except ImportError:
    assert_warns = None

from nilearn.datasets import neurovault


def _same_stat(path_1, path_2):
    path_1 = os.path.abspath(os.path.expanduser(path_1))
    path_2 = os.path.abspath(os.path.expanduser(path_2))
    return os.stat(path_1) == os.stat(path_2)


if samefile is None:
    samefile = _same_stat


def _verif_warn(warning_type, f, *args, **kwargs):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        f(*args, **kwargs)
        try:
            assert issubclass(w[0].category, warning_type)
        except IndexError:
            raise AssertionError('Warning not raised by {0}'.format(f))


if assert_warns is None:
    assert_warns = _verif_warn


_EXAMPLE_IM_META = {
    "analysis_level": None,
    "file": "http://neurovault.org/media/images/35/Fig3B_zstat1.nii.gz",
    "cognitive_contrast_cogatlas_id": None, "statistic_parameters": None,
    "is_valid": True,
    "thumbnail": "http://neurovault.org/media/images/35/glass_brain_110_1.jpg",
    "file_size": 742636, "map_type": "Z map",
    "collection": "http://neurovault.org/collections/35/",
    "brain_coverage": 82.5116091788011, "collection_id": 35,
    "contrast_definition": "high value go minus no go at probe",
    "is_thresholded": False, "cognitive_contrast_cogatlas": None,
    "name": "Fig3B_zstat1.nii.gz",
    "cognitive_paradigm_cogatlas_id": "trm_553e77e53497d",
    "description": "The difference in the parametric effect of...",
    "figure": "3B", "cognitive_paradigm_cogatlas": "cue approach task",
    "add_date": "2016-01-21T17:23:16.733390Z",
    "modify_date": "2016-01-27T21:47:42.236081Z",
    "modality": "fMRI-BOLD", "contrast_definition_cogatlas": "",
    "number_of_subjects": 21, "id": 110, "image_type": "statistic_map",
    "perc_bad_voxels": 78.4232503054965,
    "url": "http://neurovault.org/images/110/",
    "perc_voxels_outside": 3.20038201254891, "not_mni": False,
    "smoothness_fwhm": None,
    "reduced_representation":
    "http://neurovault.org/media/images/35/transform_4mm_110.npy"}

_EXAMPLE_COL_META = {
    "owner": 52, "number_of_experimental_units": None,
    "full_dataset_url": None,
    "add_date": "2014-03-25T20:52:35.182187Z", "nonlinear_transform_type": "",
    "used_slice_timing_correction": None, "software_package": "",
    "group_model_type": "", "coordinate_space": None,
    "intersubject_registration_software": "", "matrix_size": None,
    "functional_coregistration_method": "",
    "journal_name": "Nature Neuroscience",
    "used_motion_susceptibiity_correction": None,
    "motion_correction_interpolation": "", "subject_age_mean": None,
    "used_reaction_time_regressor": None, "object_image_type": "",
    "software_version": "", "used_intersubject_registration": None,
    "subject_age_min": None, "optimization": None, "number_of_images": 4,
    "scanner_model": "Skyra", "length_of_runs": None, "subject_age_max": None,
    "echo_time": None,
    "slice_thickness": None, "motion_correction_software": "",
    "DOI": "10.1038/nn.3673", "number_of_imaging_runs": None,
    "used_temporal_derivatives": None, "used_smoothing": None,
    "group_repeated_measures_method": "", "resampled_voxel_size": None,
    "length_of_trials": None, "authors": "Tom Schonberg, Akram Bakkour,...",
    "id": 35,
    "order_of_preprocessing_operations": "", "skip_distance": None,
    "smoothing_type": "", "modify_date": "2016-01-27T21:47:42.229891Z",
    "group_model_multilevel": "",
    "functional_coregistered_to_structural": None, "flip_angle": None,
    "interpolation_method": "", "group_estimation_type": "",
    "group_inference_type": None, "length_of_blocks": None,
    "used_b0_unwarping": None, "intersubject_transformation_type": None,
    "description": "", "scanner_make": "Siemens",
    "intrasubject_estimation_type": "", "order_of_acquisition": None,
    "proportion_male_subjects": None, "group_repeated_measures": None,
    "url": "http://neurovault.org/collections/35/", "handedness": "right",
    "used_motion_regressors": None, "used_motion_correction": None,
    "motion_correction_reference": "", "orthogonalization_description": "",
    "hemodynamic_response_function": "", "autocorrelation_model": "",
    "used_orthogonalization": None, "acquisition_orientation": "",
    "intrasubject_model_type": "", "target_resolution": None,
    "intrasubject_modeling_software": "", "used_dispersion_derivatives": None,
    "field_strength": 3.0, "transform_similarity_metric": "",
    "field_of_view": None, "group_comparison": None,
    "number_of_rejected_subjects": None,
    "name": "Changing value through cued approach: ...",
    "quality_control": "", "type_of_design": "eventrelated",
    "used_high_pass_filter": None, "pulse_sequence": "Multiband",
    "owner_name": "tomtom", "motion_correction_metric": "",
    "parallel_imaging": "", "optimization_method": "",
    "target_template_image": "", "repetition_time": None,
    "high_pass_filter_method": "", "contributors": "",
    "inclusion_exclusion_criteria": "", "b0_unwarping_software": "",
    "paper_url": "http://www.nature.com/doifinder/10.1038/nn.3673",
    "group_modeling_software": "",
    "doi_add_date": "2014-03-25T20:52:35.182187Z",
    "slice_timing_correction_software": "", "group_description": "",
    "smoothing_fwhm": None}


neurovault.set_logging_level(neurovault.logging.DEBUG)


class _TemporaryDirectory(object):
    def __init__(self, set_nv_dir=True):
        self.set_nv_dir_ = set_nv_dir

    def __enter__(self):
        self.temp_dir_ = tempfile.mkdtemp()
        if self.set_nv_dir_:
            neurovault.set_neurovault_directory(self.temp_dir_)
        return self.temp_dir_

    def __exit__(self, *args):
        if self.set_nv_dir_:
            neurovault.set_neurovault_directory(None)
        os.chmod(self.temp_dir_, stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
        for root, dirnames, filenames in os.walk(self.temp_dir_):
            for name in dirnames:
                os.chmod(os.path.join(root, name),
                         stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
            for name in filenames:
                os.chmod(os.path.join(root, name),
                         stat.S_IWUSR | stat.S_IRUSR)
        shutil.rmtree(self.temp_dir_)


def test_translate_types_to_sql():
    py_types = {'some_int': int, 'some_float': float,
                'some_str': str, 'some_bool': bool, 'some_dict': dict}
    sql_types = neurovault._translate_types_to_sql(py_types)
    assert_equal(sql_types['some_int'], 'INTEGER')
    assert_equal(sql_types['some_float'], 'REAL')
    assert_equal(sql_types['some_str'], 'TEXT')
    assert_equal(sql_types['some_bool'], 'INTEGER')
    assert_equal(sql_types['some_dict'], '')


def test_to_supported_type():
    assert_equal(neurovault._to_supported_type(0), 0)
    assert_equal(neurovault._to_supported_type(None), None)
    assert_equal(neurovault._to_supported_type('abc'), 'abc')
    assert_equal(neurovault._to_supported_type({'a': 0}), json.dumps({'a': 0}))
    assert_equal(neurovault._to_supported_type(u'\u2019'), u'\u2019')
    assert_equal(neurovault._to_supported_type(os), u'{0}'.format(os))


def test_remove_none_strings():
    info = {'a': 'None / Other',
            'b': '',
            'c': 'N/A',
            'd': None,
            'e': 0,
            'f': 'a',
            'g': 'Name'}
    assert_equal(neurovault._remove_none_strings(info),
                 {'a': None,
                  'b': None,
                  'c': None,
                  'd': None,
                  'e': 0,
                  'f': 'a',
                  'g': 'Name'})


def test_append_filters_to_query():
    query = neurovault._append_filters_to_query(
        neurovault._NEUROVAULT_COLLECTIONS_URL,
        {'DOI': 17})
    assert_equal(
        query, 'http://neurovault.org/api/collections/?DOI=17')
    query = neurovault._append_filters_to_query(
        neurovault._NEUROVAULT_COLLECTIONS_URL,
        {'id': 40})
    assert_equal(query, 'http://neurovault.org/api/collections/40')


def ignore_connection_errors(func):
    def decorate(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except neurovault.URLError:
            raise SkipTest('connection problem')

    return decorate


@ignore_connection_errors
def test_get_encoding():
    request = neurovault.Request('http://www.google.com')
    opener = neurovault.build_opener()
    try:
        response = opener.open(request)
    except Exception:
        return
    try:
        neurovault._get_encoding(response)
    finally:
        response.close()


@ignore_connection_errors
def test_get_batch():
    batch = neurovault._get_batch(neurovault._NEUROVAULT_COLLECTIONS_URL)
    assert('results' in batch)
    assert('count' in batch)
    assert_raises(neurovault.URLError, neurovault._get_batch, 'http://')
    with _TemporaryDirectory(set_nv_dir=False) as temp_dir:
        with open(os.path.join(temp_dir, 'test_nv.txt'), 'w'):
            pass
        assert_raises(ValueError, neurovault._get_batch, 'file://{0}'.format(
            os.path.join(temp_dir, 'test_nv.txt')))


@ignore_connection_errors
def test_scroll_server_results():
    result = list(neurovault._scroll_server_results(
        neurovault._NEUROVAULT_COLLECTIONS_URL, max_results=6, batch_size=3))
    assert_equal(len(result), 6)
    result = list(neurovault._scroll_server_results(
        neurovault._NEUROVAULT_COLLECTIONS_URL, max_results=3,
        local_filter=lambda r: False))
    assert_equal(len(result), 0)


def test_is_null():
    is_null = neurovault.IsNull()
    assert_true(is_null != 'a')
    assert_false(is_null != '')
    assert_true('a' != is_null)
    assert_false('' != is_null)
    assert_false(is_null == 'a')
    assert_true(is_null == '')
    assert_false('a' == is_null)
    assert_true('' == is_null)
    assert_equal(str(is_null), 'IsNull()')


def test_not_null():
    not_null = neurovault.NotNull()
    assert_true(not_null == 'a')
    assert_false(not_null == '')
    assert_true('a' == not_null)
    assert_false('' == not_null)
    assert_false(not_null != 'a')
    assert_true(not_null != '')
    assert_false('a' != not_null)
    assert_true('' != not_null)
    assert_equal(str(not_null), 'NotNull()')


def test_not_equal():
    not_equal = neurovault.NotEqual('a')
    assert_true(not_equal == 'b')
    assert_true(not_equal == 1)
    assert_false(not_equal == 'a')
    assert_true('b' == not_equal)
    assert_true(1 == not_equal)
    assert_false('a' == not_equal)
    assert_false(not_equal != 'b')
    assert_false(not_equal != 1)
    assert_true(not_equal != 'a')
    assert_false('b' != not_equal)
    assert_false(1 != not_equal)
    assert_true('a' != not_equal)
    assert_equal(str(not_equal), "NotEqual('a')")


def test_order_comp():
    geq = neurovault.GreaterOrEqual('2016-07-12T11:29:12.263046Z')
    assert_true('2016-08-12T11:29:12.263046Z' == geq)
    assert_true('2016-06-12T11:29:12.263046Z' != geq)
    assert_equal(str(geq), "GreaterOrEqual('2016-07-12T11:29:12.263046Z')")
    gt = neurovault.GreaterThan('abc')
    assert_false(gt == 'abc')
    assert_true(gt == 'abd')
    assert_equal(str(gt), "GreaterThan('abc')")
    lt = neurovault.LessThan(7)
    assert_false(7 == lt)
    assert_false(5 != lt)
    assert_false(lt == 'a')
    assert_equal(str(lt), 'LessThan(7)')
    leq = neurovault.LessOrEqual(4.5)
    assert_true(4.4 == leq)
    assert_false(4.6 == leq)
    assert_equal(str(leq), 'LessOrEqual(4.5)')


def test_is_in():
    is_in = neurovault.IsIn(0, 1)
    assert_true(is_in == 0)
    assert_false(is_in == 2)
    assert_true(0 == is_in)
    assert_false(2 == is_in)
    assert_false(is_in != 0)
    assert_true(is_in != 2)
    assert_false(0 != is_in)
    assert_true(2 != is_in)
    assert_equal(str(is_in), 'IsIn(0, 1)')
    countable = neurovault.IsIn(*range(11))
    assert_true(7 == countable)
    assert_false(countable == 12)


def test_not_in():
    not_in = neurovault.NotIn(0, 1)
    assert_true(not_in != 0)
    assert_false(not_in != 2)
    assert_true(0 != not_in)
    assert_false(2 != not_in)
    assert_false(not_in == 0)
    assert_true(not_in == 2)
    assert_false(0 == not_in)
    assert_true(2 == not_in)
    assert_equal(str(not_in), 'NotIn(0, 1)')


def test_contains():
    contains = neurovault.Contains('a', 0)
    assert_false(contains == 10)
    assert_true(contains == ['b', 1, 'a', 0])
    assert_true(['b', 1, 'a', 0] == contains)
    assert_true(contains != ['b', 1, 0])
    assert_true(['b', 1, 'a'] != contains)
    assert_false(contains != ['b', 1, 'a', 0])
    assert_false(['b', 1, 'a', 0] != contains)
    assert_false(contains == ['b', 1, 0])
    assert_false(['b', 1, 'a'] == contains)
    assert_equal(str(contains), "Contains('a', 0)")
    contains = neurovault.Contains('house', 'face')
    assert_true('face vs house' == contains)
    assert_false('smiling face vs frowning face' == contains)


def test_not_contains():
    not_contains = neurovault.NotContains('ab')
    assert_true(None != not_contains)
    assert_true(not_contains == 'a_b')
    assert_true('bcd' == not_contains)
    assert_true(not_contains != '_abcd')
    assert_true('_abcd' != not_contains)
    assert_false(not_contains != 'a_b')
    assert_false('bcd' != not_contains)
    assert_false(not_contains == '_abcd')
    assert_false('_abcd' == not_contains)
    assert_equal(str(not_contains), "NotContains('ab',)")


def test_pattern():
    # Python std lib doc poker hand example
    pattern_0 = neurovault.Pattern(r'[0-9akqj]{5}$')
    assert_equal(str(pattern_0), "Pattern(pattern='[0-9akqj]{5}$', flags=0)")
    pattern_1 = neurovault.Pattern(r'[0-9akqj]{5}$', re.I)
    assert_true(pattern_0 == 'ak05q')
    assert_false(pattern_0 == 'Ak05q')
    assert_false(pattern_0 == 'ak05e')
    assert_true(pattern_1 == 'ak05q')
    assert_true(pattern_1 == 'Ak05q')
    assert_false(pattern_1 == 'ak05e')
    assert_false(pattern_0 != 'ak05q')
    assert_true(pattern_0 != 'Ak05q')
    assert_true(pattern_0 != 'ak05e')
    assert_false(pattern_1 != 'ak05q')
    assert_false(pattern_1 != 'Ak05q')
    assert_true(pattern_1 != 'ak05e')

    assert_true('ak05q' == pattern_0)
    assert_false('Ak05q' == pattern_0)
    assert_false('ak05e' == pattern_0)
    assert_true('ak05q' == pattern_1)
    assert_true('Ak05q' == pattern_1)
    assert_false('ak05e' == pattern_1)
    assert_false('ak05q' != pattern_0)
    assert_true('Ak05q' != pattern_0)
    assert_true('ak05e' != pattern_0)
    assert_false('ak05q' != pattern_1)
    assert_false('Ak05q' != pattern_1)
    assert_true('ak05e' != pattern_1)


def test_result_filter():
    filter_0 = neurovault.ResultFilter(query_terms={'a': 0},
                                       callable_filter=lambda d: len(d) < 5,
                                       b=1)
    assert_equal(filter_0['a'], 0)
    assert_true(filter_0({'a': 0, 'b': 1, 'c': 2}))
    assert_false(filter_0({'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4}))
    assert_false(filter_0({'b': 1, 'c': 2, 'd': 3}))
    assert_false(filter_0({'a': 1, 'b': 1, 'c': 2}))

    filter_1 = neurovault.ResultFilter(query_terms={'c': 2})
    filter_1['d'] = neurovault.NotNull()
    assert_true(filter_1({'c': 2, 'd': 1}))
    assert_false(filter_1({'c': 2, 'd': 0}))
    filter_1['d'] = neurovault.IsIn(0, 1)
    assert_true(filter_1({'c': 2, 'd': 1}))
    assert_false(filter_1({'c': 2, 'd': 2}))
    del filter_1['d']
    assert_true(filter_1({'c': 2, 'd': 2}))
    filter_1['d'] = neurovault.NotIn(0, 1)
    assert_false(filter_1({'c': 2, 'd': 1}))
    assert_true(filter_1({'c': 2, 'd': 3}))
    filter_1.add_filter(lambda d: len(d) > 2)
    assert_false(filter_1({'c': 2, 'd': 3}))
    assert_true(filter_1({'c': 2, 'd': 3, 'e': 4}))


def test_result_filter_combinations():
    filter_0 = neurovault.ResultFilter(a=0, b=1)
    filter_1 = neurovault.ResultFilter(c=2, d=3)

    filter_0_and_1 = filter_0.AND(filter_1)
    assert_true(filter_0_and_1({'a': 0, 'b': 1, 'c': 2, 'd': 3}))
    assert_false(filter_0_and_1({'a': 0, 'b': 1, 'c': 2, 'd': None}))
    assert_false(filter_0_and_1({'a': None, 'b': 1, 'c': 2, 'd': 3}))

    filter_0_or_1 = filter_0.OR(filter_1)
    assert_true(filter_0_or_1({'a': 0, 'b': 1, 'c': 2, 'd': 3}))
    assert_true(filter_0_or_1({'a': 0, 'b': 1, 'c': 2, 'd': None}))
    assert_true(filter_0_or_1({'a': None, 'b': 1, 'c': 2, 'd': 3}))
    assert_false(filter_0_or_1({'a': None, 'b': 1, 'c': 2, 'd': None}))

    filter_0_xor_1 = filter_0.XOR(filter_1)
    assert_false(filter_0_xor_1({'a': 0, 'b': 1, 'c': 2, 'd': 3}))
    assert_true(filter_0_xor_1({'a': 0, 'b': 1, 'c': 2, 'd': None}))
    assert_true(filter_0_xor_1({'a': None, 'b': 1, 'c': 2, 'd': 3}))
    assert_false(filter_0_xor_1({'a': None, 'b': 1, 'c': 2, 'd': None}))

    not_filter_0 = filter_0.NOT()
    assert_true(not_filter_0({}))
    assert_false(not_filter_0({'a': 0, 'b': 1}))

    filter_2 = neurovault.ResultFilter(
        {'a': neurovault.NotNull()}).AND(lambda d: len(d) < 2)
    assert_true(filter_2({'a': 'a'}))
    assert_false(filter_2({'a': ''}))
    assert_false(filter_2({'a': 'a', 'b': 0}))

    filt = neurovault.ResultFilter(
        a=0).AND(neurovault.ResultFilter(b=1).OR(neurovault.ResultFilter(b=2)))
    assert_true(filt({'a': 0, 'b': 1}))
    assert_false(filt({'a': 0, 'b': 0}))


# @with_setup(tst.setup_tmpdata, tst.teardown_tmpdata)
# for some reason, when using this tst.tmpdir is None.
# TODO: find out why and use tst.setup_tmpdata.
# In the meanwhile, use _TemporaryDirectory
@ignore_connection_errors
def test_simple_download():
    with _TemporaryDirectory(set_nv_dir=False) as temp_dir:
        downloaded_file = neurovault._simple_download(
            'http://neurovault.org/media/images/35/Fig3B_zstat1.nii.gz',
            os.path.join(temp_dir, 'image_35.nii.gz'), temp_dir)
        assert_true(os.path.isfile(downloaded_file))
        assert_raises(neurovault.URLError, neurovault._simple_download,
                      'http://', 'bad.nii.gz', temp_dir)


def test_checked_get_dataset_dir():
    with _TemporaryDirectory(set_nv_dir=False) as temp_dir:
        dataset_dir = neurovault._checked_get_dataset_dir(
            'neurovault', temp_dir)
        assert_true(samefile(
            dataset_dir, os.path.join(temp_dir, 'neurovault')))
        dataset_dir = neurovault._checked_get_dataset_dir(
            'neurovault', temp_dir, write_required=True)
        assert_true(samefile(
            dataset_dir, os.path.join(temp_dir, 'neurovault')))
        os.chmod(os.path.join(temp_dir, 'neurovault'), stat.S_IREAD)
        if os.access(os.path.join(temp_dir, 'neurovault'), os.W_OK):
            return
        dataset_dir = neurovault._checked_get_dataset_dir(
            'neurovault', temp_dir)
        assert_true(samefile(
            dataset_dir, os.path.join(temp_dir, 'neurovault')))
        assert_raises(IOError, neurovault._checked_get_dataset_dir,
                      'neurovault', temp_dir, write_required=True)


def test_neurovault_metadata_db_path():
    with _TemporaryDirectory() as temp_dir:
        os.remove(os.path.join(temp_dir, '.neurovault_metadata.db'))
        assert_equal(neurovault.neurovault_metadata_db_path(),
                     os.path.join(temp_dir, '.neurovault_metadata.db'))
        os.remove(os.path.join(temp_dir, '.neurovault_metadata.db'))
        os.chmod(temp_dir, stat.S_IREAD)
        if os.access(temp_dir, os.W_OK):
            return
        assert_warns(Warning, neurovault.neurovault_metadata_db_path)


def test_neurovault_directory():
    nv_dir = neurovault.neurovault_directory()
    assert_true(os.path.isdir(nv_dir))


def test_set_neurovault_directory():
    try:
        with _TemporaryDirectory() as temp_dir:
            dataset_dir = neurovault.set_neurovault_directory(temp_dir)
            assert_true(samefile(dataset_dir, temp_dir))
    finally:
        neurovault.set_neurovault_directory(None)


def test_get_temp_dir():
    with _TemporaryDirectory() as temp_dir:
        returned_temp_dir = neurovault._get_temp_dir(temp_dir)
        assert_true(samefile(
            returned_temp_dir, temp_dir))
    temp_dir = neurovault._get_temp_dir()
    try:
        assert_true(os.path.isdir(temp_dir))
    finally:
        shutil.rmtree(temp_dir)


@ignore_connection_errors
def test_fetch_neurosynth_words():
    with _TemporaryDirectory() as temp_dir:
        words_file_name = os.path.join(
            temp_dir, 'neurosynth_words_for_image_110.json')
        neurovault._fetch_neurosynth_words(
            110, words_file_name, temp_dir)
        with open(words_file_name, 'rb') as words_file:
            words = json.loads(words_file.read().decode('utf-8'))
            assert_true(words)


def test_neurosynth_words_vectorized():
    n_im = 5
    with _TemporaryDirectory() as temp_dir:
        words_files = [
            os.path.join(temp_dir, 'words_for_image_{0}.json'.format(i)) for
            i in range(n_im)]
        words = [str(i) for i in range(n_im)]
        for i, file_name in enumerate(words_files):
            word_weights = np.zeros(n_im)
            word_weights[i] = 1
            words_dict = {'data':
                          {'values':
                           dict([(k, v) for
                                 k, v in zip(words, word_weights)])}}
            with open(file_name, 'wb') as words_file:
                words_file.write(json.dumps(words_dict).encode('utf-8'))
        freq, voc = neurovault.neurosynth_words_vectorized(words_files)
        assert_equal(freq.shape, (n_im, n_im))
        assert((freq.sum(axis=0) == np.ones(n_im)).all())


def test_base_download_manager():
    download_manager = neurovault.BaseDownloadManager(neurovault_data_dir='',
                                                      max_images=5)

    def g():
        download_manager.image(None)
        for i in range(10):
            download_manager.image({})
            yield i

    assert_equal(len(list(g())), 5)


def test_write_read_metadata():
    metadata = {'relative_path': 'collection_1',
                'absolute_path': os.path.join('tmp', 'collection_1')}
    with _TemporaryDirectory() as temp_dir:
        neurovault._write_metadata(
            metadata, os.path.join(temp_dir, 'metadata.json'))
        with open(os.path.join(temp_dir, 'metadata.json'), 'rb') as meta_file:
            written_metadata = json.loads(meta_file.read().decode('utf-8'))
        assert_true('relative_path' in written_metadata)
        assert_false('absolute_path' in written_metadata)
        read_metadata = neurovault._add_absolute_paths('tmp', written_metadata)
        assert_equal(read_metadata['absolute_path'],
                     os.path.join('tmp', 'collection_1'))


def test_add_absolute_paths():
    meta = {'col_relative_path': 'collection_1',
            'col_absolute_path': os.path.join(
                'dir_0', 'neurovault', 'collection_1')}
    meta = neurovault._add_absolute_paths(os.path.join('dir_1', 'neurovault'),
                                          meta, force=False)
    assert_equal(meta['col_absolute_path'],
                 os.path.join('dir_0', 'neurovault', 'collection_1'))
    meta = neurovault._add_absolute_paths(os.path.join('dir_1', 'neurovault'),
                                          meta, force=True)
    assert_equal(meta['col_absolute_path'],
                 os.path.join('dir_1', 'neurovault', 'collection_1'))


def test_download_manager():
    with _TemporaryDirectory():
        download_manager = neurovault.DownloadManager(
            neurovault_data_dir=neurovault.neurovault_directory())
        with download_manager:
            temp_dir = download_manager.temp_dir_
            assert_true(os.path.isdir(temp_dir))
        assert_false(os.path.isdir(temp_dir))


def test_sqlite_download_manager():
    with _TemporaryDirectory():
        download_manager = neurovault.SQLiteDownloadManager(
            neurovault_data_dir=neurovault.neurovault_directory())
        with download_manager:
            assert_false(download_manager.connection_ is None)
            download_manager._add_to_images(_EXAMPLE_IM_META)
            download_manager._add_to_collections(_EXAMPLE_COL_META)
        assert_true(download_manager.connection_ is None)

        im_110_info = neurovault.read_sql_query(
            """SELECT valid_images.id AS image_id,
            collections.id AS collection_id,
            collections.owner AS owner FROM
            valid_images INNER JOIN collections ON
            valid_images.collection_id=collections.id""")

        assert_equal(im_110_info['image_id'][0], 110)
        assert_equal(im_110_info['collection_id'][0], 35)
        assert_equal(im_110_info['owner'][0], 52)
        almost_all_image_fields = list(neurovault._ALL_IMAGE_FIELDS_SQL.keys())
        almost_all_image_fields.remove('id')
        almost_all_image_fields.remove('Age')
        download_manager = neurovault.SQLiteDownloadManager(
            neurovault_data_dir=neurovault.neurovault_directory(),
            image_fields=almost_all_image_fields)
        with download_manager:
            download_manager._update_schema()
        images = neurovault.read_sql_query("SELECT * FROM images")
        assert_true('Go_No_Go' in images.keys())
        assert_true('id' in images.keys())
        assert_false('Age' in images.keys())


def test_json_add_collection_dir():
    with _TemporaryDirectory() as data_temp_dir:
        coll_dir = os.path.join(data_temp_dir, 'collection_1')
        os.makedirs(coll_dir)
        coll_file_name = os.path.join(coll_dir, 'collection_1.json')
        with open(coll_file_name, 'wb') as coll_file:
            coll_file.write(json.dumps({'id': 1}).encode('utf-8'))
        loaded = neurovault._json_add_collection_dir(coll_file_name)
        assert_equal(loaded['absolute_path'], coll_dir)
        assert_equal(loaded['relative_path'], 'collection_1')


def test_json_add_im_files_paths():
    with _TemporaryDirectory() as data_temp_dir:
        coll_dir = os.path.join(data_temp_dir, 'collection_1')
        os.makedirs(coll_dir)
        im_file_name = os.path.join(coll_dir, 'image_1.json')
        with open(im_file_name, 'wb') as im_file:
            im_file.write(json.dumps({'id': 1}).encode('utf-8'))
        loaded = neurovault._json_add_im_files_paths(im_file_name)
        assert_equal(loaded['relative_path'],
                     os.path.join('collection_1', 'image_1.nii.gz'))
        assert_true(loaded.get('neurosynth_words_relative_path') is None)


@ignore_connection_errors
def test_server_data_scroller():
    scroller = neurovault._ServerDataScroller(wanted_collection_ids=(35,))
    for im, col in scroller.scroll():
        assert_equal(col['id'], 35)


def test_split_terms():
    terms, server_terms = neurovault._split_terms(
        {'DOI': neurovault.NotNull(),
         'name': 'my_name', 'unknown_term': 'something'},
        neurovault._COL_FILTERS_AVAILABLE_ON_SERVER)
    assert_equal(terms,
                 {'DOI': neurovault.NotNull(), 'unknown_term': 'something'})
    assert_equal(server_terms, {'name': 'my_name'})


def test_move_unknown_terms_to_local_filter():
    terms, new_filter = neurovault._move_unknown_terms_to_local_filter(
        {'a': 0, 'b': 1}, neurovault.ResultFilter(), ('a',))
    assert_equal(terms, {'a': 0})
    assert_false(new_filter({'b': 0}))
    assert_true(new_filter({'b': 1}))


class TestDownloadManager(neurovault.BaseDownloadManager):
    def __init__(self, *args, **kwargs):
        super(TestDownloadManager, self).__init__(*args, **kwargs)
        self._n_times_called = 0

    def _collection_hook(self, collection_info):
        if self._n_times_called:
            raise RuntimeError('some problem')
        return collection_info

    def _image_hook(self, image_info):
        self._n_times_called = 1
        super(TestDownloadManager, self)._image_hook(image_info)
        raise neurovault.URLError('bad download')


def test_chain_local_and_remote():
    assert_raises(ValueError, lambda: list(
        neurovault._chain_local_and_remote(
            neurovault.neurovault_directory(), 'bad_mode')))


def test_fetch_neurovault_filtered():
    with _TemporaryDirectory() as temp_dir:
        data = neurovault.fetch_neurovault(
            max_images=1, fetch_neurosynth_words=True,
            fetch_reduced_rep=True, mode='overwrite')
        if data is not None:
            assert_equal(len(data.images), 1)
            neurovault.show_neurovault_image_keys(max_images=1)
            neurovault.show_neurovault_collection_keys(max_images=1)
            assert_true(neurovault._nv_schema_exists(
                neurovault.local_database_cursor()))
            assert_true('collection_id' in neurovault.column_names(
                neurovault.local_database_cursor(), 'images'))
            meta = data.images_meta[0]
            assert_false(meta['not_mni'])
            db_data = neurovault.read_sql_query(
                """SELECT id, absolute_path,
                neurosynth_words_absolute_path FROM images WHERE id=?""",
                (meta['id'],))
            assert_equal(db_data['absolute_path'][0], meta['absolute_path'])
            db_rows = neurovault.read_sql_query(
                "SELECT * FROM images", as_columns=False)
            assert_equal(db_rows[0]['absolute_path'],
                         db_data['absolute_path'][0])
            assert_raises(ValueError, neurovault.read_sql_query,
                          """SELECT images.id AS id,
                          collections.id AS id FROM
                          images INNER JOIN collections""")
            assert_false(neurovault._absolute_paths_incorrect())
            neurovault.read_sql_query(
                "UPDATE images SET absolute_path='bad_path'")
            assert_true(neurovault._absolute_paths_incorrect())
            neurovault.refresh_db()
            assert_false(neurovault._absolute_paths_incorrect())
            neurovault.read_sql_query(
                "UPDATE collections SET absolute_path='bad'")
            assert_true(neurovault._absolute_paths_incorrect())
            neurovault.recompute_db()
            assert_false(neurovault._absolute_paths_incorrect())

            assert_warns(
                Warning, neurovault.fetch_neurovault,
                download_manager=TestDownloadManager(
                    max_images=2,
                    neurovault_data_dir=neurovault.neurovault_directory()))

        os.chmod(temp_dir, stat.S_IREAD)
        if os.access(temp_dir, os.W_OK):
            return
        assert_warns(Warning, neurovault.fetch_neurovault)


def test_fetch_neurovault_explicit():
    # test using explicit id list instead of filters,
    # and downloading an image which has no collection dir
    # or metadata yet.
    with _TemporaryDirectory():
        data = neurovault.fetch_neurovault(image_ids=[111])
        if data is not None:
            assert_equal(data['images_meta'][0]['id'], 111)
            assert_equal(os.path.dirname(data['images'][0]),
                         data['collections_meta'][0]['absolute_path'])


def test_move_col_id():
    im_terms, col_terms = neurovault._move_col_id(
        {'collection_id': 1, 'not_mni': False}, {})
    assert_equal(im_terms, {'not_mni': False})
    assert_equal(col_terms, {'id': 1})

    assert_warns(Warning, neurovault._move_col_id,
                 {'collection_id': 1, 'not_mni': False}, {'id': 2})


def test_filter_field_names():
    assert_equal(neurovault._filter_field_names('a', {}),
                 neurovault.OrderedDict())
    assert_equal(neurovault._filter_field_names(('a-b',), {'a_b': 0}),
                 neurovault.OrderedDict({'a_b': 0}))


def test_get_len():
    assert_equal(neurovault._get_len('abc'), 3)
    assert_equal(neurovault._get_len(10), 0)
