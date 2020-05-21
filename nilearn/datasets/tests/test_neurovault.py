"""
Test the neurovault module.
"""
# Author: Jerome Dockes
# License: simplified BSD

import os
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
from collections import namedtuple
from functools import wraps

import numpy as np
import pytest

from nilearn.datasets import neurovault
from nilearn.image import load_img

def _same_stat(path_1, path_2):
    path_1 = os.path.abspath(os.path.expanduser(path_1))
    path_2 = os.path.abspath(os.path.expanduser(path_2))
    return os.stat(path_1) == os.stat(path_2)


if samefile is None:
    samefile = _same_stat


class _TestTemporaryDirectory(object):

    def __enter__(self):
        self.temp_dir_ = tempfile.mkdtemp()
        return self.temp_dir_

    def __exit__(self, *args):
        os.chmod(self.temp_dir_, stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
        for root, dirnames, filenames in os.walk(self.temp_dir_):
            for name in dirnames:
                os.chmod(os.path.join(root, name),
                         stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
            for name in filenames:
                os.chmod(os.path.join(root, name),
                         stat.S_IWUSR | stat.S_IRUSR)
        shutil.rmtree(self.temp_dir_)


def test_remove_none_strings():
    info = {'a': 'None / Other',
            'b': '',
            'c': 'N/A',
            'd': None,
            'e': 0,
            'f': 'a',
            'g': 'Name'}
    assert (neurovault._remove_none_strings(info) ==
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
    assert (
        query == 'http://neurovault.org/api/collections/?DOI=17')
    query = neurovault._append_filters_to_query(
        neurovault._NEUROVAULT_COLLECTIONS_URL,
        {'id': 40})
    assert query == 'http://neurovault.org/api/collections/40'


def ignore_connection_errors(func):
    """Catch URL errors.

    Used to prevent tests from failing because of a network problem or because
    Neurovault server is down.

    """
    @wraps(func)
    def test_wrap(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except neurovault.URLError:
            raise pytest.skip(msg='connection problem')

    return test_wrap


_FakeResponse = namedtuple('_FakeResponse', ('headers'))


@ignore_connection_errors
def test_get_encoding():
    response = _FakeResponse({'Content-Type': 'text/json; charset=utf-8'})
    assert neurovault._get_encoding(response) == 'utf-8'
    response.headers.pop('Content-Type')
    pytest.raises(ValueError, neurovault._get_encoding, response)
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
    pytest.raises(neurovault.URLError, neurovault._get_batch, 'http://')
    with _TestTemporaryDirectory() as temp_dir:
        with open(os.path.join(temp_dir, 'test_nv.txt'), 'w'):
            pass
        pytest.raises(ValueError, neurovault._get_batch, 'file://{0}'.format(
            os.path.join(temp_dir, 'test_nv.txt')))
    no_results_url = ('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
                      'esearch.fcgi?db=pmc&retmode=json&term=fmri')
    pytest.raises(ValueError, neurovault._get_batch, no_results_url)


@ignore_connection_errors
def test_scroll_server_results():
    result = list(neurovault._scroll_server_results(
        neurovault._NEUROVAULT_COLLECTIONS_URL, max_results=6, batch_size=3))
    assert len(result) == 6
    result = list(neurovault._scroll_server_results(
        neurovault._NEUROVAULT_COLLECTIONS_URL, max_results=3,
        local_filter=lambda r: False))
    assert len(result) == 0
    no_results = neurovault._scroll_server_results(
        'http://BAD_URL', max_results=3,
        local_filter=lambda r: True)
    next(no_results)


def test_is_null():
    is_null = neurovault.IsNull()
    assert is_null != 'a'
    assert not is_null != ''
    assert 'a' != is_null
    assert not '' != is_null
    assert not is_null == 'a'
    assert is_null == ''
    assert not 'a' == is_null
    assert '' == is_null
    assert str(is_null) == 'IsNull()'


def test_not_null():
    not_null = neurovault.NotNull()
    assert not_null == 'a'
    assert not not_null == ''
    assert 'a' == not_null
    assert not '' == not_null
    assert not not_null != 'a'
    assert not_null != ''
    assert not 'a' != not_null
    assert '' != not_null
    assert str(not_null) == 'NotNull()'


def test_not_equal():
    not_equal = neurovault.NotEqual('a')
    assert not_equal == 'b'
    assert not_equal == 1
    assert not not_equal == 'a'
    assert 'b' == not_equal
    assert 1 == not_equal
    assert not 'a' == not_equal
    assert not not_equal != 'b'
    assert not not_equal != 1
    assert not_equal != 'a'
    assert not 'b' != not_equal
    assert not 1 != not_equal
    assert 'a' != not_equal
    assert str(not_equal) == "NotEqual('a')"


def test_order_comp():
    geq = neurovault.GreaterOrEqual('2016-07-12T11:29:12.263046Z')
    assert '2016-08-12T11:29:12.263046Z' == geq
    assert '2016-06-12T11:29:12.263046Z' != geq
    assert str(geq) == "GreaterOrEqual('2016-07-12T11:29:12.263046Z')"
    gt = neurovault.GreaterThan('abc')
    assert not gt == 'abc'
    assert gt == 'abd'
    assert str(gt) == "GreaterThan('abc')"
    lt = neurovault.LessThan(7)
    assert not 7 == lt
    assert not 5 != lt
    assert not lt == 'a'
    assert str(lt) == 'LessThan(7)'
    leq = neurovault.LessOrEqual(4.5)
    assert 4.4 == leq
    assert not 4.6 == leq
    assert str(leq) == 'LessOrEqual(4.5)'


def test_is_in():
    is_in = neurovault.IsIn(0, 1)
    assert is_in == 0
    assert not is_in == 2
    assert 0 == is_in
    assert not 2 == is_in
    assert not is_in != 0
    assert is_in != 2
    assert not 0 != is_in
    assert 2 != is_in
    assert str(is_in) == 'IsIn(0, 1)'
    countable = neurovault.IsIn(*range(11))
    assert 7 == countable
    assert not countable == 12


def test_not_in():
    not_in = neurovault.NotIn(0, 1)
    assert not_in != 0
    assert not not_in != 2
    assert 0 != not_in
    assert not 2 != not_in
    assert not not_in == 0
    assert not_in == 2
    assert not 0 == not_in
    assert 2 == not_in
    assert str(not_in) == 'NotIn(0, 1)'


def test_contains():
    contains = neurovault.Contains('a', 0)
    assert not contains == 10
    assert contains == ['b', 1, 'a', 0]
    assert ['b', 1, 'a', 0] == contains
    assert contains != ['b', 1, 0]
    assert ['b', 1, 'a'] != contains
    assert not contains != ['b', 1, 'a', 0]
    assert not ['b', 1, 'a', 0] != contains
    assert not contains == ['b', 1, 0]
    assert not ['b', 1, 'a'] == contains
    assert str(contains) == "Contains('a', 0)"
    contains = neurovault.Contains('house', 'face')
    assert 'face vs house' == contains
    assert not 'smiling face vs frowning face' == contains


def test_not_contains():
    not_contains = neurovault.NotContains('ab')
    assert None != not_contains
    assert not_contains == 'a_b'
    assert 'bcd' == not_contains
    assert not_contains != '_abcd'
    assert '_abcd' != not_contains
    assert not not_contains != 'a_b'
    assert not 'bcd' != not_contains
    assert not not_contains == '_abcd'
    assert not '_abcd' == not_contains
    assert str(not_contains) == "NotContains('ab',)"


def test_pattern():
    # Python std lib doc poker hand example
    pattern_0 = neurovault.Pattern(r'[0-9akqj]{5}$')
    assert str(pattern_0) == "Pattern(pattern='[0-9akqj]{5}$', flags=0)"
    pattern_1 = neurovault.Pattern(r'[0-9akqj]{5}$', re.I)
    assert pattern_0 == 'ak05q'
    assert not pattern_0 == 'Ak05q'
    assert not pattern_0 == 'ak05e'
    assert pattern_1 == 'ak05q'
    assert pattern_1 == 'Ak05q'
    assert not pattern_1 == 'ak05e'
    assert not pattern_0 != 'ak05q'
    assert pattern_0 != 'Ak05q'
    assert pattern_0 != 'ak05e'
    assert not pattern_1 != 'ak05q'
    assert not pattern_1 != 'Ak05q'
    assert pattern_1 != 'ak05e'

    assert 'ak05q' == pattern_0
    assert not 'Ak05q' == pattern_0
    assert not 'ak05e' == pattern_0
    assert 'ak05q' == pattern_1
    assert 'Ak05q' == pattern_1
    assert not 'ak05e' == pattern_1
    assert not 'ak05q' != pattern_0
    assert 'Ak05q' != pattern_0
    assert 'ak05e' != pattern_0
    assert not 'ak05q' != pattern_1
    assert not 'Ak05q' != pattern_1
    assert 'ak05e' != pattern_1


def test_result_filter():
    filter_0 = neurovault.ResultFilter(query_terms={'a': 0},
                                       callable_filter=lambda d: len(d) < 5,
                                       b=1)
    assert np.unicode(filter_0) == u'ResultFilter'
    assert filter_0['a'] == 0
    assert filter_0({'a': 0, 'b': 1, 'c': 2})
    assert not filter_0({'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4})
    assert not filter_0({'b': 1, 'c': 2, 'd': 3})
    assert not filter_0({'a': 1, 'b': 1, 'c': 2})

    filter_1 = neurovault.ResultFilter(query_terms={'c': 2})
    filter_1['d'] = neurovault.NotNull()
    assert filter_1({'c': 2, 'd': 1})
    assert not filter_1({'c': 2, 'd': 0})
    filter_1['d'] = neurovault.IsIn(0, 1)
    assert filter_1({'c': 2, 'd': 1})
    assert not filter_1({'c': 2, 'd': 2})
    del filter_1['d']
    assert filter_1({'c': 2, 'd': 2})
    filter_1['d'] = neurovault.NotIn(0, 1)
    assert not filter_1({'c': 2, 'd': 1})
    assert filter_1({'c': 2, 'd': 3})
    filter_1.add_filter(lambda d: len(d) > 2)
    assert not filter_1({'c': 2, 'd': 3})
    assert filter_1({'c': 2, 'd': 3, 'e': 4})


def test_result_filter_combinations():
    filter_0 = neurovault.ResultFilter(a=0, b=1)
    filter_1 = neurovault.ResultFilter(c=2, d=3)

    filter_0_and_1 = filter_0.AND(filter_1)
    assert filter_0_and_1({'a': 0, 'b': 1, 'c': 2, 'd': 3})
    assert not filter_0_and_1({'a': 0, 'b': 1, 'c': 2, 'd': None})
    assert not filter_0_and_1({'a': None, 'b': 1, 'c': 2, 'd': 3})

    filter_0_or_1 = filter_0.OR(filter_1)
    assert filter_0_or_1({'a': 0, 'b': 1, 'c': 2, 'd': 3})
    assert filter_0_or_1({'a': 0, 'b': 1, 'c': 2, 'd': None})
    assert filter_0_or_1({'a': None, 'b': 1, 'c': 2, 'd': 3})
    assert not filter_0_or_1({'a': None, 'b': 1, 'c': 2, 'd': None})

    filter_0_xor_1 = filter_0.XOR(filter_1)
    assert not filter_0_xor_1({'a': 0, 'b': 1, 'c': 2, 'd': 3})
    assert filter_0_xor_1({'a': 0, 'b': 1, 'c': 2, 'd': None})
    assert filter_0_xor_1({'a': None, 'b': 1, 'c': 2, 'd': 3})
    assert not filter_0_xor_1({'a': None, 'b': 1, 'c': 2, 'd': None})

    not_filter_0 = filter_0.NOT()
    assert not_filter_0({})
    assert not not_filter_0({'a': 0, 'b': 1})

    filter_2 = neurovault.ResultFilter(
        {'a': neurovault.NotNull()}).AND(lambda d: len(d) < 2)
    assert filter_2({'a': 'a'})
    assert not filter_2({'a': ''})
    assert not filter_2({'a': 'a', 'b': 0})

    filt = neurovault.ResultFilter(
        a=0).AND(neurovault.ResultFilter(b=1).OR(neurovault.ResultFilter(b=2)))
    assert filt({'a': 0, 'b': 1})
    assert not filt({'a': 0, 'b': 0})


def _fail(*args, **kwargs):
    raise neurovault.URLError('problem')


class _FailingDownloads():

    def __init__(self):
        self.original_fetch = None

    def __enter__(self):
        self.original_fetch = neurovault._fetch_file
        neurovault._fetch_file = _fail

    def __exit__(self, *args):
        neurovault._fetch_file = self.original_fetch


@ignore_connection_errors
def test_simple_download():
    with _TestTemporaryDirectory() as temp_dir:
        downloaded_file = neurovault._simple_download(
            'http://neurovault.org/media/images/35/Fig3B_zstat1.nii.gz',
            os.path.join(temp_dir, 'image_35.nii.gz'), temp_dir)
        assert os.path.isfile(downloaded_file)
        with _FailingDownloads():
            pytest.raises(neurovault.URLError,
                          neurovault._simple_download, 'http://',
                          os.path.join(temp_dir, 'bad.nii.gz'), temp_dir)


def test_neurosynth_words_vectorized():
    n_im = 5
    with _TestTemporaryDirectory() as temp_dir:
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
        assert freq.shape == (n_im, n_im)
        assert((freq.sum(axis=0) == np.ones(n_im)).all())
        with pytest.warns(UserWarning):
            neurovault.neurosynth_words_vectorized(
                (os.path.join(temp_dir, 'no_words_here.json'),)
            )


def test_write_read_metadata():
    metadata = {'relative_path': 'collection_1',
                'absolute_path': os.path.join('tmp', 'collection_1')}
    with _TestTemporaryDirectory() as temp_dir:
        neurovault._write_metadata(
            metadata, os.path.join(temp_dir, 'metadata.json'))
        with open(os.path.join(temp_dir, 'metadata.json'), 'rb') as meta_file:
            written_metadata = json.loads(meta_file.read().decode('utf-8'))
        assert 'relative_path' in written_metadata
        assert not 'absolute_path' in written_metadata
        read_metadata = neurovault._add_absolute_paths('tmp', written_metadata)
        assert (read_metadata['absolute_path'] ==
                     os.path.join('tmp', 'collection_1'))


def test_add_absolute_paths():
    meta = {'col_relative_path': 'collection_1',
            'col_absolute_path': os.path.join(
                'dir_0', 'neurovault', 'collection_1')}
    meta = neurovault._add_absolute_paths(os.path.join('dir_1', 'neurovault'),
                                          meta, force=False)
    assert (meta['col_absolute_path'] ==
                 os.path.join('dir_0', 'neurovault', 'collection_1'))
    meta = neurovault._add_absolute_paths(os.path.join('dir_1', 'neurovault'),
                                          meta, force=True)
    assert (meta['col_absolute_path'] ==
                 os.path.join('dir_1', 'neurovault', 'collection_1'))
    meta = {'id': 0}
    meta_transformed = neurovault._add_absolute_paths(
        os.path.join('dir_1', 'neurovault'), meta, force=True)
    assert meta == meta_transformed


def test_json_add_collection_dir():
    with _TestTemporaryDirectory() as data_temp_dir:
        coll_dir = os.path.join(data_temp_dir, 'collection_1')
        os.makedirs(coll_dir)
        coll_file_name = os.path.join(coll_dir, 'collection_1.json')
        with open(coll_file_name, 'wb') as coll_file:
            coll_file.write(json.dumps({'id': 1}).encode('utf-8'))
        loaded = neurovault._json_add_collection_dir(coll_file_name)
        assert loaded['absolute_path'] == coll_dir
        assert loaded['relative_path'] == 'collection_1'


def test_json_add_im_files_paths():
    with _TestTemporaryDirectory() as data_temp_dir:
        coll_dir = os.path.join(data_temp_dir, 'collection_1')
        os.makedirs(coll_dir)
        im_file_name = os.path.join(coll_dir, 'image_1.json')
        with open(im_file_name, 'wb') as im_file:
            im_file.write(json.dumps({'id': 1}).encode('utf-8'))
        loaded = neurovault._json_add_im_files_paths(im_file_name)
        assert (loaded['relative_path'] ==
                     os.path.join('collection_1', 'image_1.nii.gz'))
        assert loaded.get('neurosynth_words_relative_path') is None


def test_split_terms():
    terms, server_terms = neurovault._split_terms(
        {'DOI': neurovault.NotNull(),
         'name': 'my_name', 'unknown_term': 'something'},
        neurovault._COL_FILTERS_AVAILABLE_ON_SERVER)
    assert (terms ==
                 {'DOI': neurovault.NotNull(), 'unknown_term': 'something'})
    assert server_terms == {'name': 'my_name'}


def test_move_unknown_terms_to_local_filter():
    terms, new_filter = neurovault._move_unknown_terms_to_local_filter(
        {'a': 0, 'b': 1}, neurovault.ResultFilter(), ('a',))
    assert terms == {'a': 0}
    assert not new_filter({'b': 0})
    assert new_filter({'b': 1})


def test_move_col_id():
    im_terms, col_terms = neurovault._move_col_id(
        {'collection_id': 1, 'not_mni': False}, {})
    assert im_terms == {'not_mni': False}
    assert col_terms == {'id': 1}

    with pytest.warns(UserWarning):
        neurovault._move_col_id(
            {'collection_id': 1, 'not_mni': False}, {'id': 2}
        )


def test_download_image_terms():
    with _TestTemporaryDirectory() as temp_dir:
        image_info = {'id': 'a'}
        collection = {'relative_path': 'collection',
                      'absolute_path': os.path.join(temp_dir, 'collection')}
        os.makedirs(collection['absolute_path'])
        download_params = {'temp_dir': temp_dir, 'verbose': 3,
                           'fetch_neurosynth_words': True}
        with _FailingDownloads():
            neurovault._download_image_terms(
                image_info, collection, download_params)
            download_params['allow_neurosynth_failure'] = False
            pytest.raises(RuntimeError,
                          neurovault._download_image_terms,
                          image_info, collection, download_params)
            with open(os.path.join(
                collection['absolute_path'],
                'neurosynth_words_for_image_a.json'), 'w'):
                pass
            neurovault._download_image_terms(
                image_info, collection, download_params)


def test_download_image():
    image = neurovault._download_image(None, {})
    assert image is None


def test_fetch_neurovault():
    with _TestTemporaryDirectory() as temp_dir:
        # check that nothing is downloaded in offline mode
        data = neurovault.fetch_neurovault(
            mode='offline', data_dir=temp_dir)
        assert len(data.images) == 0
        # try to download an image
        data = neurovault.fetch_neurovault(
            max_images=1, fetch_neurosynth_words=True,
            mode='overwrite', data_dir=temp_dir)
        # specifying a filter while leaving the default term
        # filters in place should raise a warning.
        with pytest.warns(UserWarning):
            neurovault.fetch_neurovault(
                image_filter=lambda x: True, max_images=1,
                mode='offline'
            )
        # if neurovault was available one image matching
        # default filters should have been downloaded
        if data.images:
            assert len(data.images) == 1
            meta = data.images_meta[0]
            assert not meta['not_mni']
            assert meta['is_valid']
            assert not meta['not_mni']
            assert not meta['is_thresholded']
            assert not meta['map_type'] in [
                'ROI/mask', 'anatomical', 'parcellation']
            assert not meta['image_type'] == 'atlas'

        # using a data directory we can't write into should raise a
        # warning unless mode is 'offline'
        os.chmod(temp_dir, stat.S_IREAD | stat.S_IEXEC)
        os.chmod(os.path.join(temp_dir, 'neurovault'),
                 stat.S_IREAD | stat.S_IEXEC)
        if os.access(os.path.join(temp_dir, 'neurovault'), os.W_OK):
            return
        with pytest.warns(UserWarning):
            neurovault.fetch_neurovault(data_dir=temp_dir)


def test_fetch_neurovault_ids():
    # test using explicit id list instead of filters, and downloading
    # an image which has no collection dir or metadata yet.
    with _TestTemporaryDirectory() as data_dir:
        pytest.raises(ValueError, neurovault.fetch_neurovault_ids, mode='bad')
        data = neurovault.fetch_neurovault_ids(
            image_ids=[111], collection_ids=[307], data_dir=data_dir)
        if len(data.images) == 2:
            assert ([img['id'] for img in data['images_meta']] ==
                         [1750, 111])
            assert (os.path.dirname(data['images'][0]) ==
                         data['collections_meta'][0]['absolute_path'])
            # check image can be loaded again from disk
            data = neurovault.fetch_neurovault_ids(
                image_ids=[111], data_dir=data_dir, mode='offline')
            assert len(data.images) == 1
            # check that download_new mode forces overwrite
            modified_meta = data['images_meta'][0]
            assert modified_meta['figure'] == '3A'
            modified_meta['figure'] = '3B'
            # mess it up on disk
            meta_path = os.path.join(
                os.path.dirname(modified_meta['absolute_path']),
                'image_111_metadata.json')
            with open(meta_path, 'wb') as meta_f:
                meta_f.write(json.dumps(modified_meta).encode('UTF-8'))
            # fresh download
            data = neurovault.fetch_neurovault_ids(
                image_ids=[111], data_dir=data_dir, mode='download_new')
            data = neurovault.fetch_neurovault_ids(
                image_ids=[111], data_dir=data_dir, mode='offline')
            # should not have changed
            assert data['images_meta'][0]['figure'] == '3B'
            data = neurovault.fetch_neurovault_ids(
                image_ids=[111], data_dir=data_dir, mode='overwrite')
            data = neurovault.fetch_neurovault_ids(
                image_ids=[111], data_dir=data_dir, mode='offline')
            # should be back to the original version
            assert data['images_meta'][0]['figure'] == '3A'

def test_resampling():
    ### Tests all the usecases with resampling
    with _TestTemporaryDirectory() as data_dir:
        ### First test case : a collection hasn't been downloaded yet, user asks for resampled version

        data = neurovault.fetch_neurovault_ids(collection_ids=[4666], data_dir=data_dir,resample=True)

        # Check the expected size of the dataset
        assert (len(data['images_meta'])) == 3
        
        # Check that the resampled version is here
        assert np.all([os.path.isfile(im_meta['resampled_absolute_path']) for im_meta in data['images_meta']])

        # Load images that are fetched and check the affines
        affines = [load_img(cur_im).affine for cur_im in data['images']]
        assert np.all([np.all(affine == neurovault.STD_AFFINE) for affine in affines])

        # Check that the original version is NOT here
        assert not np.any([os.path.isfile(im_meta['absolute_path']) for im_meta in data['images_meta']])

        # Ask to download the non-resampled version. This should trigger download
        # (TODO - How can we test systematically that download does happen ?  )
        data_orig = neurovault.fetch_neurovault_ids(collection_ids=[4666], data_dir=data_dir, resample=False)

        # Check that the original version is now here
        assert np.all([os.path.isfile(im_meta['absolute_path']) for im_meta in data_orig['images_meta']])

        # Check that the affines of the original version do not correspond to the resampled one
        affines_orig = [load_img(cur_im).affine for cur_im in data_orig['images']]
        assert not np.any([np.all(affine == neurovault.STD_AFFINE) for affine in affines_orig])


    with _TestTemporaryDirectory() as data_dir:
        ### Second test case : a collection has been downloaded in the original version, user asks for resampled version

        data_orig = neurovault.fetch_neurovault_ids(collection_ids=[4666], data_dir=data_dir,resample=False)

        # Check that the original version is here
        assert np.all([os.path.isfile(im_meta['absolute_path']) for im_meta in data_orig['images_meta']])

        # Check that the resampled version is NOT here
        assert not np.any([os.path.isfile(im_meta['resampled_absolute_path']) for im_meta in data_orig['images_meta']])

        # Asks for the resampled version. This should only resample, not download. 
        # (TODO - How can we test systematically that download does not happen ?  )
        print("Asking for resampled version - Check here that download doesn't happen") 
        data = neurovault.fetch_neurovault_ids(collection_ids=[4666], data_dir=data_dir, resample=True)

        # Check that the resampled version is here
        assert np.all([os.path.isfile(im_meta['resampled_absolute_path']) for im_meta in data['images_meta']])

        #And the original version should still be here as well
        assert np.all([os.path.isfile(im_meta['absolute_path']) for im_meta in data['images_meta']])

        # Load resampled images and check the affines

        affines = [load_img(cur_im).affine for cur_im in data['images']]
        assert np.all([np.all(affine==neurovault.STD_AFFINE) for affine in affines])

        # Check that the affines of the original version do not correspond to the resampled one
        affines_orig = [load_img(cur_im).affine for cur_im in data_orig['images']]
        assert not np.any([np.all(affine==neurovault.STD_AFFINE) for affine in affines_orig])







        
