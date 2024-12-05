"""Test the neurovault module."""

# Author: Jerome Dockes

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from urllib import parse

import numpy as np
import pandas as pd
import pytest
import requests

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.conftest import _rng
from nilearn.datasets import neurovault
from nilearn.image import load_img


def _get_neurovault_data():
    """Make fake images and collections to mock neurovault in the unit tests.

    Returns two pandas DataFrames: collections and images. Each row contains
    some metadata (e.g. "map_type", "is_thresholded" for images, or
    "number_of_images" for collections) for a single (fake) image or
    collection.

    These two dataframes are like a fake neurovault database and the
    `_neurovault` function, used to simulate responses from the neurovault
    API, uses this data.

    """
    if getattr(_get_neurovault_data, "data", None) is not None:
        return _get_neurovault_data.data
    rng = _rng()
    n_collections, n_images = 73, 546
    collection_ids = rng.choice(
        np.arange(1000), size=n_collections, replace=False
    )
    collections = pd.DataFrame({"id": collection_ids})
    image_ids = rng.choice(np.arange(10000), size=n_images, replace=False)
    images = pd.DataFrame({"id": image_ids})
    not_empty = rng.binomial(1, 0.9, n_collections).astype(bool)
    images["collection_id"] = rng.choice(
        collection_ids[not_empty], size=n_images
    )
    collection_sizes = images.groupby("collection_id").count()
    collections["true_number_of_images"] = collection_sizes.reindex(
        index=collections["id"].to_numpy(), fill_value=0
    ).to_numpy()
    collections["number_of_images"] = collections[
        "true_number_of_images"
    ] + rng.binomial(1, 0.1, n_collections) * rng.integers(
        0, 100, n_collections
    )
    images["not_mni"] = rng.binomial(1, 0.1, size=n_images).astype(bool)
    images["is_valid"] = rng.binomial(1, 0.1, size=n_images).astype(bool)
    images["is_thresholded"] = rng.binomial(1, 0.1, size=n_images).astype(bool)
    images["map_type"] = rng.choice(
        [
            "T map",
            "Z map",
            "ROI/mask",
            "anatomical",
            "parcellation",
            "something else",
        ],
        size=n_images,
        p=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05],
    )
    images["image_type"] = rng.choice(
        ["statistic_map", "atlas", "other type"],
        size=n_images,
        p=[0.4, 0.4, 0.2],
    )
    images["some_key"] = "some_value"
    images[13] = rng.standard_normal(n_images)
    url = "https://neurovault.org/media/images/{}/{}.nii.gz"
    image_names = [
        hashlib.sha1(bytes(img_id)).hexdigest()[:4] for img_id in image_ids
    ]
    images["file"] = [
        url.format(col_id, img_name)
        for (col_id, img_name) in zip(images["collection_id"], image_names)
    ]
    collections = collections.set_index("id", drop=False)
    images = images.set_index("id", drop=False)
    _get_neurovault_data.data = collections, images
    return collections, images


def _parse_query(query):
    """Extract key-value pairs from a url query string.

    for example
    "collection=23&someoption&format=json"
      -> {"collection": "23", "someoption": None, "format": "json"}

    """
    parts = [p.split("=") for p in query.split("&")]
    result = {}
    for p in parts:
        if len(p) == 2:
            result[p[0]] = p[1]
        if len(p) == 1:
            result[p[0]] = None
    return result


def _neurovault_collections(parts, query):
    """Mock the Neurovault API behind the `/api/collections/` path.

    parts : the parts of the URL path after "collections"
     ie [], ["<somecollectionid>"], or ["<somecollectionid>", "images"]

    query : the parsed query string, e.g. {"offset": "15", "limit": "5"}

    returns a dictionary of API results

    See the neurovault API docs for details: https://neurovault.org/api-docs

    """
    if parts:
        return _neurovault_one_collection(parts)
    collections, _ = _get_neurovault_data()
    offset, limit = int(query.get("offset", 0)), int(query.get("limit", 2))
    batch = collections.iloc[offset : offset + limit].to_dict(orient="records")
    return {"count": len(collections), "results": batch}


def _neurovault_one_collection(parts):
    """Mock Neurovault API \
       behind the `/api/collections/<somecollectionid>` path.

    parts : parts of the URL path after "collections",
      ie ["<somecollectionid>"] or ["<somecollectionid>", "images"]

    returns a dictionary of API results

    See the neurovault API docs for details: https://neurovault.org/api-docs

    """
    col_id = int(parts[0])
    collections, images = _get_neurovault_data()
    if col_id not in collections.index:
        return {"detail": "Not found."}
    if len(parts) == 1:
        return collections.loc[col_id].to_dict()
    if parts[1] != "images":
        return ""
    col_images = images[images["collection_id"] == col_id]
    return {
        "count": len(col_images),
        "results": col_images.to_dict(orient="records"),
    }


def _neurovault_images(parts, query):
    """Mock the Neurovault API behind the `/api/images/` path.

    parts : parts of the URL path after "images",
      ie [] or ["<someimageid>"]

    query : the parsed query string, e.g. {"offset": "15", "limit": "5"}

    returns a dictionary of API results

    See the neurovault API docs for details: https://neurovault.org/api-docs

    """
    if parts:
        return _neurovault_one_image(parts[0])
    _, images = _get_neurovault_data()
    offset, limit = int(query.get("offset", 0)), int(query.get("limit", 2))
    batch = images.iloc[offset : offset + limit].to_dict(orient="records")
    return {"count": len(images), "results": batch}


def _neurovault_one_image(img_id):
    """Mock the Neurovault API behind the `/api/images/<someimageid>` path.

    returns a dictionary of API results

    See the neurovault API docs for details: https://neurovault.org/api-docs

    """
    img_id = int(img_id)
    _, images = _get_neurovault_data()
    if img_id not in images.index:
        return {"detail": "Not found."}
    return images.loc[img_id].to_dict()


def _neurovault_file(parts, query):  # noqa: ARG001
    """Mock the Neurovault API behind the `/media/images/` path."""
    return generate_fake_fmri(length=1)[0]


class _NumpyJsonEncoder(json.JSONEncoder):
    """A json encoder that can handle numpy objects."""

    def default(self, obj):
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _neurovault(match, request):  # noqa: ARG001
    """Mock response content from the Neurovault API.

    The fake data used to generate responses is provided by
    `_get_neurovault_data`.

    See the neurovault API docs for details on the queries and corresponding
    responses: https://neurovault.org/api-docs

    """
    handlers = {
        "media": {"images": _neurovault_file},
        "api": {
            "collections": _neurovault_collections,
            "images": _neurovault_images,
        },
    }
    info = parse.urlparse(request.url)
    parts = list(filter(bool, info.path.split("/")))
    endpoint, section = parts[0], parts[1]

    result = handlers[endpoint][section](parts[2:], _parse_query(info.query))
    should_jsonify_response = endpoint == "api"
    return (
        json.dumps(result, cls=_NumpyJsonEncoder).encode("UTF-8")
        if should_jsonify_response
        else result
    )


@pytest.fixture(autouse=True)
def neurovault_mocker(request_mocker):
    request_mocker.url_mapping["*neurovault.org*"] = _neurovault


def test_remove_none_strings():
    info = {
        "a": "None / Other",
        "b": "",
        "c": "N/A",
        "d": None,
        "e": 0,
        "f": "a",
        "g": "Name",
    }

    assert neurovault._remove_none_strings(info) == {
        "a": None,
        "b": None,
        "c": None,
        "d": None,
        "e": 0,
        "f": "a",
        "g": "Name",
    }


def test_append_filters_to_query():
    query = neurovault._append_filters_to_query(
        neurovault._NEUROVAULT_COLLECTIONS_URL, {"DOI": 17}
    )

    assert query == "https://neurovault.org/api/collections/?DOI=17"

    query = neurovault._append_filters_to_query(
        neurovault._NEUROVAULT_COLLECTIONS_URL, {"id": 40}
    )
    assert query == "https://neurovault.org/api/collections/40"


def test_get_batch():
    batch = neurovault._get_batch(neurovault._NEUROVAULT_COLLECTIONS_URL)
    assert "results" in batch
    assert "count" in batch


def test_get_batch_error(tmp_path):
    with pytest.raises(requests.RequestException):
        neurovault._get_batch("http://")
    with pytest.raises(ValueError):
        neurovault._get_batch(
            f"file://{tmp_path / 'test_nv.txt'!s}",
        )

    no_results_url = (
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        "esearch.fcgi?db=pmc&retmode=json&term=fmri"
    )

    with pytest.raises(ValueError):
        neurovault._get_batch(no_results_url)


def test_scroll_server_results():
    result = list(
        neurovault._scroll_server_results(
            neurovault._NEUROVAULT_COLLECTIONS_URL, max_results=6, batch_size=3
        )
    )

    assert len(result) == 6

    result = list(
        neurovault._scroll_server_results(
            neurovault._NEUROVAULT_COLLECTIONS_URL,
            max_results=3,
            local_filter=lambda r: False,
        )
    )
    assert not result

    no_results = neurovault._scroll_server_results(
        "http://BAD_URL", max_results=3, local_filter=lambda r: True
    )
    next(no_results)


def test_is_null():
    is_null = neurovault.IsNull()

    assert is_null != "a"
    assert is_null == ""
    assert str(is_null) == "IsNull()"


def test_not_null():
    not_null = neurovault.NotNull()

    assert not_null == "a"
    assert not_null != ""
    assert str(not_null) == "NotNull()"


def test_not_equal():
    not_equal = neurovault.NotEqual("a")

    assert not_equal == "b"
    assert not_equal == 1
    assert not_equal != "a"
    assert not_equal != "a"
    assert str(not_equal) == "NotEqual('a')"


def test_order_comp():
    geq = neurovault.GreaterOrEqual("2016-07-12T11:29:12.263046Z")

    assert geq == "2016-08-12T11:29:12.263046Z"
    assert geq != "2016-06-12T11:29:12.263046Z"
    assert str(geq) == "GreaterOrEqual('2016-07-12T11:29:12.263046Z')"

    gt = neurovault.GreaterThan("abc")

    assert gt != "abc"
    assert gt == "abd"
    assert str(gt) == "GreaterThan('abc')"

    lt = neurovault.LessThan(7)

    assert lt != 7
    assert lt == 5
    assert lt != "a"
    assert str(lt) == "LessThan(7)"

    leq = neurovault.LessOrEqual(4.5)

    assert leq == 4.4
    assert leq != 4.6
    assert str(leq) == "LessOrEqual(4.5)"


def test_is_in():
    is_in = neurovault.IsIn(0, 1)

    assert is_in == 0
    assert is_in != 2
    assert str(is_in) == "IsIn(0, 1)"

    countable = neurovault.IsIn(*range(11))

    assert countable == 7
    assert countable != 12


def test_not_in():
    not_in = neurovault.NotIn(0, 1)

    assert not_in != 0
    assert not_in == 2
    assert str(not_in) == "NotIn(0, 1)"


def test_contains():
    contains = neurovault.Contains("a", 0)

    assert contains != 10
    assert contains == ["b", 1, "a", 0]
    assert contains != ["b", 1, 0]
    assert contains != ["b", 1, "a"]
    assert str(contains) == "Contains('a', 0)"

    contains = neurovault.Contains("house", "face")

    assert contains == "face vs house"
    assert contains != "smiling face vs frowning face"


def test_not_contains():
    not_contains = neurovault.NotContains("ab")

    assert not_contains is not None
    assert not_contains == "a_b"
    assert not_contains == "bcd"
    assert not_contains != "_abcd"
    assert not_contains != "_abcd"
    assert str(not_contains) == "NotContains('ab',)"


def test_pattern():
    # Python std lib doc poker hand example
    pattern_0 = neurovault.Pattern(r"[0-9akqj]{5}$")

    assert str(pattern_0) == "Pattern(pattern='[0-9akqj]{5}$', flags=0)"

    pattern_1 = neurovault.Pattern(r"[0-9akqj]{5}$", re.IGNORECASE)

    assert pattern_0 == "ak05q"
    assert pattern_0 != "Ak05q"
    assert pattern_0 != "ak05e"
    assert pattern_1 == "ak05q"
    assert pattern_1 == "Ak05q"
    assert pattern_1 != "ak05e"


def test_result_filter():
    filter_0 = neurovault.ResultFilter(
        query_terms={"a": 0}, callable_filter=lambda d: len(d) < 5, b=1
    )

    assert str(filter_0) == "ResultFilter"
    assert filter_0["a"] == 0
    assert filter_0({"a": 0, "b": 1, "c": 2})
    assert not filter_0({"a": 0, "b": 1, "c": 2, "d": 3, "e": 4})
    assert not filter_0({"b": 1, "c": 2, "d": 3})
    assert not filter_0({"a": 1, "b": 1, "c": 2})

    filter_1 = neurovault.ResultFilter(query_terms={"c": 2})
    filter_1["d"] = neurovault.NotNull()

    assert filter_1({"c": 2, "d": 1})
    assert not filter_1({"c": 2, "d": 0})

    filter_1["d"] = neurovault.IsIn(0, 1)

    assert filter_1({"c": 2, "d": 1})
    assert not filter_1({"c": 2, "d": 2})

    del filter_1["d"]

    assert filter_1({"c": 2, "d": 2})

    filter_1["d"] = neurovault.NotIn(0, 1)

    assert not filter_1({"c": 2, "d": 1})
    assert filter_1({"c": 2, "d": 3})

    filter_1.add_filter(lambda d: len(d) > 2)

    assert not filter_1({"c": 2, "d": 3})
    assert filter_1({"c": 2, "d": 3, "e": 4})


def test_result_filter_combinations():
    filter_0 = neurovault.ResultFilter(a=0, b=1)
    filter_1 = neurovault.ResultFilter(c=2, d=3)

    filter_0_and_1 = filter_0.AND(filter_1)

    assert filter_0_and_1({"a": 0, "b": 1, "c": 2, "d": 3})
    assert not filter_0_and_1({"a": 0, "b": 1, "c": 2, "d": None})
    assert not filter_0_and_1({"a": None, "b": 1, "c": 2, "d": 3})

    filter_0_or_1 = filter_0.OR(filter_1)

    assert filter_0_or_1({"a": 0, "b": 1, "c": 2, "d": 3})
    assert filter_0_or_1({"a": 0, "b": 1, "c": 2, "d": None})
    assert filter_0_or_1({"a": None, "b": 1, "c": 2, "d": 3})
    assert not filter_0_or_1({"a": None, "b": 1, "c": 2, "d": None})

    filter_0_xor_1 = filter_0.XOR(filter_1)

    assert not filter_0_xor_1({"a": 0, "b": 1, "c": 2, "d": 3})
    assert filter_0_xor_1({"a": 0, "b": 1, "c": 2, "d": None})
    assert filter_0_xor_1({"a": None, "b": 1, "c": 2, "d": 3})
    assert not filter_0_xor_1({"a": None, "b": 1, "c": 2, "d": None})

    not_filter_0 = filter_0.NOT()

    assert not_filter_0({})
    assert not not_filter_0({"a": 0, "b": 1})

    filter_2 = neurovault.ResultFilter({"a": neurovault.NotNull()}).AND(
        lambda d: len(d) < 2
    )

    assert filter_2({"a": "a"})
    assert not filter_2({"a": ""})
    assert not filter_2({"a": "a", "b": 0})

    filt = neurovault.ResultFilter(a=0).AND(
        neurovault.ResultFilter(b=1).OR(neurovault.ResultFilter(b=2))
    )

    assert filt({"a": 0, "b": 1})
    assert not filt({"a": 0, "b": 0})


def test_simple_download(tmp_path):
    downloaded_file = neurovault._simple_download(
        "https://neurovault.org/media/images/35/Fig3B_zstat1.nii.gz",
        tmp_path / "image_35.nii.gz",
        tmp_path,
    )
    assert downloaded_file.is_file()


def test_simple_download_error(tmp_path, request_mocker):
    request_mocker.url_mapping["*"] = requests.RequestException()

    with pytest.raises(requests.RequestException):
        neurovault._simple_download(
            "http://",
            tmp_path / "bad.nii.gz",
            tmp_path,
        )


def test_neurosynth_words_vectorized(tmp_path):
    n_im = 5
    words_files = [tmp_path / f"words_for_image_{i}.json" for i in range(n_im)]
    words = [str(i) for i in range(n_im)]
    for i, file_name in enumerate(words_files):
        word_weights = np.zeros(n_im)
        word_weights[i] = 1
        words_dict = {"data": {"values": dict(zip(words, word_weights))}}
        with file_name.open("wb") as words_file:
            words_file.write(json.dumps(words_dict).encode("utf-8"))
    freq, _ = neurovault.neurosynth_words_vectorized(words_files)

    assert freq.shape == (n_im, n_im)
    assert (freq.sum(axis=0) == np.ones(n_im)).all()


def test_neurosynth_words_vectorized_warning(tmp_path):
    with pytest.warns(UserWarning):
        neurovault.neurosynth_words_vectorized(
            ((tmp_path, "no_words_here.json"),)
        )


def test_write_read_metadata(tmp_path):
    metadata = {
        "relative_path": "collection_1",
        "absolute_path": Path("tmp", "collection_1"),
    }
    metadata_path = tmp_path / "metadata.json"

    neurovault._write_metadata(metadata, metadata_path)
    with metadata_path.open("rb") as meta_file:
        written_metadata = json.loads(meta_file.read().decode("utf-8"))

    assert "relative_path" in written_metadata
    assert "absolute_path" not in written_metadata

    read_metadata = neurovault._add_absolute_paths(
        Path("tmp"), written_metadata
    )

    assert read_metadata["absolute_path"] == Path("tmp", "collection_1")


def test_add_absolute_paths():
    meta = {
        "col_relative_path": "collection_1",
        "col_absolute_path": Path("dir_0", "neurovault", "collection_1"),
    }
    meta = neurovault._add_absolute_paths(
        Path("dir_1", "neurovault"), meta, force=False
    )

    assert meta["col_absolute_path"] == Path(
        "dir_0", "neurovault", "collection_1"
    )

    meta = neurovault._add_absolute_paths(
        Path("dir_1", "neurovault"), meta, force=True
    )

    assert meta["col_absolute_path"] == Path(
        "dir_1", "neurovault", "collection_1"
    )

    meta = {"id": 0}
    meta_transformed = neurovault._add_absolute_paths(
        Path("dir_1", "neurovault"), meta, force=True
    )

    assert meta == meta_transformed


def test_json_add_collection_dir(tmp_path):
    coll_dir = tmp_path / "collection_1"
    coll_dir.mkdir()
    coll_file_name = coll_dir / "collection_1.json"
    with coll_file_name.open("wb") as coll_file:
        coll_file.write(json.dumps({"id": 1}).encode("utf-8"))
    loaded = neurovault._json_add_collection_dir(coll_file_name)

    assert loaded["absolute_path"] == coll_dir.absolute()
    assert loaded["relative_path"] == coll_dir


def test_json_add_im_files_paths(tmp_path):
    coll_dir = tmp_path / "collection_1"
    coll_dir.mkdir()
    im_file_name = coll_dir / "image_1.json"
    with im_file_name.open("wb") as im_file:
        im_file.write(json.dumps({"id": 1}).encode("utf-8"))
    loaded = neurovault._json_add_im_files_paths(im_file_name)

    assert loaded["relative_path"] == coll_dir / "image_1.nii.gz"
    assert loaded.get("neurosynth_words_relative_path") is None


def test_split_terms():
    terms, server_terms = neurovault._split_terms(
        {
            "DOI": neurovault.NotNull(),
            "name": "my_name",
            "unknown_term": "something",
        },
        neurovault._COL_FILTERS_AVAILABLE_ON_SERVER,
    )

    assert terms == {"DOI": neurovault.NotNull(), "unknown_term": "something"}
    assert server_terms == {"name": "my_name"}


def test_move_unknown_terms_to_local_filter():
    terms, new_filter = neurovault._move_unknown_terms_to_local_filter(
        {"a": 0, "b": 1}, neurovault.ResultFilter(), ("a",)
    )

    assert terms == {"a": 0}
    assert not new_filter({"b": 0})
    assert new_filter({"b": 1})


def test_move_col_id():
    im_terms, col_terms = neurovault._move_col_id(
        {"collection_id": 1, "not_mni": False}, {}
    )

    assert im_terms == {"not_mni": False}
    assert col_terms == {"id": 1}

    with pytest.warns(UserWarning):
        neurovault._move_col_id(
            {"collection_id": 1, "not_mni": False}, {"id": 2}
        )


def test_download_image_terms(tmp_path, request_mocker):
    image_info = {"id": "a"}
    collection = {
        "relative_path": "collection",
        "absolute_path": tmp_path / "collection",
    }
    collection["absolute_path"].mkdir(parents=True)
    download_params = {
        "temp_dir": tmp_path,
        "verbose": 3,
        "fetch_neurosynth_words": True,
    }
    request_mocker.url_mapping["*"] = requests.RequestException()
    neurovault._download_image_terms(image_info, collection, download_params)


def test_download_image_terms_error(tmp_path, request_mocker):
    image_info = {"id": "a"}
    collection = {
        "relative_path": "collection",
        "absolute_path": tmp_path / "collection",
    }
    collection["absolute_path"].mkdir(parents=True)
    download_params = {
        "temp_dir": tmp_path,
        "verbose": 3,
        "fetch_neurosynth_words": True,
    }
    request_mocker.url_mapping["*"] = requests.RequestException()
    download_params["allow_neurosynth_failure"] = False

    with pytest.raises(RuntimeError):
        neurovault._download_image_terms(
            image_info,
            collection,
            download_params,
        )
    # no fail if file already exists
    with Path(
        collection["absolute_path"], "neurosynth_words_for_image_a.json"
    ).open("w"):
        pass
    neurovault._download_image_terms(image_info, collection, download_params)


def test_download_image():
    image = neurovault._download_image(None, {})
    assert image is None


def test_fetch_neurovault(tmp_path):
    # check that nothing is downloaded in offline mode
    data = neurovault.fetch_neurovault(mode="offline", data_dir=tmp_path)

    assert len(data.images) == 0

    # try to download an image
    data = neurovault.fetch_neurovault(
        max_images=11,
        fetch_neurosynth_words=True,
        mode="overwrite",
        data_dir=tmp_path,
    )
    # specifying a filter while leaving the default term
    # filters in place should raise a warning.
    with pytest.warns(UserWarning):
        neurovault.fetch_neurovault(
            image_filter=lambda x: True, max_images=1, mode="offline"
        )

    assert data.images
    assert len(data.images) == 11
    for meta in data.images_meta:
        assert not meta["not_mni"]
        assert not meta["is_thresholded"]
        assert meta["map_type"] not in [
            "ROI/mask",
            "anatomical",
            "parcellation",
        ]
        assert meta["image_type"] != "atlas"

    # using a data directory we can't write into should raise a
    # warning unless mode is 'offline'
    tmp_path.chmod(stat.S_IREAD | stat.S_IEXEC)
    (tmp_path / "neurovault").chmod(stat.S_IREAD | stat.S_IEXEC)
    if os.access(tmp_path / "neurovault", os.W_OK):
        return

    with pytest.warns(UserWarning):
        neurovault.fetch_neurovault(data_dir=tmp_path)


def test_fetch_neurovault_errors(capsys, request_mocker):
    """Test that errors are logged when the server returns an error code.

    May "spam" your standard output with requests exceptions,
    but that's expected.
    """
    request_mocker.url_mapping["*"] = 500
    data = neurovault.fetch_neurovault()

    captured = capsys.readouterr()
    match = re.search("500 Error", captured.err)
    assert match is not None

    assert len(data.images) == 0


def test_fetch_neurovault_ids(tmp_path):
    collections, images = _get_neurovault_data()
    collections = collections.sort_values(
        by="true_number_of_images", ascending=False
    )
    other_col_id, *col_ids = collections["id"].to_numpy()[:3]
    img_ids = images[images["collection_id"] == other_col_id]["id"].to_numpy()[
        :3
    ]
    img_from_cols_ids = images[images["collection_id"].isin(col_ids)][
        "id"
    ].to_numpy()

    with pytest.raises(ValueError):
        neurovault.fetch_neurovault_ids(mode="bad")

    data = neurovault.fetch_neurovault_ids(
        image_ids=img_ids, collection_ids=col_ids, data_dir=tmp_path
    )

    expected_images = list(img_ids) + list(img_from_cols_ids)

    assert len(data.images) == len(expected_images)
    assert {img["id"] for img in data["images_meta"]} == set(expected_images)
    assert Path(data["images"][0]).parent == Path(
        data["collections_meta"][0]["absolute_path"]
    )

    # check that there are no Path objects
    for image in data.images:
        assert isinstance(image, str)
    for meta in data.images_meta + data.collections_meta:
        for value in meta.values():
            assert not isinstance(value, Path)

    # check image can be loaded again from disk
    data = neurovault.fetch_neurovault_ids(
        image_ids=[img_ids[0]], data_dir=tmp_path, mode="offline"
    )

    assert len(data.images) == 1

    # check that download_new mode forces overwrite
    modified_meta = data["images_meta"][0]

    assert modified_meta["some_key"] == "some_value"

    modified_meta["some_key"] = "some_other_value"
    # mess it up on disk
    meta_path = (
        Path(modified_meta["absolute_path"]).parent
        / f"image_{img_ids[0]}_metadata.json"
    )
    # convert Path to str for JSON serialization
    modified_meta = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in modified_meta.items()
    }
    with meta_path.open("wb") as meta_f:
        meta_f.write(json.dumps(modified_meta).encode("UTF-8"))

    # fresh download
    data = neurovault.fetch_neurovault_ids(
        image_ids=[img_ids[0]], data_dir=tmp_path, mode="download_new"
    )
    data = neurovault.fetch_neurovault_ids(
        image_ids=[img_ids[0]], data_dir=tmp_path, mode="offline"
    )

    # should not have changed
    assert data["images_meta"][0]["some_key"] == "some_other_value"

    data = neurovault.fetch_neurovault_ids(
        image_ids=[img_ids[0]], data_dir=tmp_path, mode="overwrite"
    )
    data = neurovault.fetch_neurovault_ids(
        image_ids=[img_ids[0]], data_dir=tmp_path, mode="offline"
    )

    # should be back to the original version
    assert data["images_meta"][0]["some_key"] == "some_value"


def test_should_download_resampled_images_only_if_no_previous_download(
    tmp_path,
):
    collections, _ = _get_neurovault_data()

    sample_collection = collections.iloc[0]
    sample_collection_id = sample_collection["id"]
    expected_number_of_images = sample_collection["true_number_of_images"]

    data = neurovault.fetch_neurovault_ids(
        collection_ids=[sample_collection_id],
        data_dir=tmp_path,
        resample=True,
    )

    # Check the expected size of the dataset
    assert (len(data["images_meta"])) == expected_number_of_images

    _check_resampled_version_is_here(data)

    _check_all_affines_match_neurovault_affine(data)

    _check_original_version_is_not_here(data)


def test_download_original_images_along_resamp_images_if_previously_downloaded(
    tmp_path,
):
    collections, _ = _get_neurovault_data()

    sample_collection = collections.iloc[0]
    sample_collection_id = sample_collection["id"]

    # Fetch non-resampled images
    data = neurovault.fetch_neurovault_ids(
        collection_ids=[sample_collection_id],
        data_dir=tmp_path,
        resample=True,
    )

    # Check that only the resampled version is here
    _check_resampled_version_is_here(data)
    _check_original_version_is_not_here(data)

    # Get the time of the last access to the resampled data
    access_time_resampled = (
        Path(data["images_meta"][0]["resampled_absolute_path"]).stat().st_atime
    )

    # Download original data
    data_orig = neurovault.fetch_neurovault_ids(
        collection_ids=[sample_collection_id],
        data_dir=tmp_path,
        resample=False,
    )

    # Get the time of the last access to one of the original files
    # (which should be download time)
    access_time = (
        Path(data_orig["images_meta"][0]["absolute_path"]).stat().st_atime
    )

    # Check that the last access to the original data is after the access
    # to the resampled data
    assert access_time - access_time_resampled > 0

    # Check that the original version is now here
    # (previous test should have failed anyway if not)
    _check_original_version_is_here(data_orig)

    _check_no_affine_match_neurovault_affine(data_orig)


def test_download_resamp_images_along_original_images_if_previously_downloaded(
    tmp_path,
):
    collections, _ = _get_neurovault_data()

    sample_collection = collections.iloc[0]
    sample_collection_id = sample_collection["id"]

    # Fetch non-resampled images
    data_orig = neurovault.fetch_neurovault_ids(
        collection_ids=[sample_collection_id],
        data_dir=tmp_path,
        resample=False,
    )

    _check_original_version_is_here(data_orig)

    _check_resampled_version_is_not_here(data_orig)

    # Asks for the resampled version. This should only resample, not download.

    # Get the time of the last modification to the original data
    modif_time_original = (
        Path(data_orig["images_meta"][0]["absolute_path"]).stat().st_mtime
    )

    # Ask for resampled data, which should only trigger resample
    data = neurovault.fetch_neurovault_ids(
        collection_ids=[sample_collection_id],
        data_dir=tmp_path,
        resample=True,
    )

    # Get the time of the last modification to the original data, after fetch
    modif_time_original_after = (
        Path(data["images_meta"][0]["absolute_path"]).stat().st_mtime
    )

    # The time difference should be 0
    assert np.isclose(modif_time_original, modif_time_original_after)

    _check_resampled_version_is_here(data)
    # And the original version should still be here as well
    _check_original_version_is_here(data)

    _check_all_affines_match_neurovault_affine(data)

    _check_no_affine_match_neurovault_affine(data_orig)


def _check_resampled_version_is_here(data):
    assert np.all(
        [
            Path(im_meta["resampled_absolute_path"]).is_file()
            for im_meta in data["images_meta"]
        ]
    )


def _check_resampled_version_is_not_here(data):
    assert not np.any(
        [
            Path(im_meta["resampled_absolute_path"]).is_file()
            for im_meta in data["images_meta"]
        ]
    )


def _check_original_version_is_here(data):
    assert np.all(
        [
            Path(im_meta["absolute_path"]).is_file()
            for im_meta in data["images_meta"]
        ]
    )


def _check_original_version_is_not_here(data):
    assert not np.any(
        [
            Path(im_meta["absolute_path"]).is_file()
            for im_meta in data["images_meta"]
        ]
    )


def _check_all_affines_match_neurovault_affine(data):
    affines = [load_img(cur_im).affine for cur_im in data["images"]]

    assert np.all(
        [np.all(affine == neurovault.STD_AFFINE) for affine in affines]
    )


def _check_no_affine_match_neurovault_affine(data):
    affines = [load_img(cur_im).affine for cur_im in data["images"]]

    assert not np.any(
        [np.all(affine == neurovault.STD_AFFINE) for affine in affines]
    )
