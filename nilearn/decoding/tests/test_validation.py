"""Tests for outer cross-validation of image-based decoders."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import clone, config_context
from sklearn.base import is_classifier
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import check_scoring, get_scorer
from sklearn.model_selection import KFold, LeaveOneGroupOut, StratifiedKFold

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.decoding import (
    Decoder,
    DecoderRegressor,
    FREMClassifier,
    FREMRegressor,
    cross_val_decoder_score,
)
from nilearn.image import check_niimg, get_data, index_img, iter_img
from nilearn.maskers import NiftiMasker
from nilearn.surface import SurfaceImage

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The decoding model will be trained only:UserWarning"
    ),
]


@pytest.fixture
def volume_data():
    """Return a small image, its mask, and balanced binary targets."""
    img, mask = generate_fake_fmri(shape=(6, 6, 6), length=30)
    return img, mask, np.tile([0, 1], 15)


@pytest.fixture(params=["volume", "surface"])
def image_data(request, volume_data, surf_img_2d, rng):
    """Return small volume or surface classification inputs."""
    if request.param == "volume":
        return volume_data
    img = surf_img_2d(30)
    for values in img.data.parts.values():
        values[:] = rng.normal(size=values.shape)
    mask = SurfaceImage(
        img.mesh,
        {
            part: np.ones(values.shape[0])
            for part, values in img.data.parts.items()
        },
    )
    return img, mask, np.tile([0, 1], 15)


def _make_decoder(cls, mask):
    """Build a decoder with a small, deterministic inner search."""
    classifier = is_classifier(cls())
    kwargs = {
        "mask": mask,
        "estimator": "svc" if classifier else "svr",
        "scoring": "accuracy" if classifier else "r2",
        "param_grid": {"C": [1.0]},
        "cv": 2,
        "screening_percentile": None,
        "standardize": "zscore_sample",
    }
    if classifier:
        kwargs["estimator_args"] = {"random_state": 0}
    if cls in (FREMClassifier, FREMRegressor):
        kwargs["clustering_percentile"] = 100
    return cls(**kwargs)


@pytest.mark.parametrize(
    "cls", [Decoder, DecoderRegressor, FREMClassifier, FREMRegressor]
)
def test_cross_val_decoder_score(image_data, cls, rng):
    """All decoder classes return held-out scores for real image inputs."""
    img, mask, y = image_data
    decoder = _make_decoder(cls, mask)
    if not is_classifier(decoder):
        y = rng.normal(size=y.shape)

    scores = cross_val_decoder_score(
        decoder, img, y, cv=3, error_score="raise"
    )

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    if is_classifier(decoder):
        assert ((scores >= 0) & (scores <= 1)).all()
    assert not hasattr(decoder, "coef_")
    assert not hasattr(decoder, "masker_")


@pytest.mark.parametrize(
    "scoring",
    [None, "accuracy", "roc_auc", "balanced_accuracy", get_scorer("accuracy")],
)
def test_cross_val_decoder_score_matches_manual(image_data, scoring):
    """Outer scores match independently fitted and scored image folds."""
    img, mask, y = image_data
    decoder = _make_decoder(Decoder, mask)
    cv = StratifiedKFold(3)
    scorer = check_scoring(decoder, scoring=scoring)
    expected = []
    for train, test in cv.split(np.zeros(len(y)), y):
        fitted = clone(decoder).fit(index_img(img, train), y[train])
        expected.append(scorer(fitted, index_img(img, test), y[test]))

    scores = cross_val_decoder_score(
        decoder, img, y, scoring=scoring, cv=cv, error_score="raise"
    )

    assert_allclose(scores, expected)


def test_cross_val_decoder_score_regression_scorer(image_data, rng):
    """A regression scorer and explicit KFold work with image inputs."""
    img, mask, y = image_data
    y = rng.normal(size=y.shape)
    decoder = _make_decoder(DecoderRegressor, mask)

    scores = cross_val_decoder_score(
        decoder,
        img,
        y,
        cv=KFold(3),
        scoring="neg_mean_absolute_error",
        error_score="raise",
    )

    assert scores.shape == (3,)
    assert np.isfinite(scores).all()
    assert (scores <= 0).all()


@pytest.mark.parametrize(
    "form", ["list", "tuple", "path", "str", "paths", "str_paths"]
)
def test_cross_val_decoder_score_volume_inputs(volume_data, form, tmp_path):
    """Volume paths and sample collections preserve sample ordering."""
    img, mask, y = volume_data
    decoder = _make_decoder(Decoder, mask)
    expected = cross_val_decoder_score(decoder, img, y, cv=3)
    if form in ("list", "tuple"):
        imgs = list(iter_img(img))
        if form == "tuple":
            imgs = tuple(imgs)
    elif form in ("paths", "str_paths"):
        imgs = []
        for i, sample in enumerate(iter_img(img)):
            path = tmp_path / f"sample_{i}.nii.gz"
            sample.to_filename(path)
            imgs.append(str(path) if form == "str_paths" else path)
    else:
        imgs = tmp_path / "samples.nii.gz"
        img.to_filename(imgs)
        if form == "str":
            imgs = str(imgs)

    scores = cross_val_decoder_score(
        decoder, imgs, y, cv=3, error_score="raise"
    )

    assert_allclose(scores, expected)


@pytest.mark.parametrize("n_samples", [None, 1, 10])
def test_cross_val_decoder_score_surface_inputs(surf_img_2d, n_samples):
    """Surface collections can contain one or multiple samples per image."""
    img = surf_img_2d(30)
    y = np.tile([0, 1], 15)
    decoder = _make_decoder(Decoder, None)
    batch_size = n_samples or 1
    imgs = [
        index_img(img, slice(start, start + batch_size))
        for start in range(0, 30, batch_size)
    ]
    if n_samples is None:
        for sample in imgs:
            for part, values in sample.data.parts.items():
                sample.data.parts[part] = values.ravel()
    original = [
        {part: values.copy() for part, values in sample.data.parts.items()}
        for sample in imgs
    ]

    expected = cross_val_decoder_score(decoder, img, y, cv=3)
    scores = cross_val_decoder_score(
        decoder, imgs, y, cv=3, error_score="raise"
    )

    assert_allclose(scores, expected)
    for sample, data in zip(imgs, original, strict=True):
        for part, values in data.items():
            assert_array_equal(sample.data.parts[part], values)


def test_cross_val_decoder_score_surface_preserves_input(surf_img_2d):
    """Surface data, targets, and the original estimator are preserved."""
    img = surf_img_2d(30)
    original = {part: values.copy() for part, values in img.data.parts.items()}
    y = np.tile([0, 1], 15)
    decoder = _make_decoder(Decoder, None)

    scores = cross_val_decoder_score(decoder, img, y, error_score="raise")

    assert scores.shape == (5,)
    for part, values in original.items():
        assert_array_equal(img.data.parts[part], values)
    assert_array_equal(y, np.tile([0, 1], 15))
    assert not hasattr(decoder, "masker_")


@pytest.mark.parametrize("routing", [False, True])
def test_cross_val_decoder_score_groups(image_data, routing):
    """Group metadata is subset for the inner and outer splitters."""
    img, mask, y = image_data
    groups = np.repeat(np.arange(3), 10)
    decoder = _make_decoder(Decoder, mask).set_params(cv=LeaveOneGroupOut())
    expected = []
    for train, test in LeaveOneGroupOut().split(np.zeros(len(y)), y, groups):
        fitted = clone(decoder).fit(
            index_img(img, train), y[train], groups=groups[train]
        )
        expected.append(fitted.score(index_img(img, test), y[test]))

    with config_context(enable_metadata_routing=routing):
        if routing:
            decoder.set_fit_request(groups=True)
        scores = cross_val_decoder_score(
            decoder,
            img,
            y,
            cv=LeaveOneGroupOut(),
            groups=None if routing else groups,
            params={"groups": groups},
            error_score="raise",
        )

    assert_allclose(scores, expected)
    assert_array_equal(groups, np.repeat(np.arange(3), 10))


def test_cross_val_decoder_score_training_folds(volume_data, monkeypatch):
    """Each cloned masker fits only its training images, without mutation."""
    img, mask, y = volume_data
    decoder = _make_decoder(Decoder, mask).fit(img, y)
    original_coef = decoder.coef_.copy()
    original_data = get_data(img).copy()
    original_y = y.copy()
    fits = []
    original_fit = NiftiMasker.fit

    def record_fit(masker, imgs=None, y=None):
        assert not hasattr(masker, "mask_img_")
        fits.append((masker, get_data(check_niimg(imgs)).copy()))
        return original_fit(masker, imgs, y)

    monkeypatch.setattr(NiftiMasker, "fit", record_fit)
    splits = list(StratifiedKFold(3).split(np.zeros(len(y)), y))

    cross_val_decoder_score(decoder, img, y, cv=splits, error_score="raise")

    assert len(fits) == 3
    assert len({id(masker) for masker, _ in fits}) == 3
    for (_, data), (train, _) in zip(fits, splits, strict=True):
        assert_array_equal(data, original_data[..., train])
    assert_array_equal(decoder.coef_, original_coef)
    assert_array_equal(get_data(img), original_data)
    assert_array_equal(y, original_y)


@pytest.mark.single_process
def test_cross_val_decoder_score_parallel(volume_data):
    """Parallel outer folds return the same scores in the same order."""
    img, mask, y = volume_data
    decoder = _make_decoder(Decoder, mask).set_params(n_jobs=1)

    expected = cross_val_decoder_score(decoder, img, y, cv=3, n_jobs=1)
    scores = cross_val_decoder_score(
        decoder, img, y, cv=3, n_jobs=2, pre_dispatch=2, error_score="raise"
    )

    assert_allclose(scores, expected)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"cv": 1}, "cv"),
        ({"scoring": "not_a_scorer"}, "scoring"),
        ({"groups": [0, 1], "cv": LeaveOneGroupOut()}, "inconsistent"),
    ],
)
def test_cross_val_decoder_score_invalid_parameters(
    volume_data, kwargs, message
):
    """Invalid splitters, scorers, and group lengths use sklearn errors."""
    img, mask, y = volume_data
    decoder = _make_decoder(Decoder, mask)

    with pytest.raises(ValueError, match=message):
        cross_val_decoder_score(decoder, img, y, **kwargs)


def test_cross_val_decoder_score_target_length(image_data):
    """Target length is checked against samples, not spatial dimensions."""
    img, mask, y = image_data
    decoder = _make_decoder(Decoder, mask)

    with pytest.raises(ValueError, match="inconsistent numbers of samples"):
        cross_val_decoder_score(decoder, img, y[:-1], cv=3)


def test_cross_val_decoder_score_fit_error(volume_data):
    """An invalid classification training fold can raise immediately."""
    img, mask, y = volume_data
    decoder = _make_decoder(Decoder, mask)
    splits = [(np.flatnonzero(y == 0), np.flatnonzero(y == 1))]

    with pytest.raises(ValueError, match="class"):
        cross_val_decoder_score(
            decoder, img, y, cv=splits, error_score="raise"
        )


def test_cross_val_decoder_score_error_score(volume_data):
    """A failed fold gets the requested score without dropping other folds."""
    img, mask, y = volume_data
    decoder = _make_decoder(Decoder, mask)
    splits = [(np.flatnonzero(y == 0), np.flatnonzero(y == 1))]
    splits.extend(StratifiedKFold(3).split(np.zeros(len(y)), y))

    with pytest.warns(FitFailedWarning, match="1 fits failed"):
        scores = cross_val_decoder_score(
            decoder, img, y, cv=splits, error_score=-1
        )

    assert scores.shape == (4,)
    assert scores[0] == -1
    assert ((scores[1:] >= 0) & (scores[1:] <= 1)).all()


def test_cross_val_decoder_score_dimension(volume_data):
    """A single 3D image is not a sequence of samples."""
    img, mask, y = volume_data
    decoder = _make_decoder(Decoder, mask)

    with pytest.raises(TypeError, match="Expected dimension is 4D"):
        cross_val_decoder_score(decoder, index_img(img, 0), y, cv=3)
