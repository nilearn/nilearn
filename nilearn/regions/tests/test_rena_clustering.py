import numpy as np
import pytest
from joblib import Memory
from numpy.testing import assert_array_equal
from sklearn import __version__ as sklearn_version

from nilearn._utils import compare_version
from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.conftest import _img_3d_mni
from nilearn.image import get_data
from nilearn.maskers import NiftiMasker
from nilearn.regions.rena_clustering import (
    ReNA,
    _make_edges_and_weights_surface,
)

extra_valid_checks = [
    "check_clusterer_compute_labels_predict",
    "check_complex_data",
    "check_estimators_empty_data_messages",
    "check_estimator_sparse_array",
    "check_estimator_sparse_matrix",
    "check_fit2d_1sample",
    "check_fit2d_1feature",
    "check_fit1d",
    "check_no_attributes_set_in_init",
    "check_transformers_unfitted",
    "check_transformer_n_iter",
]
# TODO remove when dropping support for sklearn_version < 1.5.0
if compare_version(sklearn_version, "<", "1.5.0"):
    extra_valid_checks.append("check_estimator_sparse_data")


@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=ReNA(_img_3d_mni(), n_clusters=2),
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


@pytest.mark.xfail(reason="invalid checks should fail")
@pytest.mark.parametrize(
    "estimator, check, name",
    check_estimator(
        estimator=ReNA(_img_3d_mni(), n_clusters=2),
        valid=False,
        extra_valid_checks=extra_valid_checks,
    ),
)
def test_check_estimator_invalid(estimator, check, name):  # noqa: ARG001
    """Check compliance with sklearn estimators."""
    check(estimator)


def test_tags():
    """Smoke test to test private tag function."""
    _, mask_img = generate_fake_fmri(shape=(10, 11, 12), length=5)
    rena = ReNA(mask_img, n_clusters=10)
    rena._more_tags()


def test_rena_clustering():
    data_img, mask_img = generate_fake_fmri(shape=(10, 11, 12), length=5)

    data = get_data(data_img)
    mask = get_data(mask_img)

    X = np.empty((data.shape[3], int(mask.sum())))
    for i in range(data.shape[3]):
        X[i, :] = np.copy(data[:, :, :, i])[get_data(mask_img) != 0]

    nifti_masker = NiftiMasker(mask_img=mask_img).fit()
    n_voxels = nifti_masker.transform(data_img).shape[1]

    rena = ReNA(mask_img, n_clusters=10)

    X_red = rena.fit_transform(X)
    X_compress = rena.inverse_transform(X_red)

    assert rena.n_clusters_ == 10
    assert X.shape == X_compress.shape

    memory = Memory(location=None)
    rena = ReNA(mask_img, n_clusters=-2, memory=memory)
    with pytest.raises(ValueError):
        rena.fit(X)

    rena = ReNA(mask_img, n_clusters=10, scaling=True)
    X_red = rena.fit_transform(X)
    X_compress = rena.inverse_transform(X_red)

    for n_iter in [-2, 0]:
        rena = ReNA(mask_img, n_iter=n_iter, memory=memory)
        with pytest.raises(ValueError):
            rena.fit(X)

    for n_clusters in [1, 2, 4, 8]:
        rena = ReNA(
            mask_img, n_clusters=n_clusters, n_iter=1, memory=memory
        ).fit(X)
        assert n_clusters != rena.n_clusters_

    del n_voxels, X_red, X_compress


def test_make_edges_and_weights_surface_smoke(surf_mask, rng):
    n_samples = 5
    n_features = surf_mask().shape[0]

    X = rng.random((n_samples, n_features))

    print(surf_mask().mesh.parts["left"].faces)

    edges, weights = _make_edges_and_weights_surface(X, surf_mask())

    assert len(edges) == 2
    assert len(weights) == 2
    for part in ["left", "right"]:
        assert part in edges
        assert part in weights

    assert_array_equal(edges["left"], np.array([[0, 1, 0], [1, 2, 2]]))
    assert_array_equal(
        edges["right"], np.array([[1, 0, 2, 0, 1, 0], [3, 1, 3, 3, 2, 2]])
    )

    assert len(weights["left"]) == 3
    assert len(weights["right"]) == 6
