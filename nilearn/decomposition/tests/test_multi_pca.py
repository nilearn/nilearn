"""Test the multi-PCA module."""

import numpy as np
import pytest
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

from nilearn.decomposition._multi_pca import _MultiPCA
from nilearn.decomposition.tests.conftest import (
    RANDOM_STATE,
    check_decomposition_estimator,
)
from nilearn.maskers import NiftiMasker, SurfaceMasker


@pytest.mark.timeout(0)
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
@pytest.mark.parametrize("length", [1, 2])
def test_multi_pca(
    length,
    data_type,
    decomposition_mask_img,
    decomposition_images,
):
    """Components are the same if we put twice the same data, \
       and that fit output is deterministic.
    """
    multi_pca = _MultiPCA(
        mask=decomposition_mask_img, n_components=3, random_state=RANDOM_STATE
    )
    multi_pca.fit(decomposition_images)

    check_decomposition_estimator(multi_pca, data_type)

    components1 = multi_pca.components_

    multi_pca = _MultiPCA(
        mask=decomposition_mask_img, n_components=3, random_state=RANDOM_STATE
    )
    multi_pca.fit(length * decomposition_images)
    components2 = multi_pca.components_

    if length == 1:
        assert_array_equal(components1, components2)
    else:
        assert_array_almost_equal(components1, components2)


@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_with_masker_without_cca_smoke(
    data_type,
    decomposition_masker,
    decomposition_images,
):
    """Multi-pca can run with a masker \
        and without canonical correlation analysis.
    """
    multi_pca = _MultiPCA(
        mask=decomposition_masker,
        do_cca=False,
        n_components=3,
    )
    multi_pca.fit(decomposition_images[:2])

    check_decomposition_estimator(multi_pca, data_type)

    # Smoke test the transform and inverse_transform
    multi_pca.inverse_transform(multi_pca.transform(decomposition_images[-2:]))


@pytest.mark.parametrize("with_activation", [False])
@pytest.mark.parametrize("data_type", ["nifti", "surface"])
def test_multi_pca_score_single_subject_n_components(
    data_type,
    decomposition_mask_img,
    decomposition_img,
    with_activation,  # noqa: ARG001
):
    """Score is one for n_components == n_sample \
       in single subject configuration.
    """
    multi_pca = _MultiPCA(
        mask=decomposition_mask_img,
        random_state=RANDOM_STATE,
        memory_level=0,
        n_components=5,
    )
    multi_pca.fit(decomposition_img)
    s = multi_pca.score(decomposition_img)

    assert_almost_equal(s, 1.0, 1)

    # Per component score
    multi_pca = _MultiPCA(
        mask=decomposition_mask_img,
        random_state=RANDOM_STATE,
        memory_level=0,
        n_components=5,
    )
    multi_pca.fit(decomposition_img)

    check_decomposition_estimator(multi_pca, data_type)

    if data_type == "nifti":
        masker = NiftiMasker(decomposition_mask_img).fit()
    elif data_type == "surface":
        masker = SurfaceMasker(decomposition_mask_img).fit()

    s = multi_pca._raw_score(
        masker.transform(decomposition_img), per_component=True
    )

    assert s.shape == (5,)
    assert np.all(s <= 1)
    assert np.all(s >= 0)
