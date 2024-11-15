import numpy as np
import pytest
from joblib import Memory
from numpy.testing import assert_array_equal
from sklearn import __version__ as sklearn_version

from nilearn._utils import compare_version
from nilearn._utils.class_inspect import check_estimator
from nilearn._utils.data_gen import generate_fake_fmri
from nilearn.conftest import _img_3d_mni
from nilearn.experimental.surface import SurfaceImage
from nilearn.maskers import SurfaceMasker
from nilearn.image import get_data
from nilearn.maskers import NiftiMasker
from nilearn.regions.rena_clustering import (
    ReNA,
    _make_edges_and_weights_surface,
    _make_edges_surface,
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


# ------------------------ surface tests ------------------------------------ #


@pytest.fixture()
def _make_surface_class_data(rng, surf_img, shape=(50,)):
    """Create a surface image classification for testing."""
    y = rng.choice([0, 1], size=shape)
    return surf_img(shape), y


@pytest.mark.parametrize("part", ["left", "right"])
def test_make_edges_surface(surf_mask, part):
    """Test if the edges and edge mask are correctly computed."""
    faces = surf_mask().mesh.parts[part].faces
    # the mask for left part has total 4 vertices out of which 2 are True
    # and for right part it has total 5 vertices out of which 3 are True
    mask = surf_mask().data.parts[part]
    edges_unmasked, edges_mask = _make_edges_surface(faces, mask)

    # only one edge remains after masking the left part (between 2 vertices)
    if part == "left":
        assert edges_unmasked[:, edges_mask].shape == (2, 1)
    # three edges remain after masking the right part (between 3 vertices)
    elif part == "right":
        assert edges_unmasked[:, edges_mask].shape == (2, 3)


def test_make_edges_and_weights_surface(
    surf_mesh, _make_surface_class_data, rng
):
    """Smoke test for _make_edges_and_weights_surface. Here we create a new
    surface mask (relative to the one used in test_make_edges_surface) to make
    sure overall edge and weight computation is robust.
    """
    # make a new mask for this test
    # the mask for left part has total 4 vertices out of which 3 are True
    # and for right part it has total 5 vertices out of which 3 are True
    data = {
        "left": np.array([False, True, True, True]),
        "right": np.array([True, True, False, True, False]),
    }
    surf_mask = SurfaceImage(surf_mesh(), data)
    # create a surface masker
    masker = SurfaceMasker(surf_mask).fit()
    # create SurfaceImage with several samples
    surf_img, _ = _make_surface_class_data
    # mask the surface image
    X = masker.transform(surf_img)
    # compute edges and weights
    edges, weights = _make_edges_and_weights_surface(X, surf_mask)

    # make sure edges and weights have two parts, left and right
    assert len(edges) == 2
    assert len(weights) == 2
    for part in ["left", "right"]:
        assert part in edges
        assert part in weights

    # make sure there are no overlapping indices between left and right parts
    assert np.intersect1d(edges["left"], edges["right"]).size == 0

    # three edges remain after masking the left part (between 3 vertices)
    # these would be the edges between 0th and 1st, 1st and 2nd,
    # and 0th and 2nd vertices of the adjacency matrix
    assert_array_equal(edges["left"], np.array([[0, 1, 0], [1, 2, 2]]))
    # three edges remain after masking the right part (between 3 vertices)
    # these would be the edges between 3rd and 4th, 3rd and 5th,
    # and 4th and 5th vertices of the adjacency matrix
    assert_array_equal(edges["right"], np.array([[3, 3, 4], [4, 5, 5]]))

    # weights are computed for each edge
    assert len(weights["left"]) == 3
    assert len(weights["right"]) == 3


def test_rena_clustering_input_mask_as_surface_masker(
    _make_surface_class_data, surf_mask
):
    """Test if ReNA clustering works when mask_img is a SurfaceMasker."""
    masker = SurfaceMasker(surf_mask()).fit()
    surf_img, _ = _make_surface_class_data
    X = masker.transform(surf_img)
    clustering = ReNA(masker, n_clusters=2)
    X_transformed = clustering.fit_transform(X)

    assert X_transformed.shape == (surf_img.shape[0], 2)
