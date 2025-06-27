import numpy as np
import pytest
from joblib import Memory
from numpy.testing import assert_array_equal
from sklearn.utils.estimator_checks import parametrize_with_checks

from nilearn._utils.data_gen import generate_fake_fmri
from nilearn._utils.estimator_checks import (
    check_estimator,
    nilearn_check_estimator,
    return_expected_failed_checks,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.conftest import _img_3d_mni, _shape_3d_default
from nilearn.image import get_data
from nilearn.maskers import NiftiMasker, SurfaceMasker
from nilearn.regions.rena_clustering import (
    ReNA,
    _make_edges_and_weights_surface,
    make_edges_surface,
)
from nilearn.surface import SurfaceImage

ESTIMATORS_TO_CHECK = [ReNA()]

if SKLEARN_LT_1_6:

    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK),
    )
    def test_check_estimator_sklearn_valid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

    @pytest.mark.xfail(reason="invalid checks should fail")
    @pytest.mark.parametrize(
        "estimator, check, name",
        check_estimator(estimators=ESTIMATORS_TO_CHECK, valid=False),
    )
    def test_check_estimator_sklearn_invalid(estimator, check, name):  # noqa: ARG001
        """Check compliance with sklearn estimators."""
        check(estimator)

else:

    @parametrize_with_checks(
        estimators=ESTIMATORS_TO_CHECK,
        expected_failed_checks=return_expected_failed_checks,
    )
    def test_check_estimator_sklearn(estimator, check):
        """Check compliance with sklearn estimators."""
        check(estimator)


@pytest.mark.parametrize(
    "estimator, check, name",
    nilearn_check_estimator(
        estimators=[ReNA(mask_img=_img_3d_mni(), n_clusters=2)]
    ),
)
def test_check_estimator_nilearn(estimator, check, name):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)


def test_rena_clustering_mask_error():
    """Check an error is raised if invalid mask is provided before fit."""
    data_img, mask_img = generate_fake_fmri(
        shape=_shape_3d_default(), length=5
    )
    rena = ReNA(n_clusters=10, mask_img=1)

    data = get_data(data_img)
    mask = get_data(mask_img)

    X = np.empty((data.shape[3], int(mask.sum())))
    for i in range(data.shape[3]):
        X[i, :] = np.copy(data[:, :, :, i])[get_data(mask_img) != 0]

    with pytest.raises(TypeError, match="The mask image should be a"):
        rena.fit_transform(X)


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


@pytest.mark.parametrize("part", ["left", "right"])
def test_make_edges_surface(surf_mask_1d, part):
    """Test if the edges and edge mask are correctly computed."""
    faces = surf_mask_1d.mesh.parts[part].faces
    # the mask for left part has total 4 vertices out of which 2 are True
    # and for right part it has total 5 vertices out of which 3 are True
    mask = surf_mask_1d.data.parts[part]
    edges_unmasked, edges_mask = make_edges_surface(faces, mask)

    # only one edge remains after masking the left part (between 2 vertices)
    if part == "left":
        assert edges_unmasked[:, edges_mask].shape == (2, 1)
    # three edges remain after masking the right part (between 3 vertices)
    elif part == "right":
        assert edges_unmasked[:, edges_mask].shape == (2, 3)


def test_make_edges_and_weights_surface(surf_mesh, surf_img_2d):
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
    surf_mask_1d = SurfaceImage(surf_mesh, data)
    # create a surface masker
    masker = SurfaceMasker(surf_mask_1d).fit()
    # mask the surface image with 50 samples
    X = masker.transform(surf_img_2d(50))
    # compute edges and weights
    edges, weights = _make_edges_and_weights_surface(X, surf_mask_1d)

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


@pytest.mark.parametrize("surf_mask_dim", [1, 2])
@pytest.mark.parametrize("mask_as", ["surface_image", "surface_masker"])
@pytest.mark.parametrize("n_clusters", [2, 4, 5])
def test_rena_clustering_input_mask_surface(
    surf_img_2d, surf_mask_dim, surf_mask_1d, surf_mask_2d, mask_as, n_clusters
):
    """Test if ReNA clustering works in both cases when mask_img is either a
    SurfaceImage or SurfaceMasker.
    """
    surf_mask = surf_mask_1d if surf_mask_dim == 1 else surf_mask_2d()
    # create a surface masker
    masker = SurfaceMasker(surf_mask).fit()
    # mask the surface image with 50 samples
    X = masker.transform(surf_img_2d(50))
    if mask_as == "surface_image":
        # instantiate ReNA with mask_img as a SurfaceImage
        clustering = ReNA(mask_img=surf_mask, n_clusters=n_clusters)
    elif mask_as == "surface_masker":
        # instantiate ReNA with mask_img as a SurfaceMasker
        clustering = ReNA(mask_img=masker, n_clusters=n_clusters)
    # fit and transform the data
    X_transformed = clustering.fit_transform(X)
    # inverse transform the transformed data
    X_inverse = clustering.inverse_transform(X_transformed)

    # make sure the n_features in transformed data were reduced to n_clusters
    assert X_transformed.shape[1] == n_clusters

    # make sure the inverse transformed data has the same shape as the original
    assert X_inverse.shape == X.shape
