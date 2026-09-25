"""Check Nilearn estimators tags."""

from nilearn._base import NilearnBaseEstimator
from nilearn.utils import InputTags
from nilearn.utils.tags import get_input_tags


class NilearnEstimator(NilearnBaseEstimator):
    """Dummy estimator that takes surface image but not nifti as inputs."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=False)
        return tags


def test_nilearn_tags():
    """Check that adding tags to Nilearn estimators work as expected."""
    est = NilearnEstimator()

    tags = est.__sklearn_tags__()

    assert not tags.input_tags.niimg_like
    assert tags.input_tags.surf_img
    # making sure 2darray still here
    # as it allows to run some sklearn checks
    assert tags.input_tags.two_d_array


def test_get_tag():
    """Check reading a tag from an estimator."""
    est = NilearnEstimator()

    assert get_input_tags(est, "surf_img")
    assert not get_input_tags(est, "niimg_like")
    assert not get_input_tags(est, "unknown_tag")

    # objects without __sklearn_tags__ have no tags
    assert not get_input_tags(object(), "niimg_like")
