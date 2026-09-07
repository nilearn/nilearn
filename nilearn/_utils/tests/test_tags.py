"""Check Nilearn estimators tags."""

from nilearn._base import NilearnBaseEstimator
from nilearn._utils.tags import InputTags


class NilearnEstimator(NilearnBaseEstimator):
    """Dummy estimator that takes surface image but not nifti as inputs."""

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=False)
        return tags


def test_nilearn_tags():
    """Check that adding tags to Nilearn estimators work as expected.

    Especially with different sklearn versions.
    """
    est = NilearnEstimator()

    tags = est.__sklearn_tags__()

    assert not tags.input_tags.niimg_like
    assert tags.input_tags.surf_img
    # making sure 2darray still here
    # as it allows to run some sklearn checks
    assert tags.input_tags.two_d_array
