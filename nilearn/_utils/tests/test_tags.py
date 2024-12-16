"""Check Nilearn estimators tags."""

from sklearn.base import BaseEstimator

from nilearn._utils.tags import SKLEARN_LT_1_6


class NilearnEstimator(BaseEstimator):
    """Dummy estimator that takes surface image but not nifti as inputs."""

    def __sklearn_tags__(self):
        # TODO
        # get rid of if block
        # bumping sklearn_version > 1.5
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(surf_img=True, niimg_like=False)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=False)
        return tags


def test_nilearn_tags():
    """Check that adding tags to Nilearn estimators work as expected.

    Especially with different sklearn versions.
    """
    est = NilearnEstimator()

    tags = est.__sklearn_tags__()
    if SKLEARN_LT_1_6:
        assert "niimg_like" not in tags["X_types"]
        assert "surf_img" in tags["X_types"]
        # making sure 2darray still here
        # as it allows to run some sklearn checks
        assert "2darray" in tags["X_types"]
    else:
        assert not tags.input_tags.niimg_like
        assert tags.input_tags.surf_img
        # making sure 2darray still here
        # as it allows to run some sklearn checks
        assert tags.input_tags.two_d_array
