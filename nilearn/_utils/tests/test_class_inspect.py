"""
Test the class_inspect module.

This test file is in nilearn/tests because Nosetest,
which we historically used,
ignores modules whose name starts with an underscore.
"""

from packaging.version import parse
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator

from nilearn._utils import class_inspect

##############################################################################
# Helpers for the tests


class A(BaseEstimator):
    def __init__(self, a=1):
        self.a = a


class B(A):
    def __init__(self, a=1, b=2):
        self.a = a
        self.b = b


##############################################################################
# The tests themselves


def test_get_params():
    b = B()
    params_a_in_b = class_inspect.get_params(A, b)
    assert params_a_in_b == {"a": 1}
    params_a_in_b = class_inspect.get_params(A, b, ignore=["a"])
    assert params_a_in_b == {}


def test_nilearn_tags():
    """Check that adding tags to Nilearn estimators work as expected."""
    sklearn_lt_1_6 = parse(sklearn_version).release[1] < 6

    class NilearnEstimator(BaseEstimator):
        """Estimator that takes surface image but not nifti as inputs."""

        def __sklearn_tags__(self):
            # TODO
            # get rid of if block
            # bumping sklearn_version > 1.5
            if sklearn_lt_1_6:
                from nilearn._utils.class_inspect import tags

                return tags(surf_img=True, niimg_like=False)
            from nilearn._utils.class_inspect import InputTags

            tags = super().__sklearn_tags__()
            tags.input_tags = InputTags(surf_img=True, niimg_like=False)
            return tags

    est = NilearnEstimator()

    tags = est.__sklearn_tags__()
    if sklearn_lt_1_6:
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
