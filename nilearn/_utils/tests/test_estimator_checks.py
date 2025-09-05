"""
Test the class_inspect module.

This test file is in nilearn/tests because Nosetest,
which we historically used,
ignores modules whose name starts with an underscore.
"""

import pytest

from nilearn._utils.estimator_checks import (
    check_img_estimator_dict_unchanged,
    check_img_estimator_fit_check_is_fitted,
)
from nilearn.maskers.base_masker import BaseMasker


def test_check_estimator_has_sklearn_is_fitted():
    """Check errors are thrown for unfitted estimator.

    Check that before fitting
    - estimator has a __sklearn_is_fitted__ method that returns false
    - running sklearn check_is_fitted on masker throws an error
    """

    class DummyEstimator:
        def __init__(self):
            pass

    with pytest.raises(
        TypeError, match="must have __sklearn_is_fitted__ method"
    ):
        check_img_estimator_fit_check_is_fitted(DummyEstimator())

    class DummyEstimator:
        def __init__(self):
            pass

        def __sklearn_is_fitted__(self):
            return True

    with pytest.raises(ValueError, match="must return False before fit"):
        check_img_estimator_fit_check_is_fitted(DummyEstimator())


def test_check_masker_dict_unchanged():
    class DummyEstimator(BaseMasker):
        """Estimator with a transform method that adds a new attribute."""

        def __init__(self, mask_img=None):
            self.mask_img = mask_img

        def fit(self, imgs):
            self.imgs = imgs
            return self

        def transform(self, imgs):
            self._imgs = imgs

    estimator = DummyEstimator()

    with pytest.raises(
        ValueError, match="Estimator changes '__dict__' keys during transform."
    ):
        check_img_estimator_dict_unchanged(estimator)

    class DummyEstimator(BaseMasker):
        """Estimator with a transform method that modifies an attribute."""

        def __init__(self, mask_img=None):
            self.mask_img = mask_img

        def fit(self, imgs):
            self.imgs = imgs
            return self

        def transform(self, imgs):
            del imgs
            self.imgs = 1

    estimator = DummyEstimator()

    with pytest.raises(
        ValueError, match="Estimator changes the following '__dict__' keys"
    ):
        check_img_estimator_dict_unchanged(estimator)
