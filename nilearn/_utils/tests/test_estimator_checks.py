"""
Test the class_inspect module.

This test file is in nilearn/tests because Nosetest,
which we historically used,
ignores modules whose name starts with an underscore.
"""

import pytest

from nilearn._utils.estimator_checks import (
    check_estimator_has_sklearn_is_fitted,
)


def test_check_estimator_has_sklearn_is_fitted():
    """Check errors are thrown for unfitted estimator.

    Check that before fitting
    - estimlator has a __sklearn_is_fitted__ method that returns false
    - running sklearn check_is_fitted on masker throws an error
    """

    class DummyEstimator:
        def __init__(self):
            pass

    with pytest.raises(
        TypeError, match="must have __sklearn_is_fitted__ method"
    ):
        check_estimator_has_sklearn_is_fitted(DummyEstimator())

    class DummyEstimator:
        def __init__(self):
            pass

        def __sklearn_is_fitted__(self):
            return True

    with pytest.raises(ValueError, match="must return False before fit"):
        check_estimator_has_sklearn_is_fitted(DummyEstimator())
