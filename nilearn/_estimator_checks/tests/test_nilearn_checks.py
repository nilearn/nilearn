import pytest

from nilearn._estimator_checks.nilearn_checks import (
    CACHE_MIXIN_CHECKS,
    COMMON_CHECKS,
    GLM_CHECKS,
    IMAGE_INPUT_COMMON_CHECKS,
    nilearn_check_generator,
)
from nilearn._estimator_checks.tests.conftest import (
    ESTIMATORS_TO_CHECK,
    GLM,
)


@pytest.mark.parametrize(
    "estimator, expected_checks",
    [(e, COMMON_CHECKS) for e in ESTIMATORS_TO_CHECK]
    + [
        (
            e,
            COMMON_CHECKS
            + CACHE_MIXIN_CHECKS
            + IMAGE_INPUT_COMMON_CHECKS
            + GLM_CHECKS,
        )
        for e in GLM
    ],
)
def test_nilearn_check_generator_common_checks(estimator, expected_checks):
    checks_found = 0
    for check in nilearn_check_generator(estimator):
        checks_found += 1 if check in expected_checks else 0
    assert checks_found == len(expected_checks)
