from pathlib import Path

import pandas as pd
import pytest

from nilearn._estimator_checks.nilearn_checks import (
    nilearn_check_estimator,
)
from nilearn._estimator_checks.tests.conftest import (
    CONNECTOME,
    DECODING,
    DECOMPOSITION,
    ESTIMATORS_TO_CHECK,
    GLM,
    MASKERS,
    REGIONS,
)
from nilearn.utils.discovery import all_estimators


def _estimators_to_test():
    """Create list of estimators to be used for nilearn checks.

    Nilearn estimator checks should be run only for the estimators whose
    package is modified. The list of modified packages is taken from from
    ``tests_to_run.txt`` file. This file is generated only when tests are run
    in CI. To generate it locally,
    `` python build_tools/github/restrict_tests_to_run.py`` command must be run
    in command line before running tests.

    If ``tests_to_run.txt`` file does not exist, it returns all estimators.
    """
    path = Path(__file__).resolve().parent.parent.parent / "tests_to_run.txt"
    if path.exists():
        data = pd.read_csv(path, sep=" ")
        packages = data.columns
        estimator_list = []

        if packages is not None:
            for package in packages:
                if "connectome" in package:
                    estimator_list.extend(CONNECTOME)
                elif "decoding" in package:
                    estimator_list.extend(DECODING)
                elif "decomposition" in package:
                    estimator_list.extend(DECOMPOSITION)
                elif "glm" in package:
                    estimator_list.extend(GLM)
                elif "maskers" in package:
                    estimator_list.extend(MASKERS)
                elif "regions" in package:
                    estimator_list.extend(REGIONS)
    else:
        estimator_list = ESTIMATORS_TO_CHECK

    return estimator_list


def test_check_estimator_count():
    """Test if all estimators provided by nilearn are covered by
    ESTIMATORS_TO_CHECK.
    """
    assert len({est.__class__ for est in ESTIMATORS_TO_CHECK}) == len(
        all_estimators()
    )


@pytest.mark.slow
@pytest.mark.flaky(reruns=1, reruns_delay=2)
@pytest.mark.parametrize(
    "estimator, name, check",
    nilearn_check_estimator(estimators=_estimators_to_test()),
)
def test_check_estimator_nilearn(estimator, name, check):  # noqa: ARG001
    """Check compliance with nilearn estimators rules."""
    check(estimator)
