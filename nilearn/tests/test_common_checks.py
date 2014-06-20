"""
Tests for the _utils/common_checks module.

This test file is in nilearn/tests because nosetests seems to ignore
modules whose name starts with an underscore.

"""
#Author: Virgile Fritsch, <virgile.fritsch@inria.fr>, Jun. 2014
from nose.tools import (assert_equal, assert_less, assert_greater_equal,
                        assert_raises)

from nilearn._utils.common_checks import check_n_jobs


def test_check_n_jobs():
    """Test check_n_jobs function..
    """
    # Standard case
    n_jobs_valid = 1
    n_jobs_safe = check_n_jobs(n_jobs_valid)
    assert_equal(n_jobs_valid, n_jobs_safe)

    # Too many CPUs case
    n_jobs_large = 100000000
    n_jobs_safe = check_n_jobs(n_jobs_large)
    assert_less(n_jobs_safe, n_jobs_large)
    assert_greater_equal(n_jobs_safe, 1)

    # Too many CPUs ignored
    n_jobs_not_enough = -100000000  # all CPUs except 100000000
    n_jobs_safe = check_n_jobs(n_jobs_not_enough)
    assert_equal(n_jobs_safe, 1)

    # Invalid value
    n_jobs = 0
    assert_raises(ValueError, check_n_jobs, n_jobs)
