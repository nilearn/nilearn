import sys

import numpy as np
import pytest

# Ignore warning requesting help with nicom
with pytest.warns(UserWarning):
    import nibabel.nicom  # noqa: F401


@pytest.fixture(scope='session', autouse=True)
def legacy_printoptions():
    np.set_printoptions(legacy='1.21')


@pytest.fixture
def max_digits():
    # Set maximum number of digits for int/str conversion for
    # duration of a test
    try:
        orig_max_str_digits = sys.get_int_max_str_digits()
        yield sys.set_int_max_str_digits
        sys.set_int_max_str_digits(orig_max_str_digits)
    except AttributeError:  # PY310 # pragma: no cover
        # Nothing to do for versions of Python that lack these methods
        # They were added as DoS protection in Python 3.11 and backported to
        # some other versions.
        yield lambda x: None
