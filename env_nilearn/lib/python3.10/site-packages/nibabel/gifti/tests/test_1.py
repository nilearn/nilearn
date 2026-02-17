"""Testing loading of gifti file

The file is ``test_1`` because we are testing a bug where, if we try to load a
file before instantiating some Gifti objects, loading fails with an
AttributeError (see: https://github.com/nipy/nibabel/issues/392).

Thus, we have to run this test before the other gifti tests to catch the gifti
code unprepared.
"""

from nibabel import load

from .test_parse_gifti_fast import DATA_FILE3


def test_load_gifti():
    # This expression should not raise an error
    load(DATA_FILE3)
