"""Test tmpdirs module"""

from os import getcwd
from os.path import abspath, dirname, isfile, realpath

from ..tmpdirs import InGivenDirectory

MY_PATH = abspath(__file__)
MY_DIR = dirname(MY_PATH)


def test_given_directory():
    # Test InGivenDirectory
    cwd = getcwd()
    with InGivenDirectory() as tmpdir:
        assert tmpdir == abspath(cwd)
        assert tmpdir == abspath(getcwd())
    with InGivenDirectory(MY_DIR) as tmpdir:
        assert tmpdir == MY_DIR
        assert realpath(MY_DIR) == realpath(abspath(getcwd()))
    # We were deleting the Given directory!  Check not so now.
    assert isfile(MY_PATH)
