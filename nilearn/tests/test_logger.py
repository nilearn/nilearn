""" Test the logger module

This test file is in nilearn/tests because nosetests ignores modules whose
name starts with an underscore.
"""
import contextlib
from nose.tools import assert_equal

from sklearn.base import BaseEstimator
from nilearn._utils.logger import log


@contextlib.contextmanager
def capture_output():
    import sys
    from nilearn._utils.compat import StringIO
    oldout, olderr = sys.stdout, sys.stderr
    try:
        out = [StringIO(), StringIO()]
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()


# Helper functions and classes
def run():
    log("function run()")


def other_run():
    # Test too large values for stack_level
    # stack_level should exceed nosetests stack levels as well
    log("function other_run()", stack_level=100)


class Run3(object):

    def run3(self):
        log("method Test3")
        run()


class Run2(BaseEstimator):

    def run2(self):
        log("method Test2")
        t = Run()
        t.run()


class Run(BaseEstimator):

    def run(self):
        log("method Test")
        run()


def test_log():
    # Stack containing one non-matching object
    with capture_output() as out:
        t = Run3()
        t.run3()
    assert_equal(out[0], "[Run3.run3] method Test3\n[run] function run()\n")

    # Stack containing two matching objects
    with capture_output() as out:
        t = Run2()
        t.run2()
    assert_equal(out[0],
                 "[Run2.run2] method Test2\n"
                 "[Run2.run2] method Test\n"
                 "[Run2.run2] function run()\n")

    # Stack containing one matching object
    with capture_output() as out:
        t = Run()
        t.run()
    assert_equal(out[0],
                 "[Run.run] method Test\n[Run.run] function run()\n")

    # Stack containing no object
    with capture_output() as out:
        run()
    assert_equal(out[0], "[run] function run()\n")

    # Test stack_level too large
    with capture_output() as out:
        other_run()
    assert_equal(out[0], "[<top_level>] function other_run()\n")

# Will be executed by nosetests upon importing
with capture_output() as out:
    log("message from no function")
assert_equal(out[0], "[<module>] message from no function\n")
