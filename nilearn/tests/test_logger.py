""" Test the logger module

This test file is in nilearn/tests because nosetests ignores modules whose
name starts with an underscore.
"""

from sklearn.base import BaseEstimator
from nilearn._utils.logger import log


# Helper functions and classes
def run():
    log("function run()")


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
    # Do smoke tests only.

    # Stack containing one non-matching object
    t = Run3()
    t.run3()

    # Stack containing two matching objects
    t = Run2()
    t.run2()

    # Stack containing one matching object
    t = Run()
    t.run()

    # Stack containing no object
    run()

# Will be executed by nosetests upon importing
log("message from no function")
