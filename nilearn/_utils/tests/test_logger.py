"""Test the logger module.

This test file is in nilearn/tests because Nosetest,
which we historically used,
ignores modules whose name starts with an underscore.
"""

import contextlib
import sys
from io import StringIO

import pytest
from sklearn.base import BaseEstimator

from nilearn._utils.logger import _has_rich, log


# Helper functions and classes
def run():
    log("function run()")


def other_run():
    # Test too large values for stack_level
    # stack_level should exceed testrunner's stack levels as well
    log("function other_run()", stack_level=100)


class Run3:
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


@pytest.mark.skipif(_has_rich(), reason="Skip test when rich is installed.")
def test_log_2_matching_object(capsys):
    t = Run2()
    t.run2()
    captured = capsys.readouterr()
    assert (
        captured.out == "[Run2.run2] method Test2\n"
        "[Run2.run2] method Test\n"
        "[Run2.run2] function run()\n"
    )


@pytest.mark.skipif(_has_rich(), reason="Skip test when rich is installed.")
def test_log_1_matching_object(capsys):
    t = Run()
    t.run()
    captured = capsys.readouterr()
    assert captured.out == "[Run.run] method Test\n[Run.run] function run()\n"


@pytest.mark.skipif(_has_rich(), reason="Skip test when rich is installed.")
def test_log_no_matching_object(capsys):
    run()
    captured = capsys.readouterr()
    assert captured.out == "[run] function run()\n"


@pytest.mark.skipif(_has_rich(), reason="Skip test when rich is installed.")
def test_log_1_non_matching_object(capsys):
    t = Run3()
    t.run3()
    captured = capsys.readouterr()
    assert captured.out == "[Run3.run3] method Test3\n[run] function run()\n"


def test_log_stack_lvl_stack_too_large(capsys):
    """Test rich and non rich output."""
    other_run()
    captured = capsys.readouterr()
    if _has_rich():
        from rich.console import Console

        console = Console(file=StringIO(), width=120)
        console.print("[blue]\\[<top_level>][/blue] function other_run()")
        output = console.file.getvalue()
        assert captured.out == output
    else:
        assert captured.out == "[<top_level>] function other_run()\n"


@contextlib.contextmanager
def capture_output():
    oldout, olderr = sys.stdout, sys.stderr
    try:
        out = [StringIO(), StringIO()]
        sys.stdout, sys.stderr = out
        yield out
    finally:
        sys.stdout, sys.stderr = oldout, olderr
        out[0] = out[0].getvalue()
        out[1] = out[1].getvalue()


# Will be executed by testrunner upon importing
with capture_output() as out:
    log("message from no function")
    if _has_rich() is False:
        if isinstance(out[0], StringIO):
            assert out[0].getvalue() == "[<module>] message from no function\n"
        else:
            assert out[0] == "[<module>] message from no function\n"
