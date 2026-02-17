# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for imageglobals module"""

from .. import imageglobals as igs


def test_errorlevel():
    orig_level = igs.error_level
    for level in (10, 20, 30):
        with igs.ErrorLevel(level):
            assert igs.error_level == level
        assert igs.error_level == orig_level
