# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Tests for SPM2 header stuff"""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from ..spatialimages import HeaderDataError, HeaderTypeError
from ..spm2analyze import Spm2AnalyzeHeader, Spm2AnalyzeImage
from . import test_spm99analyze


class TestSpm2AnalyzeHeader(test_spm99analyze.TestSpm99AnalyzeHeader):
    header_class = Spm2AnalyzeHeader

    def test_slope_inter(self):
        hdr = self.header_class()
        assert hdr.get_slope_inter() == (1.0, 0.0)
        for in_tup, exp_err, out_tup, raw_slope in (
            ((2.0,), None, (2.0, 0.0), 2.0),
            ((None,), None, (None, None), np.nan),
            ((1.0, None), None, (1.0, 0.0), 1.0),
            # non zero intercept causes error
            ((None, 1.1), HeaderTypeError, (None, None), np.nan),
            ((2.0, 1.1), HeaderTypeError, (None, None), 2.0),
            # null scalings
            ((0.0, None), HeaderDataError, (None, None), 0.0),
            ((np.nan, np.nan), None, (None, None), np.nan),
            ((np.nan, None), None, (None, None), np.nan),
            ((None, np.nan), None, (None, None), np.nan),
            ((np.inf, None), HeaderDataError, (None, None), np.inf),
            ((-np.inf, None), HeaderDataError, (None, None), -np.inf),
            ((None, 0.0), None, (None, None), np.nan),
        ):
            hdr = self.header_class()
            if not exp_err is None:
                with pytest.raises(exp_err):
                    hdr.set_slope_inter(*in_tup)
                # raw set
                if not in_tup[0] is None:
                    hdr['scl_slope'] = in_tup[0]
            else:
                hdr.set_slope_inter(*in_tup)
                assert hdr.get_slope_inter() == out_tup
                # Check set survives through checking
                hdr = Spm2AnalyzeHeader.from_header(hdr, check=True)
                assert hdr.get_slope_inter() == out_tup
            assert_array_equal(hdr['scl_slope'], raw_slope)


class TestSpm2AnalyzeImage(test_spm99analyze.TestSpm99AnalyzeImage):
    # class for testing images
    image_class = Spm2AnalyzeImage


def test_origin_affine():
    # check that origin affine works, only
    hdr = Spm2AnalyzeHeader()
    hdr.get_origin_affine()
