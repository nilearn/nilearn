"""Test design matrix creation and manipulation."""

import numpy as np
import pytest

# Use nisignal here to avoid name collisions
# (using nilearn.signal is not possible)
from nilearn import signal as nisignal
from nilearn._utils.data_gen import generate_signals
from nilearn.glm.first_level.design_matrix import _cosine_drift
from numpy.testing import assert_almost_equal, assert_array_equal


@pytest.fixture
def set_up():
    # fmriprep high pass cutoff is 128s, it's around 0.008 hz
    t_r = 2.5
    high_pass = 0.008
    signals, _, confounds = generate_signals(
        n_features=41, n_confounds=5, length=45
    )
    return signals, confounds, t_r, high_pass


def _get_drift_terms(signals, confounds=None, t_r=2.5, high_pass=0.008):
    frame_times = np.arange(signals.shape[0]) * t_r
    cosine_drift = _cosine_drift(high_pass, frame_times)[:, :-1]
    return (
        np.hstack((confounds, cosine_drift))
        if confounds is not None
        else cosine_drift
    )


def test_create_cosine_drift_terms(set_up):
    """Testing cosine filter interface and output.

    Passing confounds it will return the confounds
    """
    signals, confounds, t_r, high_pass = set_up

    cosine_confounds = nisignal._create_cosine_drift_terms(
        signals=signals, confounds=confounds, high_pass=high_pass, t_r=t_r
    )

    assert_almost_equal(cosine_confounds, _get_drift_terms(signals, confounds))


def test_create_cosine_drift_terms_no_confounds(set_up):
    """Testing cosine filter interface and output.

    Not passing confounds it will return drift terms only
    """

    signals, _, t_r, high_pass = set_up

    drift_terms_only = nisignal._create_cosine_drift_terms(
        signals=signals, confounds=None, high_pass=high_pass, t_r=t_r
    )

    assert_almost_equal(drift_terms_only, _get_drift_terms(signals))


def test_create_cosine_drift_terms_warnings(set_up):
    """Testing cosine filter interface and output.

    Passing confounds it will return the confounds
    """
    signals, confounds, t_r, high_pass = set_up

    # drift terms in confounds will create warning and no change to confounds
    confounds_with_drift = _get_drift_terms(signals, confounds)

    with pytest.warns(UserWarning, match="user supplied confounds"):
        cosine_confounds = nisignal._create_cosine_drift_terms(
            signals, confounds_with_drift, high_pass, t_r
        )
    assert_array_equal(cosine_confounds, confounds_with_drift)


def test_create_cosine_drift_terms_warning_filter_not_created(set_up):
    """Raise warning if cosine drift term is not created."""
    signals, confounds, t_r, _ = set_up
    high_pass_fail = 0.002

    with pytest.warns(UserWarning, match="Cosine filter was not created"):
        cosine_confounds = nisignal._create_cosine_drift_terms(
            signals, confounds, high_pass_fail, t_r
        )

    assert_array_equal(cosine_confounds, confounds)
