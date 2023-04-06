"""Test design matrix creation and manipulation."""

import numpy as np
import pytest

# Use nisignal here to avoid name collisions
# (using nilearn.signal is not possible)
from nilearn import signal as nisignal
from nilearn._utils.data_gen import generate_signals
from nilearn.glm.first_level.design_matrix import _cosine_drift
from numpy.testing import assert_almost_equal, assert_array_equal


def test_create_cosine_drift_terms():
    """Testing cosine filter interface and output."""
    # fmriprep high pass cutoff is 128s, it's around 0.008 hz
    t_r, high_pass = 2.5, 0.008
    signals, _, confounds = generate_signals(
        n_features=41, n_confounds=5, length=45
    )

    # Not passing confounds it will return drift terms only
    frame_times = np.arange(signals.shape[0]) * t_r
    cosine_drift = _cosine_drift(high_pass, frame_times)[:, :-1]

    cosine_confounds = nisignal._create_cosine_drift_terms(
        signals, confounds, high_pass, t_r
    )
    assert_almost_equal(cosine_confounds, np.hstack((confounds, cosine_drift)))

    # Not passing confounds it will return drift terms only
    drift_terms_only = nisignal._create_cosine_drift_terms(
        signals, None, high_pass, t_r
    )
    assert_almost_equal(drift_terms_only, cosine_drift)


def test_create_cosine_drift_terms_warnings():
    """Testing cosine filter interface and output."""
    # fmriprep high pass cutoff is 128s, it's around 0.008 hz
    t_r, high_pass = 2.5, 0.008
    signals, _, confounds = generate_signals(
        n_features=41, n_confounds=5, length=45
    )

    # Not passing confounds it will return drift terms only
    frame_times = np.arange(signals.shape[0]) * t_r
    cosine_drift = _cosine_drift(high_pass, frame_times)[:, :-1]
    confounds_with_drift = np.hstack((confounds, cosine_drift))

    # drift terms in confounds will create warning and no change to confounds
    with pytest.warns(UserWarning, match="user supplied confounds"):
        cosine_confounds = nisignal._create_cosine_drift_terms(
            signals, confounds_with_drift, high_pass, t_r
        )
    assert_array_equal(cosine_confounds, confounds_with_drift)

    # raise warning if cosine drift term is not created
    high_pass_fail = 0.002
    with pytest.warns(UserWarning, match="Cosine filter was not create"):
        cosine_confounds = nisignal._create_cosine_drift_terms(
            signals, confounds, high_pass_fail, t_r
        )
    assert_array_equal(cosine_confounds, confounds)
