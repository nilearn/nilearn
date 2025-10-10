"""Fixture for decoding tests."""

import warnings

import pytest


@pytest.fixture(autouse=True)
def suppress_specific_decoding_warning():
    """Ignore internal decoding warnings."""
    with warnings.catch_warnings():
        messages = (
            "Brain mask is bigger.*|"
            "Brain mask is smaller.*|"
            "After clustering.*|"
            "Overriding provided-default estimator.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=messages,
            category=UserWarning,
        )
        yield
