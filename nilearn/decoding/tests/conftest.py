"""Fixture for decoding tests."""

import warnings
from collections.abc import Generator

import pytest


@pytest.fixture(autouse=True)
def suppress_specific_decoding_warning() -> Generator[None, None, None]:
    """Ignore internal decoding warnings."""
    with warnings.catch_warnings():
        messages = (
            "Brain mask is bigger.*|"
            "Brain mask is smaller.*|"
            "After clustering.*|"
            "Overriding provided-default estimator.*|"
            "The decoding model will be trained only on 12 features.*"
        )
        warnings.filterwarnings(
            "ignore",
            message=messages,
            category=UserWarning,
        )
        yield
