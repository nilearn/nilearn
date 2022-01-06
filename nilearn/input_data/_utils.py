
import sys
import warnings


def _deprecation_warning(deprecated_path, correct_path):
    message = (
        "The import path {deprecated_path} is  deprecated in version "
        "0.9. Importing from {deprecated_path} will be possible at least "
        "until release 0.13.0. Please import from {correct_path} instead."
    ).format(
        deprecated_path=deprecated_path,
        correct_path=correct_path
    )
    if not getattr(sys, '_is_pytest_session', False):
        warnings.warn(message, FutureWarning)
