"""Custom warnings and errors used across Nilearn."""

__all__ = [
    "NotImplementedWarning",
]


class NotImplementedWarning(UserWarning):
    """Custom warning to warn about not implemented features.

    .. versionadded:: 0.13.0
    """
