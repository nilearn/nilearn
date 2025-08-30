"""Custom warnings and errors used acrossnilearn."""

__all__ = [
    "NotImplementedWarning",
]


class NotImplementedWarning(UserWarning):
    """Custom warning to warn about not implemented features.

    .. versionadded:: 0.13.0
    """
