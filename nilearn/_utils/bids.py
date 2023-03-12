"""BIDS related utilities."""

from typing import Any


def entities() -> dict[str, list[str]]:
    """Return a dictionary of BIDS entities.

    Note that:

    - this only contains the entities for functional data

    Returns
    -------
    Dictionary of raw and derivatives entities : dict[str, list[str]]

    """
    return {
        "raw": [
            "sub",
            "ses",
            "task",
            "acq",
            "ce",
            "rec",
            "dir",
            "run",
            "echo",
            "part",
        ],
        "derivatives": ["space", "res", "den", "desc"],
    }


def validate_label(label: Any) -> None:
    if not isinstance(label, str):
        raise TypeError(
            f"All bids labels must be string. "
            f"Got '{type(label)}' for {label} instead."
        )
    if not all(char.isalnum() for char in label):
        raise ValueError(
            f"All bids labels must be alphanumeric. Got '{label}' instead."
        )
