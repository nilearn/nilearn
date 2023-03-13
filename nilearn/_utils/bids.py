"""BIDS related utilities."""

from __future__ import annotations

from typing import Any


def bids_entities() -> dict[str, list[str]]:
    """Return a dictionary of BIDS entities.

    Entities are listed in the order they should appear in a filename.

    https://bids-specification.readthedocs.io/en/stable/appendices/entities.html # noqa

    Note that:

    - this only contains the entities for functional data

    https://github.com/bids-standard/bids-specification/blob/master/src/schema/rules/files/raw/func.yaml#L13 # noqa
    https://github.com/bids-standard/bids-specification/blob/master/src/schema/rules/files/deriv/imaging.yaml#L29 # noqa

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


def validate_bids_label(label) -> None:
    """Validate a BIDS label.

    https://bids-specification.readthedocs.io/en/stable/glossary.html#label-formats # noqa

    Parameters
    ----------
    label : Any
        Label to validate

    """
    if not isinstance(label, str):
        raise TypeError(
            f"All bids labels must be string. "
            f"Got '{type(label)}' for {label} instead."
        )
    if not all(char.isalnum() for char in label):
        raise ValueError(
            f"All bids labels must be alphanumeric. Got '{label}' instead."
        )
