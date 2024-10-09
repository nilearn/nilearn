"""Public Utility functions for the nilearn.interfaces.bids module."""

from __future__ import annotations


def bids_entities():
    """Return a dictionary of BIDS entities.

    Entities are listed in the order they should appear in a filename.

    https://bids-specification.readthedocs.io/en/stable/appendices/entities.html

    Note that:

    - this only contains the entities for functional data

    https://github.com/bids-standard/bids-specification/blob/master/src/schema/rules/files/raw/func.yaml#L13
    https://github.com/bids-standard/bids-specification/blob/master/src/schema/rules/files/deriv/imaging.yaml#L29

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
        "derivatives": ["hemi", "space", "res", "den", "desc"],
    }


def check_bids_label(label):
    """Validate a BIDS label.

    https://bids-specification.readthedocs.io/en/stable/glossary.html#label-formats

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


def create_bids_filename(fields, entities_to_include=None):
    """Create BIDS filename from dictionary of entity-label pairs.

    Parameters
    ----------
    fields : :obj:`dict` of :obj:`str`
        Dictionary of entity-label pairs, for example:

        {
         "suffix": "T1w",
         "extension": "nii.gz",
         "entities": {"acq":  "ap",
                      "desc": "preproc"}
        }.

    Returns
    -------
    BIDS filename : :obj:`str`

    """
    if entities_to_include is None:
        entities_to_include = bids_entities()["raw"]

    filename = ""

    for key in entities_to_include:
        if key in fields["entities"]:
            value = fields["entities"][key]
            if value not in (None, ""):
                filename += f"{key}-{value}_"
    if "suffix" in fields:
        filename += f"{fields['suffix']}"
    if "extension" in fields:
        filename += f".{fields['extension']}"

    return filename
