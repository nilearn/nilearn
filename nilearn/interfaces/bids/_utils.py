"""Utility functions for the nilearn.interfaces.bids module."""

from __future__ import annotations

import json
import warnings

import nilearn


def _clean_contrast_name(contrast_name):
    """Remove prohibited characters from name and convert to camelCase.

    .. versionadded:: 0.9.2

    BIDS filenames, in which the contrast name will appear as a
    contrast-<name> key/value pair, must be alphanumeric strings.

    Parameters
    ----------
    contrast_name : :obj:`str`
        Contrast name to clean.

    Returns
    -------
    new_name : :obj:`str`
        Contrast name converted to alphanumeric-only camelCase.
    """
    new_name = contrast_name[:]

    # Some characters translate to words
    new_name = new_name.replace('-', ' Minus ')
    new_name = new_name.replace('+', ' Plus ')
    new_name = new_name.replace('>', ' Gt ')
    new_name = new_name.replace('<', ' Lt ')

    # Others translate to spaces
    new_name = new_name.replace('_', ' ')

    # Convert to camelCase
    new_name = new_name.split(' ')
    new_name[0] = new_name[0].lower()
    new_name[1:] = [c.title() for c in new_name[1:]]
    new_name = ' '.join(new_name)

    # Remove non-alphanumeric characters
    new_name = ''.join(ch for ch in new_name if ch.isalnum())

    # Let users know if the name was changed
    if new_name != contrast_name:
        warnings.warn(
            f'Contrast name "{contrast_name}" changed to "{new_name}"'
        )
    return new_name


def _generate_model_metadata(out_file, model):
    """Generate a sidecar JSON file containing model metadata.

    .. versionadded:: 0.9.2

    Parameters
    ----------
    out_file : :obj:`str`
        Output JSON filename, to be created by the function.
    model : :obj:`~nilearn.glm.first_level.FirstLevelModel` or
            :obj:`~nilearn.glm.second_level.SecondLevelModel`
        First- or second-level model from which to save outputs.
    """
    # Define which FirstLevelModel attributes are BIDS compliant and which
    # should be bundled in a new "ModelParameters" field.
    DATA_ATTRIBUTES = [
        't_r',
    ]
    PARAMETER_ATTRIBUTES = [
        'drift_model',
        'hrf_model',
        'standardize',
        'high_pass',
        'target_shape',
        'signal_scaling',
        'drift_order',
        'scaling_axis',
        'smoothing_fwhm',
        'target_affine',
        'slice_time_ref',
        'fir_delays',
    ]
    ATTRIBUTE_RENAMING = {
        't_r': 'RepetitionTime',
    }

    # Fields for the top level of the dictionary
    DATA_ATTRIBUTES.sort()
    data_attributes = {
        attr_name: getattr(model, attr_name)
        for attr_name in DATA_ATTRIBUTES
        if hasattr(model, attr_name)
    }
    data_attributes = {
        ATTRIBUTE_RENAMING.get(k, k): v for k, v in data_attributes.items()
    }

    # Fields for a nested section of the dictionary
    # The ModelParameters field is an ad-hoc way to retain useful info.
    PARAMETER_ATTRIBUTES.sort()
    model_attributes = {
        attr_name: getattr(model, attr_name)
        for attr_name in PARAMETER_ATTRIBUTES
        if hasattr(model, attr_name)
    }
    model_attributes = {
        ATTRIBUTE_RENAMING.get(k, k): v for k, v in model_attributes.items()
    }

    model_metadata = {
        'Description': 'A statistical map generated by Nilearn.',
        **data_attributes,
        'ModelParameters': model_attributes,
    }

    with open(out_file, 'w') as f_obj:
        json.dump(model_metadata, f_obj, indent=4, sort_keys=True)


def _generate_dataset_description(out_file, model_level):
    """Generate a BIDS dataset_description.json file with relevant metadata.

    .. versionadded:: 0.9.2

    Parameters
    ----------
    out_file : :obj:`str`
        Output JSON filename, to be created by the function.
    model_level : {1, 2}
        The level of the model. 1 means a first-level model.
        2 means a second-level model.
    """
    dataset_description = {
        'GeneratedBy': {
            'Name': 'nilearn',
            'Version': nilearn.__version__,
            'Description': 'A Nilearn {} GLM.'.format(
                'first-level' if model_level == 1 else 'second-level'
            ),
            'CodeURL': (
                'https://github.com/nilearn/nilearn/releases/tag/'
                '{}'.format(nilearn.__version__)
            )
        }
    }

    with open(out_file, 'w') as f_obj:
        json.dump(dataset_description, f_obj, indent=4, sort_keys=True)


def _bids_entities():
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


def _check_bids_label(label):
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
