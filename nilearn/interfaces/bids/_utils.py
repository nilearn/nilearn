"""Utility functions for the nilearn.interfaces.bids module."""
import warnings


def _clean_contrast_name(contrast_name):
    """Remove prohibited characters from name and convert to camelCase.

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
