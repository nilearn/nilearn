# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Utilities for calculations related to MRI"""

__all__ = ['calculate_dwell_time']

GYROMAGNETIC_RATIO = 42.576  # MHz/T for hydrogen nucleus
PROTON_WATER_FAT_SHIFT = 3.4  # ppm


class MRIError(ValueError):
    pass


def calculate_dwell_time(water_fat_shift, echo_train_length, field_strength):
    """Calculate the dwell time

    Parameters
    ----------
    water_fat_shift : float
        The water fat shift of the recording, in pixels.
    echo_train_length : int
        The echo train length of the imaging sequence.
    field_strength : float
        Strength of the magnet in Tesla, e.g. 3.0 for a 3T magnet recording.

    Returns
    -------
    dwell_time : float
        The dwell time in seconds.

    Raises
    ------
    MRIError
        if values are out of range
    """
    if field_strength < 0:
        raise MRIError('Field strength should be positive')
    if echo_train_length <= 0:
        raise MRIError('Echo train length should be >= 1')
    return (
        (echo_train_length - 1)
        * water_fat_shift
        / (GYROMAGNETIC_RATIO * PROTON_WATER_FAT_SHIFT * field_strength * (echo_train_length + 1))
    )
