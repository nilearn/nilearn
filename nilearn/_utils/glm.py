"""Misc utilities for the library.

Authors: Bertrand Thirion, Matthew Brett, Ana Luisa Pinho, 2020
"""
import numpy as np


def _check_run_sample_masks(n_runs, sample_masks):
    """Check that number of sample_mask matches number of runs."""
    if not isinstance(sample_masks, (list, tuple, np.ndarray)):
        raise TypeError(
            f"sample_mask has an unhandled type: {sample_masks.__class__}"
        )

    if isinstance(sample_masks, np.ndarray):
        sample_masks = (sample_masks,)

    checked_sample_masks = [_convert_bool2index(sm) for sm in sample_masks]

    if len(checked_sample_masks) != n_runs:
        raise ValueError(
            f"Number of sample_mask ({len(checked_sample_masks)}) not "
            f"matching number of runs ({n_runs})."
        )
    return checked_sample_masks


def _convert_bool2index(sample_mask):
    """Convert boolean to index."""
    check_boolean = [
        type(i) is bool or type(i) is np.bool_ for i in sample_mask
    ]
    if all(check_boolean):
        sample_mask = np.where(sample_mask)[0]
    return sample_mask
