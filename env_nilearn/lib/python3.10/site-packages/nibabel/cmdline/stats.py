#!python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""
Compute image statistics
"""

import argparse

from nibabel.imagestats import count_nonzero_voxels, mask_volume
from nibabel.loadsave import load


def _get_parser():
    """Return command-line argument parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('infile', help='Neuroimaging volume to compute statistics on.')
    p.add_argument(
        '-V',
        '--Volume',
        action='store_true',
        required=False,
        help='Compute mask volume of a given mask image.',
    )
    p.add_argument(
        '--units',
        default='mm3',
        required=False,
        choices=('mm3', 'vox'),
        help='Preferred output units',
    )
    return p


def main(args=None):
    """Main program function."""
    parser = _get_parser()
    opts = parser.parse_args(args)
    from_img = load(opts.infile)

    if opts.Volume:
        if opts.units == 'mm3':
            computed_volume = mask_volume(from_img)
        elif opts.units == 'vox':
            computed_volume = count_nonzero_voxels(from_img)
        else:
            raise ValueError(f'{opts.units} is not a valid unit. Choose "mm3" or "vox".')
        print(computed_volume)
        return 0
