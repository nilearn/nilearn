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
Conform neuroimaging volume to arbitrary shape and voxel size.
"""

import argparse
from pathlib import Path

from nibabel import __version__
from nibabel.loadsave import load, save
from nibabel.processing import conform


def _get_parser():
    """Return command-line argument parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('infile', help='Neuroimaging volume to conform.')
    p.add_argument('outfile', help='Name of output file.')
    p.add_argument(
        '--out-shape',
        nargs=3,
        default=(256, 256, 256),
        type=int,
        help='Shape of the conformed output.',
    )
    p.add_argument(
        '--voxel-size',
        nargs=3,
        default=(1, 1, 1),
        type=int,
        help='Voxel size in millimeters of the conformed output.',
    )
    p.add_argument('--orientation', default='RAS', help='Orientation of the conformed output.')
    p.add_argument('-f', '--force', action='store_true', help='Overwrite existing output files.')
    p.add_argument('-V', '--version', action='version', version=f'{p.prog} {__version__}')

    return p


def main(args=None):
    """Main program function."""
    parser = _get_parser()
    opts = parser.parse_args(args)
    from_img = load(opts.infile)

    if not opts.force and Path(opts.outfile).exists():
        raise FileExistsError(f'Output file exists: {opts.outfile}')

    out_img = conform(
        from_img=from_img,
        out_shape=opts.out_shape,
        voxel_size=opts.voxel_size,
        order=3,
        cval=0.0,
        orientation=opts.orientation,
    )

    save(out_img, opts.outfile)
