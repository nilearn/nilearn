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
Convert neuroimaging file to new parameters
"""

import argparse
import warnings
from pathlib import Path

import nibabel as nib


def _get_parser():
    """Return command-line argument parser."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('infile', help='Neuroimaging volume to convert')
    p.add_argument('outfile', help='Name of output file')
    p.add_argument(
        '--out-dtype', action='store', help='On-disk data type; valid argument to numpy.dtype()'
    )
    p.add_argument(
        '--image-type',
        action='store',
        help='Name of NiBabel image class to create, e.g. Nifti1Image. '
        'If specified, will be used prior to setting dtype. If unspecified, '
        'a new image like `infile` will be created and converted to a type '
        'matching the extension of `outfile`.',
    )
    p.add_argument(
        '-f',
        '--force',
        action='store_true',
        help='Overwrite output file if it exists, and ignore warnings if possible',
    )
    p.add_argument('-V', '--version', action='version', version=f'{p.prog} {nib.__version__}')

    return p


def main(args=None):
    """Main program function."""
    parser = _get_parser()
    opts = parser.parse_args(args)
    orig = nib.load(opts.infile)

    if not opts.force and Path(opts.outfile).exists():
        raise FileExistsError(f'Output file exists: {opts.outfile}')

    if opts.image_type:
        klass = getattr(nib, opts.image_type)
    else:
        klass = orig.__class__

    out_img = klass.from_image(orig)
    if opts.out_dtype:
        try:
            out_img.set_data_dtype(opts.out_dtype)
        except Exception as e:
            if opts.force:
                warnings.warn(f'Ignoring error: {e!r}')
            else:
                raise

    nib.save(out_img, opts.outfile)
