#!python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""Print nifti diagnostics for header files"""

from argparse import ArgumentParser

import nibabel as nib

__author__ = 'Matthew Brett'
__copyright__ = 'Copyright (c) 2011-18 Matthew Brett and NiBabel contributors'
__license__ = 'MIT'


def main(args=None):
    """Go go team"""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=f'%(prog)s {nib.__version__}')
    parser.add_argument(
        '-1',
        '--nifti1',
        dest='header_class',
        action='store_const',
        const=nib.Nifti1Header,
        default=nib.Nifti1Header,
    )
    parser.add_argument(
        '-2', '--nifti2', dest='header_class', action='store_const', const=nib.Nifti2Header
    )
    parser.add_argument('files', nargs='*', metavar='FILE', help='Nifti file names')

    args = parser.parse_args(args=args)

    for fname in args.files:
        with nib.openers.ImageOpener(fname) as fobj:
            hdr = fobj.read(args.header_class.template_dtype.itemsize)
        result = args.header_class.diagnose_binaryblock(hdr)
        if len(result):
            print(f'Picky header check output for "{fname}"\n')
            print(result + '\n')
        else:
            print(f'Header for "{fname}" is clean')
