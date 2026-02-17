"""
Convert tractograms (TCK -> TRK).
"""

import argparse
import os

import nibabel as nib
from nibabel.orientations import aff2axcodes
from nibabel.streamlines import Field


def parse_args():
    DESCRIPTION = 'Convert tractograms (TCK -> TRK).'
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument('anatomy', help='reference anatomical image (.nii|.nii.gz.')
    parser.add_argument(
        'tractograms', metavar='tractogram', nargs='+', help='list of tractograms (.tck).'
    )
    parser.add_argument(
        '-f', '--force', action='store_true', help='overwrite existing output files.'
    )

    args = parser.parse_args()
    return args, parser


def main():
    args, parser = parse_args()

    try:
        nii = nib.load(args.anatomy)
    except Exception:
        parser.error('Expecting anatomical image as first argument.')

    for tractogram in args.tractograms:
        tractogram_format = nib.streamlines.detect_format(tractogram)
        if tractogram_format is not nib.streamlines.TckFile:
            print(f"Skipping non TCK file: '{tractogram}'")
            continue

        filename, _ = os.path.splitext(tractogram)
        output_filename = filename + '.trk'
        if os.path.isfile(output_filename) and not args.force:
            print(f"Skipping existing file: '{output_filename}'. Use -f to overwrite.")
            continue

        # Build header using infos from the anatomical image.
        header = {}
        header[Field.VOXEL_TO_RASMM] = nii.affine.copy()
        header[Field.VOXEL_SIZES] = nii.header.get_zooms()[:3]
        header[Field.DIMENSIONS] = nii.shape[:3]
        header[Field.VOXEL_ORDER] = ''.join(aff2axcodes(nii.affine))

        tck = nib.streamlines.load(tractogram)
        nib.streamlines.save(tck.tractogram, output_filename, header=header)
