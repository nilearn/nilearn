"""
Convert tractograms (TRK -> TCK).
"""

import argparse
import os

import nibabel as nib


def parse_args():
    DESCRIPTION = 'Convert tractograms (TRK -> TCK).'
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument(
        'tractograms', metavar='tractogram', nargs='+', help='list of tractograms (.trk).'
    )
    parser.add_argument(
        '-f', '--force', action='store_true', help='overwrite existing output files.'
    )

    args = parser.parse_args()
    return args, parser


def main():
    args, parser = parse_args()
    for tractogram in args.tractograms:
        tractogram_format = nib.streamlines.detect_format(tractogram)
        if tractogram_format is not nib.streamlines.TrkFile:
            print(f"Skipping non TRK file: '{tractogram}'")
            continue

        filename, _ = os.path.splitext(tractogram)
        output_filename = filename + '.tck'
        if os.path.isfile(output_filename) and not args.force:
            print(f"Skipping existing file: '{output_filename}'. Use -f to overwrite.")
            continue

        trk = nib.streamlines.load(tractogram)
        nib.streamlines.save(trk.tractogram, output_filename)
