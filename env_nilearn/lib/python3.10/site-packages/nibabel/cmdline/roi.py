import argparse
import os
import sys

import nibabel as nb


def lossless_slice(img, slicers):
    if not nb.imageclasses.spatial_axes_first(img):
        raise ValueError('Cannot slice an image that is not known to have spatial axes first')

    scaling = hasattr(img.header, 'set_slope_inter')

    data = img.dataobj._get_unscaled(slicers) if scaling else img.dataobj[slicers]
    roi_img = img.__class__(data, affine=img.slicer.slice_affine(slicers), header=img.header)

    if scaling:
        roi_img.header.set_slope_inter(img.dataobj.slope, img.dataobj.inter)
    return roi_img


def parse_slice(crop, allow_step=True):
    if crop is None:
        return slice(None)
    start, stop, *extra = (int(val) if val else None for val in crop.split(':'))
    if len(extra) > 1:
        raise ValueError(f'Cannot parse specification: {crop}')
    if not allow_step and extra and extra[0] not in (1, None):
        raise ValueError(f'Step entry not permitted: {crop}')

    step = extra[0] if extra else None
    if step not in (1, -1, None):
        raise ValueError(f'Downsampling is not supported: {crop}')

    return slice(start, stop, step)


def sanitize(args):
    # Argparse likes to treat "-1:..." as a flag
    return [f' {arg}' if arg[0] == '-' and ':' in arg else arg for arg in args]


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        description='Crop images to a region of interest',
        epilog='If a start or stop value is omitted, the start or end of the axis is assumed.',
    )
    parser.add_argument('--version', action='version', version=nb.__version__)
    parser.add_argument(
        '-i', metavar='I1:I2[:-1]', help='Start/stop [flip] along first axis (0-indexed)'
    )
    parser.add_argument(
        '-j', metavar='J1:J2[:-1]', help='Start/stop [flip] along second axis (0-indexed)'
    )
    parser.add_argument(
        '-k', metavar='K1:K2[:-1]', help='Start/stop [flip] along third axis (0-indexed)'
    )
    parser.add_argument('-t', metavar='T1:T2', help='Start/stop along fourth axis (0-indexed)')
    parser.add_argument('in_file', help='Image file to crop')
    parser.add_argument('out_file', help='Output file name')

    opts = parser.parse_args(args=sanitize(args))

    try:
        islice = parse_slice(opts.i)
        jslice = parse_slice(opts.j)
        kslice = parse_slice(opts.k)
        tslice = parse_slice(opts.t, allow_step=False)
    except ValueError as err:
        print(f'Could not parse input arguments. Reason follows.\n{err}')
        return 1

    kwargs = {}
    if os.path.realpath(opts.in_file) == os.path.realpath(opts.out_file):
        kwargs['mmap'] = False
    img = nb.load(opts.in_file, **kwargs)

    slicers = (islice, jslice, kslice, tslice)[: img.ndim]
    expected_shape = nb.fileslice.predict_shape(slicers, img.shape)
    if any(dim == 0 for dim in expected_shape):
        print(f'Cannot take zero-length slices. Predicted shape {expected_shape}.')
        return 1

    try:
        sliced_img = lossless_slice(img, slicers)
    except Exception:
        print('Could not slice image. Full traceback follows.')
        raise
    nb.save(sliced_img, opts.out_file)
    return 0
