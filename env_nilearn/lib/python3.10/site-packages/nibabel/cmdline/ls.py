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
Output a summary table for neuroimaging files (resolution, dimensionality, etc.)
"""

import sys
from optparse import Option, OptionParser

import numpy as np

import nibabel as nib
import nibabel.cmdline.utils
from nibabel.cmdline.utils import _err, ap, safe_get, table2string, verbose

__copyright__ = 'Copyright (c) 2011-18 Yaroslav Halchenko and NiBabel contributors'
__license__ = 'MIT'


MAX_UNIQUE = 1000  # maximal number of unique values to report for --counts


def get_opt_parser():
    # use module docstring for help output
    p = OptionParser(
        usage=f'{sys.argv[0]} [OPTIONS] [FILE ...]\n\n' + __doc__,
        version='%prog ' + nib.__version__,
    )

    p.add_options(
        [
            Option(
                '-v',
                '--verbose',
                action='count',
                dest='verbose',
                default=0,
                help='Make more noise.  Could be specified multiple times',
            ),
            Option(
                '-H',
                '--header-fields',
                dest='header_fields',
                default='',
                help='Header fields (comma separated) to be printed as well (if present)',
            ),
            Option(
                '-s',
                '--stats',
                action='store_true',
                dest='stats',
                default=False,
                help='Output basic data statistics',
            ),
            Option(
                '-c',
                '--counts',
                action='store_true',
                dest='counts',
                default=False,
                help='Output counts - number of entries for each numeric value '
                '(useful for int ROI maps)',
            ),
            Option(
                '--all-counts',
                action='store_true',
                dest='all_counts',
                default=False,
                help=f'Output all counts, even if number of unique values > {MAX_UNIQUE}',
            ),
            Option(
                '-z',
                '--zeros',
                action='store_true',
                dest='stats_zeros',
                default=False,
                help='Include zeros into output basic data statistics (--stats, --counts)',
            ),
        ]
    )

    return p


def proc_file(f, opts):
    verbose(1, f'Loading {f}')

    row = [f'@l{f}']
    try:
        vol = nib.load(f)
        h = vol.header
    except Exception as e:
        row += ['failed']
        verbose(2, f'Failed to gather information -- {e}')
        return row

    row += [
        str(safe_get(h, 'data_dtype')),
        f"@l[{ap(safe_get(h, 'data_shape'), '%3g')}]",
        f"@l{ap(safe_get(h, 'zooms'), '%.2f', 'x')}",
    ]
    # Slope
    if (
        hasattr(h, 'has_data_slope')
        and (h.has_data_slope or h.has_data_intercept)
        and not h.get_slope_inter() in ((1.0, 0.0), (None, None))
    ):
        row += ['@l*{:.3g}+{:.3g}'.format(*h.get_slope_inter())]
    else:
        row += ['']

    if hasattr(h, 'extensions') and len(h.extensions):
        row += [f'@l#exts: {len(h.extensions)}']
    else:
        row += ['']

    if opts.header_fields:
        # signals "all fields"
        if opts.header_fields == 'all':
            # TODO: might vary across file types, thus prior sensing
            # would be needed
            header_fields = h.keys()
        else:
            header_fields = opts.header_fields.split(',')

        for f in header_fields:
            if not f:  # skip empty
                continue
            try:
                row += [str(h[f])]
            except (KeyError, ValueError):
                row += [_err()]

    try:
        if (
            hasattr(h, 'get_qform')
            and hasattr(h, 'get_sform')
            and (h.get_qform() != h.get_sform()).any()
        ):
            row += ['sform']
        else:
            row += ['']
    except Exception as e:
        verbose(2, f'Failed to obtain qform or sform -- {e}')
        if isinstance(h, nib.AnalyzeHeader):
            row += ['']
        else:
            row += [_err()]

    if opts.stats or opts.counts:
        # We are doomed to load data
        try:
            d = np.asarray(vol.dataobj)
            if not opts.stats_zeros:
                d = d[np.nonzero(d)]
            else:
                # at least flatten it -- functionality below doesn't
                # depend on the original shape, so let's use a flat view
                d = d.reshape(-1)
            if opts.stats:
                # just # of elements
                row += [f'@l[{np.prod(d.shape)}]']
                # stats
                row += [f'@l[{np.min(d):.2g}, {np.max(d):.2g}]' if len(d) else '-']
            if opts.counts:
                items, inv = np.unique(d, return_inverse=True)
                if len(items) > 1000 and not opts.all_counts:
                    counts = _err(f'{len(items)} uniques. Use --all-counts')
                else:
                    freq = np.bincount(inv)
                    counts = ' '.join(f'{i:g}:{f}' for i, f in zip(items, freq))
                row += ['@l' + counts]
        except OSError as e:
            verbose(2, f'Failed to obtain stats/counts -- {e}')
            row += [_err()]
    return row


def main(args=None):
    """Show must go on"""

    parser = get_opt_parser()
    (opts, files) = parser.parse_args(args=args)

    nibabel.cmdline.utils.verbose_level = opts.verbose

    if nibabel.cmdline.utils.verbose_level < 3:
        # suppress nibabel format-compliance warnings
        nib.imageglobals.logger.level = 50

    rows = [proc_file(f, opts) for f in files]

    print(table2string(rows))
