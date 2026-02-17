"""Code for PAR/REC to NIfTI converter command"""

import csv
import os
import sys
from optparse import Option, OptionParser

import numpy as np
import numpy.linalg as npl

import nibabel
import nibabel.nifti1 as nifti1
import nibabel.parrec as pr
from nibabel.affines import apply_affine, from_matvec, to_matvec
from nibabel.filename_parser import splitext_addext
from nibabel.mriutils import MRIError, calculate_dwell_time
from nibabel.orientations import apply_orientation, inv_ornt_aff, io_orientation
from nibabel.parrec import one_line
from nibabel.volumeutils import fname_ext_ul_case


def get_opt_parser():
    # use module docstring for help output
    p = OptionParser(
        usage=f'{sys.argv[0]} [OPTIONS] <PAR files>\n\n' + __doc__,
        version='%prog ' + nibabel.__version__,
    )
    p.add_option(
        Option(
            '-v',
            '--verbose',
            action='store_true',
            dest='verbose',
            default=False,
            help="""Make some noise.""",
        )
    )
    p.add_option(
        Option(
            '-o',
            '--output-dir',
            action='store',
            type='string',
            dest='outdir',
            default=None,
            help='Destination directory for NIfTI files. Default: current directory.',
        )
    )
    p.add_option(
        Option(
            '-c',
            '--compressed',
            action='store_true',
            dest='compressed',
            default=False,
            help='Whether to write compressed NIfTI files or not.',
        )
    )
    p.add_option(
        Option(
            '-p',
            '--permit-truncated',
            action='store_true',
            dest='permit_truncated',
            default=False,
            help=one_line(
                """Permit conversion of truncated recordings. Support for
                   this is experimental, and results *must* be checked
                   afterward for validity."""
            ),
        )
    )
    p.add_option(
        Option(
            '-b',
            '--bvs',
            action='store_true',
            dest='bvs',
            default=False,
            help='Output bvals/bvecs files in addition to NIFTI image.',
        )
    )
    p.add_option(
        Option(
            '-d',
            '--dwell-time',
            action='store_true',
            default=False,
            dest='dwell_time',
            help=one_line(
                """Calculate the scan dwell time. If supplied, the magnetic
                   field strength should also be supplied using
                   --field-strength (default 3). The field strength must be
                   supplied because it is not encoded in the PAR/REC
                   format."""
            ),
        )
    )
    p.add_option(
        Option(
            '--field-strength',
            action='store',
            type='float',
            dest='field_strength',
            help=one_line(
                """The magnetic field strength of the recording, only needed
                   for --dwell-time. The field strength must be supplied
                   because it is not encoded in the PAR/REC format."""
            ),
        )
    )
    p.add_option(
        Option(
            '-i',
            '--volume-info',
            action='store_true',
            dest='vol_info',
            default=False,
            help=one_line(
                """Export .PAR volume labels corresponding to the fourth
                   dimension of the data.  The dimension info will be stored in
                   CSV format with the first row containing dimension labels
                   and the subsequent rows (one per volume), the corresponding
                   indices.  Only labels that vary along the 4th dimension are
                   exported (e.g. for a single volume structural scan there
                   are no dynamic labels and no output file will be created).
                   """
            ),
        )
    )
    p.add_option(
        Option(
            '--origin',
            action='store',
            dest='origin',
            default='scanner',
            help=one_line(
                """Reference point of the q-form transformation of the NIfTI
                   image. If 'scanner' the (0,0,0) coordinates will refer to
                   the scanner's iso center. If 'fov', this coordinate will be
                   the center of the recorded volume (field of view). Default:
                   'scanner'."""
            ),
        )
    )
    p.add_option(
        Option(
            '--minmax',
            action='store',
            nargs=2,
            dest='minmax',
            help=one_line(
                """Minimum and maximum settings to be stored in the NIfTI
                   header. If any of them is set to 'parse', the scaled data is
                   scanned for the actual minimum and maximum.  To bypass this
                   potentially slow and memory intensive step (the data has to
                   be scaled and fully loaded into memory), fixed values can be
                   provided as space-separated pair, e.g. '5.4 120.4'. It is
                   possible to set a fixed minimum as scan for the actual
                   maximum (and vice versa). Default: 'parse parse'."""
            ),
        )
    )
    p.set_defaults(minmax=('parse', 'parse'))
    p.add_option(
        Option(
            '--store-header',
            action='store_true',
            dest='store_header',
            default=False,
            help=one_line(
                """If set, all information from the PAR header is stored in
                   an extension of the NIfTI file header.  Default: off"""
            ),
        )
    )
    p.add_option(
        Option(
            '--scaling',
            action='store',
            dest='scaling',
            default='dv',
            help=one_line(
                """Choose data scaling setting. The PAR header defines two
                   different data scaling settings: 'dv' (values displayed on
                   console) and 'fp' (floating point values). Either one can be
                   chosen, or scaling can be disabled completely ('off').  Note
                   that neither method will actually scale the data, but just
                   store the corresponding settings in the NIfTI header, unless
                   non-uniform scaling is used, in which case the data is
                   stored in the file in scaled form. Default: 'dv'"""
            ),
        )
    )
    p.add_option(
        Option(
            '--keep-trace',
            action='store_true',
            dest='keep_trace',
            default=False,
            help=one_line(
                """Do not discard the diagnostic Philips DTI
                   trace volume, if it exists in the data."""
            ),
        )
    )
    p.add_option(
        Option(
            '--overwrite',
            action='store_true',
            dest='overwrite',
            default=False,
            help='Overwrite file if it exists. Default: False',
        )
    )
    p.add_option(
        Option(
            '--strict-sort',
            action='store_true',
            dest='strict_sort',
            default=False,
            help=one_line(
                """Use additional keys in determining the order
                to sort the slices within the .REC file.  This may be necessary
                for more complicated scans with multiple echos,
                cardiac phases, ASL label states, etc."""
            ),
        )
    )
    return p


def verbose(msg, indent=0):
    if verbose.switch:
        print(' ' * indent + msg)


def error(msg, exit_code):
    sys.stderr.write(msg + '\n')
    sys.exit(exit_code)


def proc_file(infile, opts):
    # figure out the output filename, and see if it exists
    basefilename = splitext_addext(os.path.basename(infile))[0]
    if opts.outdir is not None:
        # set output path
        basefilename = os.path.join(opts.outdir, basefilename)

    # prep a file
    if opts.compressed:
        verbose('Using gzip compression')
        outfilename = basefilename + '.nii.gz'
    else:
        outfilename = basefilename + '.nii'
    if os.path.isfile(outfilename) and not opts.overwrite:
        raise OSError(f'Output file "{outfilename}" exists, use --overwrite to overwrite it')

    # load the PAR header and data
    scaling = 'dv' if opts.scaling == 'off' else opts.scaling
    infile = fname_ext_ul_case(infile)
    pr_img = pr.load(
        infile,
        permit_truncated=opts.permit_truncated,
        scaling=scaling,
        strict_sort=opts.strict_sort,
    )
    pr_hdr = pr_img.header
    affine = pr_hdr.get_affine(origin=opts.origin)
    slope, intercept = pr_hdr.get_data_scaling(scaling)
    if opts.scaling != 'off':
        verbose(f'Using data scaling "{opts.scaling}"')
    # get original scaling, and decide if we scale in-place or not
    if opts.scaling == 'off':
        slope = np.array([1.0])
        intercept = np.array([0.0])
        in_data = pr_img.dataobj.get_unscaled()
        out_dtype = pr_hdr.get_data_dtype()
    elif not np.any(np.diff(slope)) and not np.any(np.diff(intercept)):
        # Single scalefactor case
        slope = slope.ravel()[0]
        intercept = intercept.ravel()[0]
        in_data = pr_img.dataobj.get_unscaled()
        out_dtype = pr_hdr.get_data_dtype()
    else:
        # Multi scalefactor case
        slope = np.array([1.0])
        intercept = np.array([0.0])
        in_data = np.array(pr_img.dataobj)
        out_dtype = np.float64
    # Reorient data block to LAS+ if necessary
    ornt = io_orientation(np.diag([-1, 1, 1, 1]).dot(affine))
    if np.array_equal(
        ornt,
        [
            [0, 1],
            [1, 1],
            [2, 1],
        ],
    ):  # already in LAS+
        t_aff = np.eye(4)
    else:  # Not in LAS+
        t_aff = inv_ornt_aff(ornt, pr_img.shape)
        affine = np.dot(affine, t_aff)
        in_data = apply_orientation(in_data, ornt)

    bvals, bvecs = pr_hdr.get_bvals_bvecs()
    if not opts.keep_trace:  # discard Philips DTI trace if present
        if bvecs is not None:
            bad_mask = np.logical_and(bvals != 0, (bvecs == 0).all(axis=1))
            if bad_mask.sum() > 0:
                pl = 's' if bad_mask.sum() != 1 else ''
                verbose(f'Removing {bad_mask.sum()} DTI trace volume{pl}')
                good_mask = ~bad_mask
                in_data = in_data[..., good_mask]
                bvals = bvals[good_mask]
                bvecs = bvecs[good_mask]

    # Make corresponding NIfTI image
    nimg = nifti1.Nifti1Image(in_data, affine, pr_hdr)
    nhdr = nimg.header
    nhdr.set_data_dtype(out_dtype)
    nhdr.set_slope_inter(slope, intercept)
    nhdr.set_sform(affine, code=1)
    nhdr.set_qform(affine, code=1)

    if 'parse' in opts.minmax:
        # need to get the scaled data
        verbose('Loading (and scaling) the data to determine value range')
    if opts.minmax[0] == 'parse':
        nhdr['cal_min'] = in_data.min() * slope + intercept
    else:
        nhdr['cal_min'] = float(opts.minmax[0])
    if opts.minmax[1] == 'parse':
        nhdr['cal_max'] = in_data.max() * slope + intercept
    else:
        nhdr['cal_max'] = float(opts.minmax[1])

    # container for potential NIfTI1 header extensions
    if opts.store_header:
        # dump the full PAR header content into an extension
        with open(infile, 'rb') as fobj:  # contents must be bytes
            hdr_dump = fobj.read()
            dump_ext = nifti1.Nifti1Extension('comment', hdr_dump)
        nhdr.extensions.append(dump_ext)

    verbose(f'Writing {outfilename}')
    nibabel.save(nimg, outfilename)

    # write out bvals/bvecs if requested
    if opts.bvs:
        if bvals is None and bvecs is None:
            verbose('No DTI volumes detected, bvals and bvecs not written')
        elif bvecs is None:
            verbose(
                'DTI volumes detected, but no diffusion direction info was'
                'found.  Writing .bvals file only.'
            )
            with open(basefilename + '.bvals', 'w') as fid:
                # np.savetxt could do this, but it's just a loop anyway
                for val in bvals:
                    fid.write(f'{val} ')
                fid.write('\n')
        else:
            verbose('Writing .bvals and .bvecs files')
            # Transform bvecs with reorientation affine
            orig2new = npl.inv(t_aff)
            bv_reorient = from_matvec(to_matvec(orig2new)[0], [0, 0, 0])
            bvecs = apply_affine(bv_reorient, bvecs)
            with open(basefilename + '.bvals', 'w') as fid:
                # np.savetxt could do this, but it's just a loop anyway
                for val in bvals:
                    fid.write(f'{val} ')
                fid.write('\n')
            with open(basefilename + '.bvecs', 'w') as fid:
                for row in bvecs.T:
                    for val in row:
                        fid.write(f'{val} ')
                    fid.write('\n')

    # export data labels varying along the 4th dimensions if requested
    if opts.vol_info:
        labels = pr_img.header.get_volume_labels()
        if len(labels) > 0:
            vol_keys = list(labels.keys())
            with open(basefilename + '.ordering.csv', 'w', newline='') as csvfile:
                csvwriter = csv.writer(csvfile, delimiter=',')
                csvwriter.writerow(vol_keys)
                for vals in zip(*[labels[k] for k in vol_keys]):
                    csvwriter.writerow(vals)

    # write out dwell time if requested
    if opts.dwell_time:
        try:
            dwell_time = calculate_dwell_time(
                pr_hdr.get_water_fat_shift(), pr_hdr.get_echo_train_length(), opts.field_strength
            )
        except MRIError:
            verbose('No EPI factors, dwell time not written')
        else:
            verbose(
                f'Writing dwell time ({dwell_time!r} sec) '
                f'calculated assuming {opts.field_strength}T magnet'
            )
            with open(basefilename + '.dwell_time', 'w') as fid:
                fid.write(f'{dwell_time!r}\n')
    # done


def main():
    parser = get_opt_parser()
    (opts, infiles) = parser.parse_args()

    verbose.switch = opts.verbose

    if opts.origin not in ('scanner', 'fov'):
        error(f"Unrecognized value for --origin: '{opts.origin}'.", 1)
    if opts.dwell_time and opts.field_strength is None:
        error('Need --field-strength for dwell time calculation', 1)

    # store any exceptions
    errs = []
    for infile in infiles:
        verbose(f'Processing {infile}')
        try:
            proc_file(infile, opts)
        except Exception as e:
            errs.append(f'{infile}: {e}')

    if len(errs):
        error(f'Caught {len(errs)} exceptions. Dump follows:\n\n' + '\n'.join(errs), 1)
    else:
        verbose('Done')
