"""
Downloading NeuroImaging datasets: functional datasets (task + resting-state)
"""
import fnmatch
import glob
import warnings
import os
import re
import json

import nibabel as nib
import numpy as np
import numbers

from io import BytesIO

import nibabel
import pandas as pd
from scipy.io import loadmat
try:
    from scipy.io.matlab import MatReadError
except ImportError:  # SciPy < 1.8
    from scipy.io.matlab.miobase import MatReadError
from sklearn.utils import Bunch

from .utils import (_get_dataset_dir, _fetch_files, _get_dataset_descr,
                    _read_md5_sum_file, _tree, _filter_columns, _fetch_file, _uncompress_file)
from .._utils import check_niimg, fill_doc
from .._utils.numpy_conversions import csv_to_array
from nilearn.image import get_data


_LEGACY_FORMAT_MSG = (
    "`legacy_format` will default to `False` in release 0.11. "
    "Dataset fetchers will then return pandas dataframes by default "
    "instead of recarrays."
)


@fill_doc
def fetch_haxby(data_dir=None, subjects=(2,),
                fetch_stimuli=False, url=None, resume=True, verbose=1):
    """Download and loads complete haxby dataset.

    See :footcite:`Haxby2001`.

    Parameters
    ----------
    %(data_dir)s
    subjects : list or int, optional
        Either a list of subjects or the number of subjects to load, from 1 to
        6. By default, 2nd subject will be loaded. Empty list returns no subject
        data. Default=(2,).

    fetch_stimuli : boolean, optional
        Indicate if stimuli images must be downloaded. They will be presented
        as a dictionary of categories. Default=False.
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :

        - 'anat': string list. Paths to anatomic images.
        - 'func': string list. Paths to nifti file with bold data.
        - 'session_target': string list. Paths to text file containing session and target data.
        - 'mask': string. Path to fullbrain mask file.
        - 'mask_vt': string list. Paths to nifti ventral temporal mask file.
        - 'mask_face': string list. Paths to nifti ventral temporal mask file.
        - 'mask_house': string list. Paths to nifti ventral temporal mask file.
        - 'mask_face_little': string list. Paths to nifti ventral temporal mask file.
        - 'mask_house_little': string list. Paths to nifti ventral temporal mask file.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    PyMVPA provides a tutorial making use of this dataset:
    http://www.pymvpa.org/tutorial.html

    More information about its structure:
    http://dev.pymvpa.org/datadb/haxby2001.html

    See `additional information
    <http://www.sciencemag.org/content/293/5539/2425>`

    Run 8 in subject 5 does not contain any task labels.
    The anatomical image for subject 6 is unavailable.

    """
    if isinstance(subjects, numbers.Number) and subjects > 6:
        subjects = 6

    if subjects is not None and isinstance(subjects, (list, tuple)):
        for sub_id in subjects:
            if sub_id not in [1, 2, 3, 4, 5, 6]:
                raise ValueError("You provided invalid subject id {0} in a "
                                 "list. Subjects must be selected in "
                                 "[1, 2, 3, 4, 5, 6]".format(sub_id))

    dataset_name = 'haxby2001'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # Get the mask
    url_mask = 'https://www.nitrc.org/frs/download.php/7868/mask.nii.gz'
    mask = _fetch_files(data_dir, [('mask.nii.gz', url_mask, {})],
                        verbose=verbose)[0]

    # Dataset files
    if url is None:
        url = 'http://data.pymvpa.org/datasets/haxby2001/'
    md5sums = _fetch_files(data_dir, [('MD5SUMS', url + 'MD5SUMS', {})],
                           verbose=verbose)[0]
    md5sums = _read_md5_sum_file(md5sums)

    # definition of dataset files
    sub_files = ['bold.nii.gz', 'labels.txt',
                 'mask4_vt.nii.gz', 'mask8b_face_vt.nii.gz',
                 'mask8b_house_vt.nii.gz', 'mask8_face_vt.nii.gz',
                 'mask8_house_vt.nii.gz', 'anat.nii.gz']
    n_files = len(sub_files)

    if subjects is None:
        subjects = []

    if isinstance(subjects, numbers.Number):
        subject_mask = np.arange(1, subjects + 1)
    else:
        subject_mask = np.array(subjects)

    files = [
            (os.path.join('subj%d' % i, sub_file),
             url + 'subj%d-2010.01.14.tar.gz' % i,
             {'uncompress': True,
              'md5sum': md5sums.get('subj%d-2010.01.14.tar.gz' % i, None)})
            for i in subject_mask
            for sub_file in sub_files
            if not (sub_file == 'anat.nii.gz' and i == 6)  # no anat for sub. 6
    ]

    files = _fetch_files(data_dir, files, resume=resume, verbose=verbose)

    if ((isinstance(subjects, numbers.Number) and subjects == 6) or
            np.any(subject_mask == 6)):
        files.append(None)  # None value because subject 6 has no anat

    kwargs = {}
    if fetch_stimuli:
        stimuli_files = [(os.path.join('stimuli', 'README'),
                          url + 'stimuli-2010.01.14.tar.gz',
                          {'uncompress': True})]
        readme = _fetch_files(data_dir, stimuli_files, resume=resume,
                              verbose=verbose)[0]
        kwargs['stimuli'] = _tree(os.path.dirname(readme), pattern='*.jpg',
                                  dictionary=True)

    fdescr = _get_dataset_descr(dataset_name)

    # return the data
    return Bunch(
            anat=files[7::n_files],
            func=files[0::n_files],
            session_target=files[1::n_files],
            mask_vt=files[2::n_files],
            mask_face=files[3::n_files],
            mask_house=files[4::n_files],
            mask_face_little=files[5::n_files],
            mask_house_little=files[6::n_files],
            mask=mask,
            description=fdescr,
            **kwargs)


@fill_doc
def fetch_adhd(n_subjects=30, data_dir=None, url=None, resume=True,
               verbose=1):
    """Download and load the ADHD resting-state dataset.

    See :footcite:`ADHDdataset`.

    Parameters
    ----------
    n_subjects : int, optional
        The number of subjects to load from maximum of 40 subjects.
        By default, 30 subjects will be loaded. If None is given,
        all 40 subjects will be loaded. Default=30.
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :

         - 'func': Paths to functional resting-state images
         - 'phenotypic': Explanations of preprocessing steps
         - 'confounds': CSV files containing the nuisance variables

    References
    ----------
    .. footbibliography::

    """
    if url is None:
        url = 'https://www.nitrc.org/frs/download.php/'

    # Preliminary checks and declarations
    dataset_name = 'adhd'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    ids = ['0010042', '0010064', '0010128', '0021019', '0023008', '0023012',
           '0027011', '0027018', '0027034', '0027037', '1019436', '1206380',
           '1418396', '1517058', '1552181', '1562298', '1679142', '2014113',
           '2497695', '2950754', '3007585', '3154996', '3205761', '3520880',
           '3624598', '3699991', '3884955', '3902469', '3994098', '4016887',
           '4046678', '4134561', '4164316', '4275075', '6115230', '7774305',
           '8409791', '8697774', '9744150', '9750701']
    nitrc_ids = range(7782, 7822)
    max_subjects = len(ids)
    if n_subjects is None:
        n_subjects = max_subjects
    if n_subjects > max_subjects:
        warnings.warn('Warning: there are only %d subjects' % max_subjects)
        n_subjects = max_subjects
    ids = ids[:n_subjects]
    nitrc_ids = nitrc_ids[:n_subjects]

    opts = dict(uncompress=True)

    # Dataset description
    fdescr = _get_dataset_descr(dataset_name)

    # First, get the metadata
    phenotypic = ('ADHD200_40subs_motion_parameters_and_phenotypics.csv',
        url + '7781/adhd40_metadata.tgz', opts)

    phenotypic = _fetch_files(data_dir, [phenotypic], resume=resume,
                              verbose=verbose)[0]

    # Load the csv file
    phenotypic = np.genfromtxt(phenotypic, names=True, delimiter=',',
                               dtype=None, encoding=None)

    # Keep phenotypic information for selected subjects
    int_ids = np.asarray(ids, dtype=int)
    phenotypic = phenotypic[[np.where(phenotypic['Subject'] == i)[0][0]
                             for i in int_ids]]

    # Download dataset files

    archives = [url + '%i/adhd40_%s.tgz' % (ni, ii)
                for ni, ii in zip(nitrc_ids, ids)]
    functionals = ['data/%s/%s_rest_tshift_RPI_voreg_mni.nii.gz' % (i, i)
                   for i in ids]
    confounds = ['data/%s/%s_regressors.csv' % (i, i) for i in ids]

    functionals = _fetch_files(
        data_dir, zip(functionals, archives, (opts,) * n_subjects),
        resume=resume, verbose=verbose)

    confounds = _fetch_files(
        data_dir, zip(confounds, archives, (opts,) * n_subjects),
        resume=resume, verbose=verbose)

    return Bunch(func=functionals, confounds=confounds,
                 phenotypic=phenotypic, description=fdescr)


@fill_doc
def fetch_miyawaki2008(data_dir=None, url=None, resume=True, verbose=1):
    """Download and loads Miyawaki et al. 2008 dataset (153MB).

    See :footcite:`Miyawaki2008`.

    Parameters
    ----------
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :

        - 'func': string list
            Paths to nifti file with bold data
        - 'label': string list
            Paths to text file containing session and target data
        - 'mask': string
            Path to nifti mask file to define target volume in visual
            cortex
        - 'background': string
            Path to nifti file containing a background image usable as a
            background image for miyawaki images.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    This dataset is available on the `brainliner website
    <http://brainliner.jp/data/brainliner-admin/Reconstruct>`_

    See `additional information
    <http://www.cns.atr.jp/dni/en/downloads/
    fmri-data-set-for-visual-image-reconstruction/>`_

    """
    url = 'https://www.nitrc.org/frs/download.php' \
          '/8486/miyawaki2008.tgz?i_agree=1&download_now=1'
    opts = {'uncompress': True}

    # Dataset files

    # Functional MRI:
    #   * 20 random scans (usually used for training)
    #   * 12 figure scans (usually used for testing)

    func_figure = [(os.path.join('func', 'data_figure_run%02d.nii.gz' % i),
                    url, opts) for i in range(1, 13)]

    func_random = [(os.path.join('func', 'data_random_run%02d.nii.gz' % i),
                    url, opts) for i in range(1, 21)]

    # Labels, 10x10 patches, stimuli shown to the subject:
    #   * 20 random labels
    #   * 12 figure labels (letters and shapes)

    label_filename = 'data_%s_run%02d_label.csv'
    label_figure = [(os.path.join('label', label_filename % ('figure', i)),
                     url, opts) for i in range(1, 13)]

    label_random = [(os.path.join('label', label_filename % ('random', i)),
                     url, opts) for i in range(1, 21)]

    # Masks

    file_mask = [
        'mask.nii.gz',
        'LHlag0to1.nii.gz',
        'LHlag10to11.nii.gz',
        'LHlag1to2.nii.gz',
        'LHlag2to3.nii.gz',
        'LHlag3to4.nii.gz',
        'LHlag4to5.nii.gz',
        'LHlag5to6.nii.gz',
        'LHlag6to7.nii.gz',
        'LHlag7to8.nii.gz',
        'LHlag8to9.nii.gz',
        'LHlag9to10.nii.gz',
        'LHV1d.nii.gz',
        'LHV1v.nii.gz',
        'LHV2d.nii.gz',
        'LHV2v.nii.gz',
        'LHV3A.nii.gz',
        'LHV3.nii.gz',
        'LHV4v.nii.gz',
        'LHVP.nii.gz',
        'RHlag0to1.nii.gz',
        'RHlag10to11.nii.gz',
        'RHlag1to2.nii.gz',
        'RHlag2to3.nii.gz',
        'RHlag3to4.nii.gz',
        'RHlag4to5.nii.gz',
        'RHlag5to6.nii.gz',
        'RHlag6to7.nii.gz',
        'RHlag7to8.nii.gz',
        'RHlag8to9.nii.gz',
        'RHlag9to10.nii.gz',
        'RHV1d.nii.gz',
        'RHV1v.nii.gz',
        'RHV2d.nii.gz',
        'RHV2v.nii.gz',
        'RHV3A.nii.gz',
        'RHV3.nii.gz',
        'RHV4v.nii.gz',
        'RHVP.nii.gz'
    ]

    file_mask = [(os.path.join('mask', m), url, opts) for m in file_mask]

    file_names = func_figure + func_random + \
                 label_figure + label_random + \
                 file_mask

    dataset_name = 'miyawaki2008'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, file_names, resume=resume, verbose=verbose)

    # Fetch the background image
    bg_img = _fetch_files(data_dir, [('bg.nii.gz', url, opts)], resume=resume,
                          verbose=verbose)[0]

    fdescr = _get_dataset_descr(dataset_name)

    # Return the data
    return Bunch(
        func=files[:32],
        label=files[32:64],
        mask=files[64],
        mask_roi=files[65:],
        background=bg_img,
        description=fdescr)


@fill_doc
def fetch_localizer_contrasts(contrasts, n_subjects=None, get_tmaps=False,
                              get_masks=False, get_anats=False,
                              data_dir=None, url=None, resume=True, verbose=1,
                              legacy_format=True):
    """Download and load Brainomics/Localizer dataset (94 subjects).

    "The Functional Localizer is a simple and fast acquisition
    procedure based on a 5-minute functional magnetic resonance
    imaging (fMRI) sequence that can be run as easily and as
    systematically as an anatomical scan. This protocol captures the
    cerebral bases of auditory and visual perception, motor actions,
    reading, language comprehension and mental calculation at an
    individual level. Individual functional maps are reliable and
    quite precise. The procedure is described in more detail on the
    Functional Localizer page."
    (see https://osf.io/vhtf6/)

    You may cite :footcite:`Papadopoulos-Orfanos2017`
    when using this dataset.

    Scientific results obtained using this dataset are described
    in :footcite:`Pinel2007`.

    Parameters
    ----------
    contrasts : list of str
        The contrasts to be fetched (for all 94 subjects available).
        Allowed values are::

        - "checkerboard"
        - "horizontal checkerboard"
        - "vertical checkerboard"
        - "horizontal vs vertical checkerboard"
        - "vertical vs horizontal checkerboard"
        - "sentence listening"
        - "sentence reading"
        - "sentence listening and reading"
        - "sentence reading vs checkerboard"
        - "calculation (auditory cue)"
        - "calculation (visual cue)"
        - "calculation (auditory and visual cue)"
        - "calculation (auditory cue) vs sentence listening"
        - "calculation (visual cue) vs sentence reading"
        - "calculation vs sentences"
        - "calculation (auditory cue) and sentence listening"
        - "calculation (visual cue) and sentence reading"
        - "calculation and sentence listening/reading"
        - "calculation (auditory cue) and sentence listening vs "
        - "calculation (visual cue) and sentence reading"
        - "calculation (visual cue) and sentence reading vs checkerboard"
        - "calculation and sentence listening/reading vs button press"
        - "left button press (auditory cue)"
        - "left button press (visual cue)"
        - "left button press"
        - "left vs right button press"
        - "right button press (auditory cue)"
        - "right button press (visual cue)"
        - "right button press"
        - "right vs left button press"
        - "button press (auditory cue) vs sentence listening"
        - "button press (visual cue) vs sentence reading"
        - "button press vs calculation and sentence listening/reading"

        or equivalently on can use the original names::

        - "checkerboard"
        - "horizontal checkerboard"
        - "vertical checkerboard"
        - "horizontal vs vertical checkerboard"
        - "vertical vs horizontal checkerboard"
        - "auditory sentences"
        - "visual sentences"
        - "auditory&visual sentences"
        - "visual sentences vs checkerboard"
        - "auditory calculation"
        - "visual calculation"
        - "auditory&visual calculation"
        - "auditory calculation vs auditory sentences"
        - "visual calculation vs sentences"
        - "auditory&visual calculation vs sentences"
        - "auditory processing"
        - "visual processing"
        - "visual processing vs auditory processing"
        - "auditory processing vs visual processing"
        - "visual processing vs checkerboard"
        - "cognitive processing vs motor"
        - "left auditory click"
        - "left visual click"
        - "left auditory&visual click"
        - "left auditory & visual click vs right auditory&visual click"
        - "right auditory click"
        - "right visual click"
        - "right auditory&visual click"
        - "right auditory & visual click vs left auditory&visual click"
        - "auditory click vs auditory sentences"
        - "visual click vs visual sentences"
        - "auditory&visual motor vs cognitive processing"

    n_subjects : int or list, optional
        The number or list of subjects to load. If None is given,
        all 94 subjects are used.

    get_tmaps : boolean, optional
        Whether t maps should be fetched or not. Default=False.

    get_masks : boolean, optional
        Whether individual masks should be fetched or not.
        Default=False.

    get_anats : boolean, optional
        Whether individual structural images should be fetched or not.
        Default=False.
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s
    %(legacy_format)s

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :

        - 'cmaps': string list
            Paths to nifti contrast maps
        - 'tmaps' string list (if 'get_tmaps' set to True)
            Paths to nifti t maps
        - 'masks': string list
            Paths to nifti files corresponding to the subjects individual masks
        - 'anats': string
            Path to nifti files corresponding to the subjects structural images

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nilearn.datasets.fetch_localizer_calculation_task
    nilearn.datasets.fetch_localizer_button_task

    """
    if isinstance(contrasts, str):
        raise ValueError('Contrasts should be a list of strings, but '
                         'a single string was given: "%s"' % contrasts)
    if n_subjects is None:
        n_subjects = 94  # 94 subjects available
    if (isinstance(n_subjects, numbers.Number) and
            ((n_subjects > 94) or (n_subjects < 1))):
        warnings.warn("Wrong value for \'n_subjects\' (%d). The maximum "
                      "value will be used instead (\'n_subjects=94\')")
        n_subjects = 94  # 94 subjects available

    # we allow the user to use alternatives to Brainomics contrast names
    contrast_name_wrapper = {
        # Checkerboard
        "checkerboard": "checkerboard",
        "horizontal checkerboard": "horizontal checkerboard",
        "vertical checkerboard": "vertical checkerboard",
        "horizontal vs vertical checkerboard":
            "horizontal vs vertical checkerboard",
        "vertical vs horizontal checkerboard":
            "vertical vs horizontal checkerboard",
        # Sentences
        "sentence listening": "auditory sentences",
        "sentence reading": "visual sentences",
        "sentence listening and reading": "auditory&visual sentences",
        "sentence reading vs checkerboard": "visual sentences vs checkerboard",
        # Calculation
        "calculation (auditory cue)": "auditory calculation",
        "calculation (visual cue)": "visual calculation",
        "calculation (auditory and visual cue)": "auditory&visual calculation",
        "calculation (auditory cue) vs sentence listening":
            "auditory calculation vs auditory sentences",
        "calculation (visual cue) vs sentence reading":
            "visual calculation vs sentences",
        "calculation vs sentences": "auditory&visual calculation vs sentences",
        # Calculation + Sentences
        "calculation (auditory cue) and sentence listening":
            "auditory processing",
        "calculation (visual cue) and sentence reading":
            "visual processing",
        "calculation (visual cue) and sentence reading vs "
        "calculation (auditory cue) and sentence listening":
            "visual processing vs auditory processing",
        "calculation (auditory cue) and sentence listening vs "
        "calculation (visual cue) and sentence reading":
            "auditory processing vs visual processing",
        "calculation (visual cue) and sentence reading vs checkerboard":
            "visual processing vs checkerboard",
        "calculation and sentence listening/reading vs button press":
            "cognitive processing vs motor",
        # Button press
        "left button press (auditory cue)": "left auditory click",
        "left button press (visual cue)": "left visual click",
        "left button press": "left auditory&visual click",
        "left vs right button press": "left auditory & visual click vs "
            + "right auditory&visual click",
        "right button press (auditory cue)": "right auditory click",
        "right button press (visual cue)": "right visual click",
        "right button press": "right auditory & visual click",
        "right vs left button press": "right auditory & visual click "
           + "vs left auditory&visual click",
        "button press (auditory cue) vs sentence listening":
            "auditory click vs auditory sentences",
        "button press (visual cue) vs sentence reading":
            "visual click vs visual sentences",
        "button press vs calculation and sentence listening/reading":
            "auditory&visual motor vs cognitive processing"}
    allowed_contrasts = list(contrast_name_wrapper.values())

    # convert contrast names
    contrasts_wrapped = []
    # get a unique ID for each contrast. It is used to give a unique name to
    # each download file and avoid name collisions.
    contrasts_indices = []
    for contrast in contrasts:
        if contrast in allowed_contrasts:
            contrasts_wrapped.append(contrast.title().replace(" ", ""))
            contrasts_indices.append(allowed_contrasts.index(contrast))
        elif contrast in contrast_name_wrapper:
            name = contrast_name_wrapper[contrast]
            contrasts_wrapped.append(name.title().replace(" ", ""))
            contrasts_indices.append(allowed_contrasts.index(name))
        else:
            raise ValueError("Contrast \'%s\' is not available" % contrast)

    # Get the dataset OSF index
    dataset_name = "brainomics_localizer"
    index_url = "https://osf.io/hwbm2/download"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    index_file = _fetch_file(index_url, data_dir, verbose=verbose)
    with open(index_file, "rt") as of:
        index = json.load(of)

    # Build data URLs that will be fetched
    files = {}
    # Download from the relevant OSF project, using hashes generated
    # from the OSF API. Note the trailing slash. For more info, see:
    # https://gist.github.com/emdupre/3cb4d564511d495ea6bf89c6a577da74
    root_url = "https://osf.io/download/{0}/"
    if isinstance(n_subjects, numbers.Number):
        subject_mask = np.arange(1, n_subjects + 1)
    else:
        subject_mask = np.array(n_subjects)
    subject_ids = ["S%02d" % s for s in subject_mask]
    data_types = ["cmaps"]
    if get_tmaps:
        data_types.append("tmaps")
    filenames = []

    def _is_valid_path(path, index, verbose):
        if path not in index:
            if verbose > 0:
                print("Skipping path '{0}'...".format(path))
            return False
        return True

    for subject_id in subject_ids:
        for data_type in data_types:
            for contrast_id, contrast in enumerate(contrasts_wrapped):
                name_aux = str.replace(
                    str.join('_', [data_type, contrast]), ' ', '_')
                file_path = os.path.join(
                    "brainomics_data", subject_id, "%s.nii.gz" % name_aux)
                path = "/".join([
                    "/localizer", "derivatives", "spm_1st_level",
                    "sub-%s" % subject_id,
                    "sub-%s_task-localizer_acq-%s_%s.nii.gz" % (
                        subject_id, contrast, data_type)])
                if _is_valid_path(path, index, verbose=verbose):
                    file_url = root_url.format(index[path][1:])
                    opts = {"move": file_path}
                    filenames.append((file_path, file_url, opts))
                    files.setdefault(data_type, []).append(file_path)

    # Fetch masks if asked by user
    if get_masks:
        for subject_id in subject_ids:
            file_path = os.path.join(
                "brainomics_data", subject_id, "boolean_mask_mask.nii.gz")
            path = "/".join([
                "/localizer", "derivatives", "spm_1st_level",
                "sub-%s" % subject_id, "sub-%s_mask.nii.gz" % subject_id])
            if _is_valid_path(path, index, verbose=verbose):
                file_url = root_url.format(index[path][1:])
                opts = {"move": file_path}
                filenames.append((file_path, file_url, opts))
                files.setdefault("masks", []).append(file_path)

    # Fetch anats if asked by user
    if get_anats:
        for subject_id in subject_ids:
            file_path = os.path.join(
                "brainomics_data", subject_id,
                "normalized_T1_anat_defaced.nii.gz")
            path = "/".join([
                "/localizer", "derivatives", "spm_preprocessing",
                "sub-%s" % subject_id, "sub-%s_T1w.nii.gz" % subject_id])
            if _is_valid_path(path, index, verbose=verbose):
                file_url = root_url.format(index[path][1:])
                opts = {"move": file_path}
                filenames.append((file_path, file_url, opts))
                files.setdefault("anats", []).append(file_path)

    # Fetch subject characteristics
    participants_file = os.path.join("brainomics_data", "participants.tsv")
    path = "/localizer/participants.tsv"
    if _is_valid_path(path, index, verbose=verbose):
        file_url = root_url.format(index[path][1:])
        opts = {"move": participants_file}
        filenames.append((participants_file, file_url, opts))

    # Fetch behavioural
    behavioural_file = os.path.join(
        "brainomics_data", "phenotype", "behavioural.tsv")
    path = "/localizer/phenotype/behavioural.tsv"
    if _is_valid_path(path, index, verbose=verbose):
        file_url = root_url.format(index[path][1:])
        opts = {"move": behavioural_file}
        filenames.append((behavioural_file, file_url, opts))

    # Actual data fetching
    fdescr = _get_dataset_descr(dataset_name)
    _fetch_files(data_dir, filenames, verbose=verbose)
    for key, value in files.items():
        files[key] = [os.path.join(data_dir, val) for val in value]

    # Load covariates file
    from numpy.lib.recfunctions import join_by
    participants_file = os.path.join(data_dir, participants_file)
    csv_data = pd.read_csv(participants_file, delimiter='\t')
    behavioural_file = os.path.join(data_dir, behavioural_file)
    csv_data2 = pd.read_csv(behavioural_file, delimiter='\t')
    csv_data = csv_data.merge(csv_data2)
    subject_names = csv_data["participant_id"].tolist()
    subjects_indices = []
    for name in subject_ids:
        if name not in subject_names:
            continue
        subjects_indices.append(subject_names.index(name))
    csv_data = csv_data.iloc[subjects_indices]
    if legacy_format:
        warnings.warn(_LEGACY_FORMAT_MSG)
        csv_data = csv_data.to_records(index=False)
    return Bunch(ext_vars=csv_data, description=fdescr, **files)


@fill_doc
def fetch_localizer_calculation_task(n_subjects=1, data_dir=None, url=None,
                                     verbose=1, legacy_format=True):
    """Fetch calculation task contrast maps from the localizer.

    Parameters
    ----------
    n_subjects : int, optional
        The number of subjects to load. If None is given,
        all 94 subjects are used. Default=1.
    %(data_dir)s
    %(url)s
    %(verbose)s
    %(legacy_format)s

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'cmaps': string list, giving paths to nifti contrast maps

    Notes
    -----
    This function is only a caller for the fetch_localizer_contrasts in order
    to simplify examples reading and understanding.
    The 'calculation (auditory and visual cue)' contrast is used.

    See Also
    --------
    nilearn.datasets.fetch_localizer_button_task
    nilearn.datasets.fetch_localizer_contrasts

    """
    data = fetch_localizer_contrasts(["calculation (auditory and visual cue)"],
                                     n_subjects=n_subjects,
                                     get_tmaps=False, get_masks=False,
                                     get_anats=False, data_dir=data_dir,
                                     url=url, resume=True, verbose=verbose,
                                     legacy_format=legacy_format)
    return data


@fill_doc
def fetch_localizer_button_task(data_dir=None, url=None,
                                verbose=1, legacy_format=True):
    """Fetch left vs right button press contrast maps from the localizer.

    Parameters
    ----------
    %(data_dir)s
    %(url)s
    %(verbose)s
    %(legacy_format)s

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :

        - 'cmaps': string list, giving paths to nifti contrast maps
        - 'tmap': string, giving paths to nifti contrast maps
        - 'anat': string, giving paths to normalized anatomical image

    Notes
    -----
    This function is only a caller for the fetch_localizer_contrasts in order
    to simplify examples reading and understanding.
    The 'left vs right button press' contrast is used.

    See Also
    --------
    nilearn.datasets.fetch_localizer_calculation_task
    nilearn.datasets.fetch_localizer_contrasts

    """
    data = fetch_localizer_contrasts(["left vs right button press"],
                                     n_subjects=[2],
                                     get_tmaps=True, get_masks=False,
                                     get_anats=True, data_dir=data_dir,
                                     url=url, resume=True, verbose=verbose,
                                     legacy_format=legacy_format)
    # Additional keys for backward compatibility
    data['tmap'] = data['tmaps'][0]
    data['anat'] = data['anats'][0]
    return data


@fill_doc
def fetch_abide_pcp(data_dir=None, n_subjects=None, pipeline='cpac',
                    band_pass_filtering=False, global_signal_regression=False,
                    derivatives=['func_preproc'],
                    quality_checked=True, url=None, verbose=1,
                    legacy_format=True, **kwargs):
    """Fetch ABIDE dataset.

    Fetch the Autism Brain Imaging Data Exchange (ABIDE) dataset wrt criteria
    that can be passed as parameter. Note that this is the preprocessed
    version of ABIDE provided by the preprocess connectome projects (PCP).
    See :footcite:`Nielsen2013`.

    Parameters
    ----------
    %(data_dir)s
    n_subjects : int, optional
        The number of subjects to load. If None is given,
        all available subjects are used (this number depends on the
        preprocessing pipeline used).

    pipeline : string {'cpac', 'css', 'dparsf', 'niak'}, optional
        Possible pipelines are "ccs", "cpac", "dparsf" and "niak".
        Default='cpac'.

    band_pass_filtering : boolean, optional
        Due to controversies in the literature, band pass filtering is
        optional. If true, signal is band filtered between 0.01Hz and 0.1Hz.
        Default=False.

    global_signal_regression : boolean optional
        Indicates if global signal regression should be applied on the
        signals. Default=False.

    derivatives : string list, optional
        Types of downloaded files. Possible values are: alff, degree_binarize,
        degree_weighted, dual_regression, eigenvector_binarize,
        eigenvector_weighted, falff, func_mask, func_mean, func_preproc, lfcd,
        reho, rois_aal, rois_cc200, rois_cc400, rois_dosenbach160, rois_ez,
        rois_ho, rois_tt, and vmhc. Please refer to the PCP site for more
        details. Default=['func_preproc'].

    quality_checked : boolean, optional
        If true (default), restrict the list of the subjects to the one that
        passed quality assessment for all raters. Default=True.
    %(url)s
    %(verbose)s
    %(legacy_format)s
    kwargs : parameter list, optional
        Any extra keyword argument will be used to filter downloaded subjects
        according to the CSV phenotypic file. Some examples of filters are
        indicated below.

    SUB_ID : list of integers in [50001, 50607], optional
        Ids of the subjects to be loaded.

    DX_GROUP : integer in {1, 2}, optional
        1 is autism, 2 is control.

    DSM_IV_TR : integer in [0, 4], optional
        O is control, 1 is autism, 2 is Asperger, 3 is PPD-NOS,
        4 is Asperger or PPD-NOS.

    AGE_AT_SCAN : float in [6.47, 64], optional
        Age of the subject.

    SEX : integer in {1, 2}, optional
        1 is male, 2 is female.

    HANDEDNESS_CATEGORY : string in {'R', 'L', 'Mixed', 'Ambi'}, optional
        R = Right, L = Left, Ambi = Ambidextrous.

    HANDEDNESS_SCORE : integer in [-100, 100], optional
        Positive = Right, Negative = Left, 0 = Ambidextrous.

    Notes
    -----
    Code and description of preprocessing pipelines are provided on the
    `PCP website <http://preprocessed-connectomes-project.github.io/>`.

    References
    ----------
    .. footbibliography::

    """
    # People keep getting it wrong and submitting a string instead of a
    # list of strings. We'll make their life easy
    if isinstance(derivatives, str):
        derivatives = [derivatives, ]

    # Parameter check
    for derivative in derivatives:
        if derivative not in [
                'alff', 'degree_binarize', 'degree_weighted',
                'dual_regression', 'eigenvector_binarize',
                'eigenvector_weighted', 'falff', 'func_mask', 'func_mean',
                'func_preproc', 'lfcd', 'reho', 'rois_aal', 'rois_cc200',
                'rois_cc400', 'rois_dosenbach160', 'rois_ez', 'rois_ho',
                'rois_tt', 'vmhc']:
            raise KeyError('%s is not a valid derivative' % derivative)

    strategy = ''
    if not band_pass_filtering:
        strategy += 'no'
    strategy += 'filt_'
    if not global_signal_regression:
        strategy += 'no'
    strategy += 'global'

    # General file: phenotypic information
    dataset_name = 'ABIDE_pcp'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    if url is None:
        url = ('https://s3.amazonaws.com/fcp-indi/data/Projects/'
               'ABIDE_Initiative')

    if quality_checked:
        kwargs['qc_rater_1'] = 'OK'
        kwargs['qc_anat_rater_2'] = ['OK', 'maybe']
        kwargs['qc_func_rater_2'] = ['OK', 'maybe']
        kwargs['qc_anat_rater_3'] = 'OK'
        kwargs['qc_func_rater_3'] = 'OK'

    # Fetch the phenotypic file and load it
    csv = 'Phenotypic_V1_0b_preprocessed1.csv'
    path_csv = _fetch_files(data_dir, [(csv, url + '/' + csv, {})],
                            verbose=verbose)[0]

    # Note: the phenotypic file contains string that contains comma which mess
    # up numpy array csv loading. This is why I do a pass to remove the last
    # field. This can be
    # done simply with pandas but we don't want such dependency ATM
    # pheno = pandas.read_csv(path_csv).to_records()
    with open(path_csv, 'r') as pheno_f:
        pheno = ['i' + pheno_f.readline()]

        # This regexp replaces commas between double quotes
        for line in pheno_f:
            pheno.append(re.sub(r',(?=[^"]*"(?:[^"]*"[^"]*")*[^"]*$)', ";", line))

    # bytes (encode()) needed for python 2/3 compat with numpy
    pheno = '\n'.join(pheno).encode()
    pheno = BytesIO(pheno)
    pheno = pd.read_csv(pheno, comment='$')

    # First, filter subjects with no filename
    pheno = pheno[pheno['FILE_ID'] != 'no_filename']
    # Apply user defined filters
    user_filter = _filter_columns(pheno, kwargs)
    pheno = pheno[user_filter]

    # Go into specific data folder and url
    data_dir = os.path.join(data_dir, pipeline, strategy)
    url = '/'.join([url, 'Outputs', pipeline, strategy])

    # Get the files
    results = {}
    file_ids = pheno['FILE_ID'].tolist()
    if n_subjects is not None:
        file_ids = file_ids[:n_subjects]
        pheno = pheno[:n_subjects]

    if legacy_format:
        warnings.warn(_LEGACY_FORMAT_MSG)
        pheno = pheno.to_records(index=False)

    results['description'] = _get_dataset_descr(dataset_name)
    results['phenotypic'] = pheno
    for derivative in derivatives:
        ext = '.1D' if derivative.startswith('rois') else '.nii.gz'
        files = []
        for file_id in file_ids:
            file_ = [(
                file_id + '_' + derivative + ext,
                '/'.join([url, derivative, file_id + '_' + derivative + ext]),
                {}
            )]
            files.append(_fetch_files(data_dir, file_, verbose=verbose)[0])
        # Load derivatives if needed
        if ext == '.1D':
            files = [np.loadtxt(f) for f in files]
        results[derivative] = files
    return Bunch(**results)


def _load_mixed_gambles(zmap_imgs):
    """Ravel zmaps (one per subject) along time axis, resulting,
    in a n_subjects * n_trials 3D niimgs and, and then make
    gain vector y of same length.
    """
    X = []
    y = []
    mask = []
    for zmap_img in zmap_imgs:
        # load subject data
        this_X = get_data(zmap_img)
        affine = zmap_img.affine
        finite_mask = np.all(np.isfinite(this_X), axis=-1)
        this_mask = np.logical_and(np.all(this_X != 0, axis=-1),
                                   finite_mask)
        this_y = np.array([np.arange(1, 9)] * 6).ravel()

        # gain levels
        if len(this_y) != this_X.shape[-1]:
            raise RuntimeError("%s: Expecting %i volumes, got %i!" % (
                zmap_img, len(this_y), this_X.shape[-1]))

        # standardize subject data
        this_X -= this_X.mean(axis=-1)[..., np.newaxis]
        std = this_X.std(axis=-1)
        std[std == 0] = 1
        this_X /= std[..., np.newaxis]

        # commit subject data
        X.append(this_X)
        y.extend(this_y)
        mask.append(this_mask)
    y = np.array(y)
    X = np.concatenate(X, axis=-1)
    mask = np.sum(mask, axis=0) > .5 * len(mask)
    mask = np.logical_and(mask, np.all(np.isfinite(X), axis=-1))
    X = X[mask, :].T
    tmp = np.zeros(list(mask.shape) + [len(X)])
    tmp[mask, :] = X.T
    mask_img = nibabel.Nifti1Image(mask.astype("uint8"), affine)
    X = nibabel.four_to_three(nibabel.Nifti1Image(tmp, affine))
    return X, y, mask_img


@fill_doc
def fetch_mixed_gambles(n_subjects=1, data_dir=None, url=None, resume=True,
                        return_raw_data=False, verbose=1):
    """Fetch Jimura "mixed gambles" dataset.

    See :footcite:`Jimura2012`.

    Parameters
    ----------
    n_subjects : :obj:`int`, optional
        The number of subjects to load. If ``None`` is given, all the
        subjects are used. Default=1.
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s
    return_raw_data : :obj:`bool`, optional
        If ``False``, then the data will transformed into an ``(X, y)``
        pair, suitable for machine learning routines. ``X`` is a list
        of ``n_subjects * 48`` :class:`~nibabel.nifti1.Nifti1Image`
        objects (where 48 is the number of trials), and ``y`` is an
        array of shape ``(n_subjects * 48,)``.
        Default=False.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, the attributes of interest are:

        - 'zmaps': :obj:`list` of :obj:`str`
          Paths to realigned gain betamaps (one nifti per subject).
        - 'gain': :obj:`list` of :class:`~nibabel.nifti1.Nifti1Image` \
        or ``None``
          If ``make_Xy`` is ``True``, this is a list of
          ``n_subjects * 48`` :class:`~nibabel.nifti1.Nifti1Image`
          objects, else it is ``None``.
        - 'y': :class:`~numpy.ndarray` of shape ``(n_subjects * 48,)`` \
        or ``None``
          If ``make_Xy`` is ``True``, then this is a
          :class:`~numpy.ndarray` of shape ``(n_subjects * 48,)``,
          else it is ``None``.

    References
    ----------
    .. footbibliography::

    """
    if n_subjects > 16:
        warnings.warn('Warning: there are only 16 subjects!')
        n_subjects = 16
    if url is None:
        url = ("https://www.nitrc.org/frs/download.php/7229/"
               "jimura_poldrack_2012_zmaps.zip")
    opts = dict(uncompress=True)
    files = [("zmaps%ssub%03i_zmaps.nii.gz" % (os.sep, (j + 1)), url, opts)
             for j in range(n_subjects)]
    data_dir = _get_dataset_dir('jimura_poldrack_2012_zmaps',
                                data_dir=data_dir)
    zmap_fnames = _fetch_files(data_dir, files, resume=resume, verbose=verbose)
    subject_id = np.repeat(np.arange(n_subjects), 6 * 8)
    data = Bunch(zmaps=zmap_fnames,
                 subject_id=subject_id)
    if not return_raw_data:
        X, y, mask_img = _load_mixed_gambles(check_niimg(data.zmaps,
                                                         return_iterator=True))
        data.zmaps, data.gain, data.mask_img = X, y, mask_img
    return data


@fill_doc
def fetch_megatrawls_netmats(dimensionality=100, timeseries='eigen_regression',
                             matrices='partial_correlation', data_dir=None,
                             resume=True, verbose=1):
    """Downloads and returns Network Matrices data from MegaTrawls release in HCP.

    This data can be used to predict relationships between imaging data and
    non-imaging behavioural measures such as age, sex, education, etc.
    The network matrices are estimated from functional connectivity
    datasets of 461 subjects. Full technical details in references.

    More information available in :footcite:`Smith2015b`,
    :footcite:`Smith2015a`, :footcite:`Filippini2009`,
    :footcite:`Smith2014`, and :footcite:`Reilly2009`.

    Parameters
    ----------
    dimensionality : int, optional
        Valid inputs are 25, 50, 100, 200, 300. By default, network matrices
        estimated using Group ICA brain parcellations of 100 components/dimensions
        will be returned. Default=100.

    timeseries : str, optional
        Valid inputs are 'multiple_spatial_regression' or 'eigen_regression'. By
        default 'eigen_regression', matrices estimated using first principal
        eigen component timeseries signals extracted from each subject data
        parcellations will be returned. Otherwise, 'multiple_spatial_regression'
        matrices estimated using spatial regressor based timeseries signals
        extracted from each subject data parcellations will be returned.
        Default='eigen_regression'.

    matrices : str, optional
        Valid inputs are 'full_correlation' or 'partial_correlation'. By default,
        partial correlation matrices will be returned otherwise if selected
        full correlation matrices will be returned.
        Default='partial_correlation'.
    %(data_dir)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : Bunch
        Dictionary-like object, the attributes are :

        - 'dimensions': int, consists of given input in dimensions.

        - 'timeseries': str, consists of given input in timeseries method.

        - 'matrices': str, consists of given type of specific matrices.

        - 'correlation_matrices': ndarray, consists of correlation matrices
          based on given type of matrices. Array size will depend on given
          dimensions (n, n).

        - 'description': data description

    References
    ----------
    .. footbibliography::

    Notes
    -----
    See description for terms & conditions on data usage.

    """
    url = "http://www.nitrc.org/frs/download.php/8037/Megatrawls.tgz"
    opts = {'uncompress': True}

    error_message = "Invalid {0} input is provided: {1}, choose one of them {2}"
    # standard dataset terms
    dimensionalities = [25, 50, 100, 200, 300]
    if dimensionality not in dimensionalities:
        raise ValueError(error_message.format('dimensionality', dimensionality,
                                              dimensionalities))
    timeseries_methods = ['multiple_spatial_regression', 'eigen_regression']
    if timeseries not in timeseries_methods:
        raise ValueError(error_message.format('timeseries', timeseries,
                                              timeseries_methods))
    output_matrices_names = ['full_correlation', 'partial_correlation']
    if matrices not in output_matrices_names:
        raise ValueError(error_message.format('matrices', matrices,
                                              output_matrices_names))

    dataset_name = 'Megatrawls'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir, verbose=verbose)
    description = _get_dataset_descr(dataset_name)

    timeseries_map = dict(multiple_spatial_regression='ts2', eigen_regression='ts3')
    matrices_map = dict(full_correlation='Znet1.txt', partial_correlation='Znet2.txt')
    filepath = [(os.path.join(
        '3T_Q1-Q6related468_MSMsulc_d%d_%s' % (dimensionality, timeseries_map[timeseries]),
        matrices_map[matrices]), url, opts)]

    # Fetch all the files
    files = _fetch_files(data_dir, filepath, resume=resume, verbose=verbose)

    # Load the files into arrays
    correlation_matrices = csv_to_array(files[0])

    return Bunch(
        dimensions=dimensionality,
        timeseries=timeseries,
        matrices=matrices,
        correlation_matrices=correlation_matrices,
        description=description)


@fill_doc
def fetch_surf_nki_enhanced(n_subjects=10, data_dir=None,
                            url=None, resume=True, verbose=1):
    """Download and load the NKI enhanced resting-state dataset,
    preprocessed and projected to the fsaverage5 space surface.

    See :footcite:`Nooner2012`.

    Direct download link :footcite:`NKIdataset`.

    .. versionadded:: 0.3

    Parameters
    ----------
    n_subjects : int, optional
        The number of subjects to load from maximum of 102 subjects.
        By default, 10 subjects will be loaded. If None is given,
        all 102 subjects will be loaded. Default=10.
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :

        - 'func_left': Paths to Gifti files containing resting state
                        time series left hemisphere
        - 'func_right': Paths to Gifti files containing resting state
                         time series right hemisphere
        - 'phenotypic': array containing tuple with subject ID, age,
                         dominant hand and sex for each subject.
        - 'description': data description of the release and references.

    References
    ----------
    .. footbibliography::

    """
    if url is None:
        url = 'https://www.nitrc.org/frs/download.php/'

    # Preliminary checks and declarations
    dataset_name = 'nki_enhanced_surface'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    ids = ['A00028185', 'A00033747', 'A00035072', 'A00035827', 'A00035840',
           'A00037112', 'A00037511', 'A00038998', 'A00039391', 'A00039431',
           'A00039488', 'A00040524', 'A00040623', 'A00040944', 'A00043299',
           'A00043520', 'A00043677', 'A00043722', 'A00045589', 'A00050998',
           'A00051063', 'A00051064', 'A00051456', 'A00051457', 'A00051477',
           'A00051513', 'A00051514', 'A00051517', 'A00051528', 'A00051529',
           'A00051539', 'A00051604', 'A00051638', 'A00051658', 'A00051676',
           'A00051678', 'A00051679', 'A00051726', 'A00051774', 'A00051796',
           'A00051835', 'A00051882', 'A00051925', 'A00051927', 'A00052070',
           'A00052117', 'A00052118', 'A00052126', 'A00052180', 'A00052197',
           'A00052214', 'A00052234', 'A00052307', 'A00052319', 'A00052499',
           'A00052502', 'A00052577', 'A00052612', 'A00052639', 'A00053202',
           'A00053369', 'A00053456', 'A00053474', 'A00053546', 'A00053576',
           'A00053577', 'A00053578', 'A00053625', 'A00053626', 'A00053627',
           'A00053874', 'A00053901', 'A00053927', 'A00053949', 'A00054038',
           'A00054153', 'A00054173', 'A00054358', 'A00054482', 'A00054532',
           'A00054533', 'A00054534', 'A00054621', 'A00054895', 'A00054897',
           'A00054913', 'A00054929', 'A00055061', 'A00055215', 'A00055352',
           'A00055353', 'A00055542', 'A00055738', 'A00055763', 'A00055806',
           'A00056097', 'A00056098', 'A00056164', 'A00056372', 'A00056452',
           'A00056489', 'A00056949']

    nitrc_ids = range(8260, 8464)
    max_subjects = len(ids)
    if n_subjects is None:
        n_subjects = max_subjects
    if n_subjects > max_subjects:
        warnings.warn('Warning: there are only %d subjects' % max_subjects)
        n_subjects = max_subjects
    ids = ids[:n_subjects]

    # Dataset description
    fdescr = _get_dataset_descr(dataset_name)

    # First, get the metadata
    phenotypic_file = 'NKI_enhanced_surface_phenotypics.csv'
    phenotypic = (phenotypic_file, url + '8470/pheno_nki_nilearn.csv',
                  {'move': phenotypic_file})

    phenotypic = _fetch_files(data_dir, [phenotypic], resume=resume,
                              verbose=verbose)[0]

    # Load the csv file
    phenotypic = np.genfromtxt(phenotypic, skip_header=True,
                               names=['Subject', 'Age',
                                      'Dominant Hand', 'Sex'],
                               delimiter=',', dtype=['U9', '<f8',
                                                     'U1', 'U1'],
                               encoding=None)

    # Keep phenotypic information for selected subjects
    int_ids = np.asarray(ids)
    phenotypic = phenotypic[[np.where(phenotypic['Subject'] == i)[0][0]
                             for i in int_ids]]

    # Download subjects' datasets
    func_right = []
    func_left = []
    for i in range(len(ids)):

        archive = url + '%i/%s_%s_preprocessed_fsaverage5_fwhm6.gii'
        func = os.path.join('%s', '%s_%s_preprocessed_fwhm6.gii')
        rh = _fetch_files(data_dir,
                          [(func % (ids[i], ids[i], 'right'),
                           archive % (nitrc_ids[2*i+1], ids[i], 'rh'),
                           {'move': func % (ids[i], ids[i], 'right')}
                            )],
                          resume=resume, verbose=verbose)
        lh = _fetch_files(data_dir,
                          [(func % (ids[i], ids[i], 'left'),
                           archive % (nitrc_ids[2*i], ids[i], 'lh'),
                           {'move': func % (ids[i], ids[i], 'left')}
                            )],
                          resume=resume, verbose=verbose)

        func_right.append(rh[0])
        func_left.append(lh[0])

    return Bunch(func_left=func_left, func_right=func_right,
                 phenotypic=phenotypic,
                 description=fdescr)


@fill_doc
def _fetch_development_fmri_participants(data_dir, url, verbose):
    """Helper function to fetch_development_fmri.

    This function helps in downloading and loading participants data from .tsv
    uploaded on Open Science Framework (OSF).

    The original .tsv file contains many columns but this function picks only
    those columns that are relevant.

    Parameters
    ----------
    %(data_dir)s
    %(url)s
    %(verbose)s

    Returns
    -------
    participants : numpy.ndarray
        Contains data of each subject age, age group, child or adult,
        gender, handedness.

    """
    dataset_name = 'development_fmri'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    if url is None:
        url = 'https://osf.io/yr3av/download'

    files = [('participants.tsv', url, {'move': 'participants.tsv'})]
    path_to_participants = _fetch_files(data_dir, files, verbose=verbose)[0]

    # Load path to participants
    dtype = [('participant_id', 'U12'), ('Age', '<f8'), ('AgeGroup', 'U6'),
             ('Child_Adult', 'U5'), ('Gender', 'U4'), ('Handedness', 'U4')]
    names = ['participant_id', 'Age', 'AgeGroup', 'Child_Adult', 'Gender',
             'Handedness']
    participants = csv_to_array(path_to_participants, skip_header=True,
                                dtype=dtype, names=names)
    return participants


@fill_doc
def _fetch_development_fmri_functional(participants, data_dir, url, resume,
                                       verbose):
    """Helper function to fetch_development_fmri.

    This function helps in downloading functional MRI data in Nifti
    and its confound corresponding to each subject.

    The files are downloaded from Open Science Framework (OSF).

    Parameters
    ----------
    participants : numpy.ndarray
        Should contain column participant_id which represents subjects id. The
        number of files are fetched based on ids in this column.
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    func : list of str (Nifti files)
        Paths to functional MRI data (4D) for each subject.

    regressors : list of str (tsv files)
        Paths to regressors related to each subject.

    """
    dataset_name = 'development_fmri'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    if url is None:
        # Download from the relevant OSF project, using hashes generated
        # from the OSF API. Note the trailing slash. For more info, see:
        # https://gist.github.com/emdupre/3cb4d564511d495ea6bf89c6a577da74
        url = 'https://osf.io/download/{}/'

    confounds = '{}_task-pixar_desc-confounds_regressors.tsv'
    func = '{0}_task-pixar_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz'

    # The gzip contains unique download keys per Nifti file and confound
    # pre-extracted from OSF. Required for downloading files.
    package_directory = os.path.dirname(os.path.abspath(__file__))
    dtype = [('participant_id', 'U12'), ('key_regressor', 'U24'),
             ('key_bold', 'U24')]
    names = ['participant_id', 'key_r', 'key_b']
    # csv file contains download information related to OpenScience(osf)
    osf_data = csv_to_array(os.path.join(package_directory, "data",
                                         "development_fmri.csv"),
                            skip_header=True, dtype=dtype, names=names)

    funcs = []
    regressors = []

    for participant_id in participants['participant_id']:
        this_osf_id = osf_data[osf_data['participant_id'] == participant_id]
        # Download regressors
        confound_url = url.format(this_osf_id['key_r'][0])
        regressor_file = [(confounds.format(participant_id),
                           confound_url,
                           {'move': confounds.format(participant_id)})]
        path_to_regressor = _fetch_files(data_dir, regressor_file,
                                         verbose=verbose)[0]
        regressors.append(path_to_regressor)
        # Download bold images
        func_url = url.format(this_osf_id['key_b'][0])
        func_file = [(func.format(participant_id, participant_id), func_url,
                      {'move': func.format(participant_id)})]
        path_to_func = _fetch_files(data_dir, func_file, resume=resume,
                                    verbose=verbose)[0]
        funcs.append(path_to_func)
    return funcs, regressors


@fill_doc
def fetch_development_fmri(n_subjects=None, reduce_confounds=True,
                           data_dir=None, resume=True, verbose=1,
                           age_group='both'):
    """Fetch movie watching based brain development dataset (fMRI)

    The data is downsampled to 4mm resolution for convenience with a repetition time (TR)
    of 2 secs. The origin of the data is coming from OpenNeuro. See Notes below.

    Please cite :footcite:`Richardson2018`
    if you are using this dataset.

    .. versionadded:: 0.5.2

    Parameters
    ----------
    n_subjects : int, optional
        The number of subjects to load. If None, all the subjects are
        loaded. Total 155 subjects.

    reduce_confounds : bool, optional
        If True, the returned confounds only include 6 motion parameters,
        mean framewise displacement, signal from white matter, csf, and
        6 anatomical compcor parameters. This selection only serves the
        purpose of having realistic examples. Depending on your research
        question, other confounds might be more appropriate.
        If False, returns all :term:`fMRIPrep` confounds.
        Default=True.
    %(data_dir)s
    %(resume)s
    %(verbose)s
    age_group : str, optional
        Default='both'. Which age group to fetch

        - 'adults' = fetch adults only (n=33, ages 18-39)
        - 'child' = fetch children only (n=122, ages 3-12)
        - 'both' = fetch full sample (n=155)

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :

        - 'func': list of str (Nifti files)
            Paths to downsampled functional MRI data (4D) for each subject.

        - 'confounds': list of str (tsv files)
            Paths to confounds related to each subject.

        - 'phenotypic': numpy.ndarray
            Contains each subject age, age group, child or adult, gender,
            handedness.

    Notes
    -----
    The original data is downloaded from OpenNeuro
    https://openneuro.org/datasets/ds000228/versions/1.0.0

    This fetcher downloads downsampled data that are available on Open
    Science Framework (OSF). Located here: https://osf.io/5hju4/files/

    Preprocessing details: https://osf.io/wjtyq/

    Note that if n_subjects > 2, and age_group is 'both',
    fetcher will return a ratio of children and adults representative
    of the total sample.

    References
    ----------
    .. footbibliography::

    """
    dataset_name = 'development_fmri'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=1)
    keep_confounds = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y',
                      'rot_z', 'framewise_displacement', 'a_comp_cor_00',
                      'a_comp_cor_01', 'a_comp_cor_02', 'a_comp_cor_03',
                      'a_comp_cor_04', 'a_comp_cor_05', 'csf',
                      'white_matter']

    # Dataset description
    fdescr = _get_dataset_descr(dataset_name)

    # Participants data: ids, demographics, etc
    participants = _fetch_development_fmri_participants(data_dir=data_dir,
                                                        url=None,
                                                        verbose=verbose)

    adult_count, child_count = _filter_func_regressors_by_participants(
            participants, age_group)  # noqa: E126
    max_subjects = adult_count + child_count

    n_subjects = _set_invalid_n_subjects_to_max(n_subjects,
                                                max_subjects,
                                                age_group)

    # To keep the proportion of children versus adults
    percent_total = float(n_subjects) / max_subjects
    n_child = np.round(percent_total * child_count).astype(int)
    n_adult = np.round(percent_total * adult_count).astype(int)

    # We want to return adults by default (i.e., `age_group=both`) or
    # if explicitly requested.
    if (age_group != 'child') and (n_subjects == 1):
        n_adult, n_child = 1, 0

    if (age_group == 'both') and (n_subjects == 2):
        n_adult, n_child = 1, 1

    participants = _filter_csv_by_n_subjects(participants, n_adult, n_child)

    funcs, regressors = _fetch_development_fmri_functional(participants,
                                                           data_dir=data_dir,
                                                           url=None,
                                                           resume=resume,
                                                           verbose=verbose)

    if reduce_confounds:
        regressors = _reduce_confounds(regressors, keep_confounds)
    return Bunch(func=funcs, confounds=regressors, phenotypic=participants,
                 description=fdescr)


def _filter_func_regressors_by_participants(participants, age_group):
    """ Filter functional and regressors based on participants
    """
    valid_age_groups = ('both', 'child', 'adult')
    if age_group not in valid_age_groups:
        raise ValueError("Wrong value for age_group={0}. "
                         "Valid arguments are: {1}".format(age_group,
                                                           valid_age_groups)
                         )

    child_adult = participants['Child_Adult'].tolist()

    if age_group != 'adult':
        child_count = child_adult.count('child')
    else:
        child_count = 0

    if age_group != 'child':
        adult_count = child_adult.count('adult')
    else:
        adult_count = 0
    return adult_count, child_count


def _filter_csv_by_n_subjects(participants, n_adult, n_child):
    """Restrict the csv files to the adequate number of subjects
    """
    child_ids = participants[participants['Child_Adult'] ==
                             'child']['participant_id'][:n_child]
    adult_ids = participants[participants['Child_Adult'] ==
                             'adult']['participant_id'][:n_adult]
    ids = np.hstack([adult_ids, child_ids])
    participants = participants[np.in1d(participants['participant_id'], ids)]
    participants = participants[np.argsort(participants, order='Child_Adult')]
    return participants


def _set_invalid_n_subjects_to_max(n_subjects, max_subjects, age_group):
    """ If n_subjects is invalid, sets it to max.
    """
    if n_subjects is None:
        n_subjects = max_subjects

    if (isinstance(n_subjects, numbers.Number) and
            ((n_subjects > max_subjects) or (n_subjects < 1))):
        warnings.warn("Wrong value for n_subjects={0}. The maximum "
                      "value (for age_group={1}) will be used instead: "
                      "n_subjects={2}"
                      .format(n_subjects, age_group, max_subjects))
        n_subjects = max_subjects
    return n_subjects


def _reduce_confounds(regressors, keep_confounds):
    reduced_regressors = []
    for in_file in regressors:
        out_file = in_file.replace('desc-confounds',
                                   'desc-reducedConfounds')
        if not os.path.isfile(out_file):
            confounds = pd.read_csv(in_file, delimiter='\t').to_records()
            selected_confounds = confounds[keep_confounds]
            header = '\t'.join(selected_confounds.dtype.names)
            np.savetxt(out_file, np.array(selected_confounds.tolist()),
                       header=header, delimiter='\t', comments='')
        reduced_regressors.append(out_file)
    return reduced_regressors


# datasets originally belonging to nistats follow


@fill_doc
def fetch_language_localizer_demo_dataset(data_dir=None, verbose=1):
    """Download language localizer demo dataset.

    Parameters
    ----------
    %(data_dir)s
    %(verbose)s

    Returns
    -------
    data_dir : string
        Path to downloaded dataset.

    downloaded_files : list of string
        Absolute paths of downloaded files on disk

    """
    url = 'https://osf.io/3dj2a/download'
    # When it starts working again change back to:
    # url = 'https://osf.io/nh987/download'
    main_folder = 'fMRI-language-localizer-demo-dataset'

    data_dir = _get_dataset_dir(main_folder, data_dir=data_dir,
                                verbose=verbose)
    # The files_spec needed for _fetch_files
    files_spec = [(main_folder + '.zip', url, {'move': main_folder + '.zip'})]
    # Only download if directory is empty
    # Directory will have been created by the call to _get_dataset_dir above
    if not os.listdir(data_dir):
        downloaded_files = _fetch_files(data_dir, files_spec, resume=True,
                                        verbose=verbose)
        _uncompress_file(downloaded_files[0])

    file_list = [os.path.join(path, f) for
                 path, dirs, files in os.walk(data_dir) for f in files]
    return data_dir, sorted(file_list)


@fill_doc
def fetch_bids_langloc_dataset(data_dir=None, verbose=1):
    """Download language localizer example :term:`bids<BIDS>` dataset.

    Parameters
    ----------
    %(data_dir)s
    %(verbose)s

    Returns
    -------
    data_dir : string
        Path to downloaded dataset.

    downloaded_files : list of string
        Absolute paths of downloaded files on disk.

    """
    url = 'https://files.osf.io/v1/resources/9q7dv/providers/osfstorage/5888d9a76c613b01fc6acc4e'  # noqa: E501
    dataset_name = 'bids_langloc_example'
    main_folder = 'bids_langloc_dataset'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    # The files_spec needed for _fetch_files
    files_spec = [(main_folder + '.zip', url, {'move': main_folder + '.zip'})]
    if not os.path.exists(os.path.join(data_dir, main_folder)):
        downloaded_files = _fetch_files(data_dir, files_spec, resume=True,
                                        verbose=verbose)
        _uncompress_file(downloaded_files[0])
    main_path = os.path.join(data_dir, main_folder)
    file_list = [os.path.join(path, f) for
                 path, dirs, files in os.walk(main_path) for f in files]
    return os.path.join(data_dir, main_folder), sorted(file_list)


@fill_doc
def fetch_openneuro_dataset_index(
    data_dir=None,
    dataset_version='ds000030_R1.0.4',
    verbose=1,
):
    """Download a file with OpenNeuro :term:`BIDS` dataset index.

    .. deprecated:: 0.9.2
        `fetch_openneuro_dataset_index` will be removed in 0.11.

    Downloading the index allows to explore the dataset directories
    to select specific files to download. The index is a sorted list of urls.

    Parameters
    ----------
    %(data_dir)s
    dataset_version : :obj:`str`, optional
        Dataset version name. Assumes it is of the form [name]_[version].
        Default='ds000030_R1.0.4'.

        .. warning:: Any value other than the default will be ignored.

    %(verbose)s'

    Returns
    -------
    urls_path : :obj:`str`
        Path to downloaded dataset index.
    urls : :obj:`list` of :obj:`str`
        Sorted list of dataset directories.
    """
    warnings.warn(
        (
            'The "fetch_openneuro_dataset_index" function was deprecated in '
            'version 0.9.2, and will be removed in 0.11. '
            'Please use "fetch_ds000030_urls" instead.'
        ),
        DeprecationWarning,
    )

    DATASET_VERSION = 'ds000030_R1.0.4'
    if dataset_version != DATASET_VERSION:
        warnings.warn(
            (
                'An improper dataset_version has been provided. '
                '"ds000030_R1.0.4" will be downloaded.'
            ),
            UserWarning,
        )

    urls_path, urls = fetch_ds000030_urls(data_dir=data_dir, verbose=verbose)
    return urls_path, urls


@fill_doc
def fetch_ds000030_urls(data_dir=None, verbose=1):
    """Fetch URLs for files from the ds000030 :term:`BIDS` dataset.

    .. versionadded:: 0.9.2

    This dataset is version 1.0.4 of the "UCLA Consortium for
    Neuropsychiatric Phenomics LA5c" dataset
    :footcite:p:`Poldrack2016`.

    Downloading the index allows users to explore the dataset directories
    to select specific files to download.
    The index is a sorted list of urls.

    Parameters
    ----------
    %(data_dir)s
    %(verbose)s

    Returns
    -------
    urls_path : :obj:`str`
        Path to downloaded dataset index.

    urls : :obj:`list` of :obj:`str`
        Sorted list of dataset directories.

    References
    ----------
    .. footbibliography::
    """
    DATA_PREFIX = 'ds000030/ds000030_R1.0.4/uncompressed'
    FILE_URL = 'https://osf.io/86xj7/download'

    data_dir = _get_dataset_dir(
        DATA_PREFIX,
        data_dir=data_dir,
        verbose=verbose,
    )

    final_download_path = os.path.join(data_dir, 'urls.json')
    downloaded_file_path = _fetch_files(
        data_dir=data_dir,
        files=[(
            final_download_path,
            FILE_URL,
            {'move': final_download_path},
        )],
        resume=True
    )
    urls_path = downloaded_file_path[0]
    with open(urls_path, 'r') as json_file:
        urls = json.load(json_file)

    return urls_path, urls


def select_from_index(urls, inclusion_filters=None, exclusion_filters=None,
                      n_subjects=None):
    """Select subset of urls with given filters.

    Parameters
    ----------
    urls : list of str
        List of dataset urls obtained from index download.

    inclusion_filters : list of str, optional
        List of unix shell-style wildcard strings
        that will be used to filter the url list.
        If a filter matches the url it is retained for download.
        Multiple filters work on top of each other.
        Like an "and" logical operator, creating a more restrictive query.
        Inclusion and exclusion filters apply together.
        For example the filter '*task-rest*'' would keep only urls
        that contain the 'task-rest' string.

    exclusion_filters : list of str, optional
        List of unix shell-style wildcard strings
        that will be used to filter the url list.
        If a filter matches the url it is discarded for download.
        Multiple filters work on top of each other.
        Like an "and" logical operator, creating a more restrictive query.
        Inclusion and exclusion filters apply together.
        For example the filter '*task-rest*' would discard all urls
        that contain the 'task-rest' string.

    n_subjects : int, optional
        Number of subjects to download from the dataset. All by default.

    Returns
    -------
    urls : list of string
        Sorted list of filtered dataset directories.

    """
    inclusion_filters = inclusion_filters or []
    exclusion_filters = exclusion_filters or []
    # We apply filters to the urls
    for exclusion in exclusion_filters:
        urls = [url for url in urls if not fnmatch.fnmatch(url, exclusion)]
    for inclusion in inclusion_filters:
        urls = [url for url in urls if fnmatch.fnmatch(url, inclusion)]

    # subject selection filter
    # from the url list we infer all available subjects like 'sub-xxx/'
    subject_regex = 'sub-[a-z|A-Z|0-9]*[_./]'

    def infer_subjects(urls):
        subjects = set()
        for url in urls:
            if 'sub-' in url:
                subjects.add(re.search(subject_regex, url).group(0)[:-1])
        return sorted(subjects)

    # We get a list of subjects (for the moment the first n subjects)
    selected_subjects = set(infer_subjects(urls)[:n_subjects])
    # We exclude urls of subjects not selected
    urls = [
        url for url in urls
        if 'sub-' not in url or re.search(subject_regex, url).group(0)[:-1]
           in selected_subjects
    ]
    return urls


def patch_openneuro_dataset(file_list):
    """Add symlinks for files not named according to :term:`BIDS` conventions.

    .. warning::
        This function uses a series of hardcoded patterns to generate the
        corrected filenames.
        These patterns are not comprehensive and this function is not
        guaranteed to produce BIDS-compliant files.

    Parameters
    ----------
    file_list : :obj:`list` of :obj:`str`
        A list of filenames to update.
    """
    REPLACEMENTS = {
        '_T1w_brainmask': '_desc-brain_mask',
        '_T1w_preproc': '_desc-preproc_T1w',
        '_T1w_space-MNI152NLin2009cAsym_brainmask':
            '_space-MNI152NLin2009cAsym_desc-brain_mask',
        '_T1w_space-MNI152NLin2009cAsym_class-':
            '_space-MNI152NLin2009cAsym_label-',
        '_T1w_space-MNI152NLin2009cAsym_preproc':
            '_space-MNI152NLin2009cAsym_desc-preproc_T1w',
        '_bold_confounds': '_desc-confounds_regressors',
        '_bold_space-MNI152NLin2009cAsym_brainmask':
            '_space-MNI152NLin2009cAsym_desc-brain_mask',
        '_bold_space-MNI152NLin2009cAsym_preproc':
            '_space-MNI152NLin2009cAsym_desc-preproc_bold'
    }

    # Create a symlink if a file with the modified filename does not exist
    for old_pattern, new_pattern in REPLACEMENTS.items():
        for name in file_list:
            if old_pattern in name:
                new_name = name.replace(old_pattern, new_pattern)
                if not os.path.exists(new_name):
                    os.symlink(name, new_name)


@fill_doc
def fetch_openneuro_dataset(
    urls=None,
    data_dir=None,
    dataset_version='ds000030_R1.0.4',
    verbose=1,
):
    """Download OpenNeuro :term:`BIDS` dataset.

    This function specifically downloads files from a series of URLs.
    Unless you use :func:`fetch_ds000030_urls` or the default parameters,
    it is up to the user to ensure that the URLs are correct,
    and that they are associated with an OpenNeuro dataset.

    Parameters
    ----------
    urls : list of string, optional
        List of URLs to dataset files to download.
        If not specified, all files from the default dataset
        (``ds000030_R1.0.4``) will be downloaded.
    %(data_dir)s
    dataset_version : string, optional
        Dataset version name. Assumes it is of the form [name]_[version].
        Default is ``ds000030_R1.0.4``.
    %(verbose)s

    Returns
    -------
    data_dir : string
        Path to downloaded dataset.

    downloaded_files : list of string
        Absolute paths of downloaded files on disk.

    Notes
    -----
    The default dataset downloaded by this function is the
    "UCLA Consortium for Neuropsychiatric Phenomics LA5c" dataset
    :footcite:p:`Poldrack2016`.

    This copy includes filenames that are not compliant with the current
    version of :term:`BIDS`, so this function also calls
    :func:`patch_openneuro_dataset` to generate BIDS-compliant symlinks.

    See Also
    --------
    :func:`fetch_ds000030_urls`
    :func:`patch_openneuro_dataset`

    References
    ----------
    .. footbibliography::
    """
    # if urls are not specified we download the complete dataset index
    if urls is None:
        DATASET_VERSION = 'ds000030_R1.0.4'
        if dataset_version != DATASET_VERSION:
            warnings.warn(
                'If `dataset_version` is not "ds000030_R1.0.4", '
                '`urls` must be specified. Downloading "ds000030_R1.0.4".'
            )

        data_prefix = '{}/{}/uncompressed'.format(
            DATASET_VERSION.split('_')[0],
            DATASET_VERSION,
        )
        orig_data_dir = data_dir
        data_dir = _get_dataset_dir(
            data_prefix,
            data_dir=data_dir,
            verbose=verbose,
        )

        _, urls = fetch_ds000030_urls(
            data_dir=orig_data_dir,
            verbose=verbose,
        )
    else:
        data_prefix = '{}/{}/uncompressed'.format(
            dataset_version.split('_')[0],
            dataset_version,
        )
        data_dir = _get_dataset_dir(
            data_prefix,
            data_dir=data_dir,
            verbose=verbose,
        )

    # The files_spec needed for _fetch_files
    files_spec = []
    files_dir = []

    # Check that data prefix is found in each URL
    bad_urls = [url for url in urls if data_prefix not in url]
    if bad_urls:
        raise ValueError(
            f'data_prefix ({data_prefix}) is not found in at least one URL. '
            'This indicates that the URLs do not correspond to the '
            'dataset_version provided.\n'
            f'Affected URLs: {bad_urls}'
        )

    for url in urls:
        url_path = url.split(data_prefix + '/')[1]
        file_dir = os.path.join(data_dir, url_path)
        files_spec.append((os.path.basename(file_dir), url, {}))
        files_dir.append(os.path.dirname(file_dir))

    # download the files
    downloaded = []
    for file_spec, file_dir in zip(files_spec, files_dir):
        # Timeout errors are common in the s3 connection so we try to avoid
        # failure of the dataset download for a transient instability
        success = False
        download_attempts = 4
        while download_attempts > 0 and not success:
            try:
                downloaded_files = _fetch_files(
                    file_dir,
                    [file_spec],
                    resume=True,
                    verbose=verbose,
                )
                downloaded += downloaded_files
                success = True
            except Exception:
                download_attempts -= 1

        if not success:
            raise Exception(f'multiple failures downloading {file_spec[1]}')

    patch_openneuro_dataset(downloaded)

    return data_dir, sorted(downloaded)


@fill_doc
def fetch_localizer_first_level(data_dir=None, verbose=1):
    """Download a first-level localizer fMRI dataset

    Parameters
    ----------
    %(data_dir)s
    %(verbose)s

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, with the keys:
        epi_img: the input 4D image
        events: a csv file describing the paardigm

    """
    url = 'https://osf.io/2bqxn/download'
    epi_img = 'sub-12069_task-localizer_space-MNI305.nii.gz'
    events = 'sub-12069_task-localizer_events.tsv'
    opts = {'uncompress': True}
    options = ('epi_img', 'events')
    dir_ = 'localizer_first_level'
    filenames = [(os.path.join(dir_, name), url, opts)
                 for name in [epi_img, events]]

    dataset_name = 'localizer_first_level'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, filenames, verbose=verbose)

    params = dict(list(zip(options, files)))
    return Bunch(**params)


def _download_spm_auditory_data(data_dir, subject_dir, subject_id):
    print('Data absent, downloading...')
    url = ('http://www.fil.ion.ucl.ac.uk/spm/download/data/MoAEpilot/'
           'MoAEpilot.zip')
    archive_path = os.path.join(subject_dir, os.path.basename(url))
    _fetch_file(url, subject_dir)
    try:
        _uncompress_file(archive_path)
    except Exception:
        print('Archive corrupted, trying to download it again.')
        return fetch_spm_auditory(data_dir=data_dir, data_name='',
                                  subject_id=subject_id)


def _prepare_downloaded_spm_auditory_data(subject_dir):
    """ Uncompresses downloaded spm_auditory dataset and organizes
    the data into appropriate directories.

    Parameters
    ----------
    subject_dir : string
        Path to subject's data directory.

    Returns
    -------
    _subject_data : skl.Bunch object
        Scikit-Learn Bunch object containing data of a single subject
        from the SPM Auditory dataset.

    """
    subject_data = {}
    spm_auditory_data_files = ["fM00223/fM00223_%03i.img" % index
                               for index in range(4, 100)]
    spm_auditory_data_files.append("sM00223/sM00223_002.img")

    for file_name in spm_auditory_data_files:
        file_path = os.path.join(subject_dir, file_name)
        if os.path.exists(file_path):
            subject_data[file_name] = file_path
        else:
            print('%s missing from filelist!' % file_name)
            return None

    _subject_data = {}
    _subject_data['func'] = sorted(
        [subject_data[x] for x in subject_data.keys()
         if re.match(r'^fM00223_0\d\d\.img$',
                     os.path.basename(x))])

    # volumes for this dataset of shape (64, 64, 64, 1); let's fix this
    for x in _subject_data['func']:
        vol = nib.load(x)
        if len(vol.shape) == 4:
            vol = nib.Nifti1Image(get_data(vol)[:, :, :, 0],
                                  vol.affine)
            nib.save(vol, x)

    _subject_data['anat'] = [subject_data[x] for x in subject_data.keys()
                             if re.match(r'^sM00223_002\.img$',
                                         os.path.basename(x))][0]

    # ... same thing for anat
    vol = nib.load(_subject_data['anat'])
    if len(vol.shape) == 4:
        vol = nib.Nifti1Image(get_data(vol)[:, :, :, 0],
                              vol.affine)
        nib.save(vol, _subject_data['anat'])

    return Bunch(**_subject_data)


def _make_path_events_file_spm_auditory_data(spm_auditory_data):
    """Accepts data for spm_auditory dataset as Bunch
    and constructs the filepath for its events descriptor file.

    Parameters
    ----------
    spm_auditory_data : Bunch

    Returns
    -------
    events_filepath : string
        Full path to the events.tsv file for spm_auditory dataset.

    """
    events_file_location = os.path.dirname(spm_auditory_data['func'][0])
    events_filename = os.path.basename(events_file_location) + '_events.tsv'
    events_filepath = os.path.join(events_file_location, events_filename)
    return events_filepath


def _make_events_file_spm_auditory_data(events_filepath):
    """Accepts destination filepath including filename and
    creates the events.tsv file for the spm_auditory dataset.

    Parameters
    ----------
    events_filepath : string
        The path where the events file will be created.

    Returns
    -------
    None

    """
    tr = 7.
    epoch_duration = 6 * tr  # duration in seconds
    conditions = ['rest', 'active'] * 8
    n_blocks = len(conditions)
    duration = epoch_duration * np.ones(n_blocks)
    onset = np.linspace(0, (n_blocks - 1) * epoch_duration, n_blocks)
    events = pd.DataFrame(
        {'onset': onset, 'duration': duration, 'trial_type': conditions})
    events.to_csv(events_filepath, sep='\t', index=False,
                  columns=['onset', 'duration', 'trial_type'])


@fill_doc
def fetch_spm_auditory(data_dir=None, data_name='spm_auditory',
                       subject_id='sub001', verbose=1):
    """Function to fetch SPM auditory single-subject data.

    See :footcite:`spm_auditory`.

    Parameters
    ----------
    %(data_dir)s
    data_name : string, optional
        Name of the dataset. Default='spm_auditory'.

    subject_id : string, optional
        Indicates which subject to retrieve.
        Default='sub001'.
    %(verbose)s

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are:
        - 'func': string list. Paths to functional images
        - 'anat': string list. Path to anat image

    References
    ----------
    .. footbibliography::

    """
    data_dir = _get_dataset_dir(data_name, data_dir=data_dir,
                                verbose=verbose)
    subject_dir = os.path.join(data_dir, subject_id)
    if not os.path.exists(subject_dir):
        _download_spm_auditory_data(data_dir, subject_dir, subject_id)
    spm_auditory_data = _prepare_downloaded_spm_auditory_data(subject_dir)
    try:
        spm_auditory_data['events']
    except KeyError:
        events_filepath = _make_path_events_file_spm_auditory_data(
            spm_auditory_data)
        if not os.path.isfile(events_filepath):
            _make_events_file_spm_auditory_data(events_filepath)
        spm_auditory_data['events'] = events_filepath
    return spm_auditory_data


def _get_func_data_spm_multimodal(subject_dir, session, _subject_data):
    session_func = sorted(glob.glob(
        os.path.join(
            subject_dir,
            ('fMRI/Session%i/fMETHODS-000%i-*-01.img' % (
                session, session + 4)
             )
        )
    ))
    if len(session_func) < 390:
        print('Missing %i functional scans for session %i.' % (
            390 - len(session_func), session))
        return None

    _subject_data['func%i' % (session)] = session_func
    return _subject_data


def _get_session_trials_spm_multimodal(subject_dir, session, _subject_data):
    sess_trials = os.path.join(
        subject_dir,
        'fMRI/trials_ses%i.mat' % (session))
    if not os.path.isfile(sess_trials):
        print('Missing session file: %s' % sess_trials)
        return None

    _subject_data['trials_ses%i' % (session)] = sess_trials
    return _subject_data


def _get_anatomical_data_spm_multimodal(subject_dir, _subject_data):
    anat = os.path.join(subject_dir, 'sMRI/smri.img')
    if not os.path.isfile(anat):
        print('Missing structural image.')
        return None

    _subject_data['anat'] = anat
    return _subject_data


def _glob_spm_multimodal_fmri_data(subject_dir):
    """glob data from subject_dir."""
    _subject_data = {'slice_order': 'descending'}

    for session in range(1, 3):
        # glob func data for session
        _subject_data = _get_func_data_spm_multimodal(subject_dir,
                                                      session,
                                                      _subject_data)
        if not _subject_data:
            return None
        # glob trials .mat file
        _subject_data = _get_session_trials_spm_multimodal(subject_dir,
                                                           session,
                                                           _subject_data)
        if not _subject_data:
            return None
        try:
            events = _make_events_file_spm_multimodal_fmri(_subject_data,
                                                           session)
        except MatReadError as mat_err:
            warnings.warn(
                '{}. An events.tsv file '
                'cannot be generated'.format(str(mat_err)))
        else:
            events_filepath = _make_events_filepath_spm_multimodal_fmri(
                _subject_data, session)
            events.to_csv(events_filepath, sep='\t', index=False)
            _subject_data['events{}'.format(session)] = events_filepath

    # glob for anat data
    _subject_data = _get_anatomical_data_spm_multimodal(subject_dir,
                                                        _subject_data)
    if not _subject_data:
        return None

    return Bunch(**_subject_data)


def _download_data_spm_multimodal(data_dir, subject_dir, subject_id):
    print('Data absent, downloading...')
    urls = [
        # fmri
        ('http://www.fil.ion.ucl.ac.uk/spm/download/data/mmfaces/'
         'multimodal_fmri.zip'),
        # structural
        ('http://www.fil.ion.ucl.ac.uk/spm/download/data/mmfaces/'
         'multimodal_smri.zip')
    ]

    for url in urls:
        archive_path = os.path.join(subject_dir, os.path.basename(url))
        _fetch_file(url, subject_dir)
        try:
            _uncompress_file(archive_path)
        except Exception:
            print('Archive corrupted, trying to download it again.')
            return fetch_spm_multimodal_fmri(data_dir=data_dir,
                                             data_name='',
                                             subject_id=subject_id)

    return _glob_spm_multimodal_fmri_data(subject_dir)


def _make_events_filepath_spm_multimodal_fmri(_subject_data, session):
    key = 'trials_ses{}'.format(session)
    events_file_location = os.path.dirname(_subject_data[key])
    events_filename = 'session{}_events.tsv'.format(session)
    events_filepath = os.path.join(events_file_location, events_filename)
    return events_filepath


def _make_events_file_spm_multimodal_fmri(_subject_data, session):
    tr = 2.
    timing = loadmat(_subject_data['trials_ses%i' % (session)],
                     squeeze_me=True, struct_as_record=False)
    faces_onsets = timing['onsets'][0].ravel()
    scrambled_onsets = timing['onsets'][1].ravel()
    onsets = np.hstack((faces_onsets, scrambled_onsets))
    onsets *= tr  # because onsets were reporting in 'scans' units
    conditions = (
        ['faces'] * len(faces_onsets) + ['scrambled'] * len(scrambled_onsets)
    )
    duration = np.ones_like(onsets)
    events = pd.DataFrame({'trial_type': conditions, 'onset': onsets,
                           'duration': duration})
    return events


@fill_doc
def fetch_spm_multimodal_fmri(data_dir=None, data_name='spm_multimodal_fmri',
                              subject_id='sub001', verbose=1):
    """Fetcher for Multi-modal Face Dataset.

    See :footcite:`spm_multiface`.

    Parameters
    ----------
    %(data_dir)s
    data_name : string, optional
        Name of the dataset. Default='spm_multimodal_fmri'.

    subject_id : string, optional
        Indicates which subject to retrieve. Default='sub001'.
    %(verbose)s

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are:
        - 'func1': string list. Paths to functional images for session 1
        - 'func2': string list. Paths to functional images for session 2
        - 'trials_ses1': string list. Path to onsets file for session 1
        - 'trials_ses2': string list. Path to onsets file for session 2
        - 'anat': string. Path to anat file

    References
    ----------
    .. footbibliography::

    """
    data_dir = _get_dataset_dir(data_name, data_dir=data_dir, verbose=verbose)
    subject_dir = os.path.join(data_dir, subject_id)

    # maybe data_dir already contains the data ?
    data = _glob_spm_multimodal_fmri_data(subject_dir)
    if data is not None:
        return data

    # No. Download the data
    return _download_data_spm_multimodal(data_dir, subject_dir, subject_id)


@fill_doc
def fetch_fiac_first_level(data_dir=None, verbose=1):
    """Download a first-level fiac fMRI dataset (2 sessions)

    Parameters
    ----------
    %(data_dir)s
    %(verbose)s

    """
    data_dir = _get_dataset_dir('fiac_nilearn.glm', data_dir=data_dir,
                                verbose=verbose)

    def _glob_fiac_data():
        """glob data from subject_dir."""
        _subject_data = {}
        subject_dir = os.path.join(data_dir, 'nipy-data-0.2/data/fiac/fiac0')
        for session in [1, 2]:
            # glob func data for session
            session_func = os.path.join(subject_dir, 'run%i.nii.gz' % session)
            if not os.path.isfile(session_func):
                print('Missing functional scan for session %i.' % session)
                return None

            _subject_data['func%i' % session] = session_func

            # glob design matrix .npz file
            sess_dmtx = os.path.join(subject_dir,
                                     'run%i_design.npz' % session)
            if not os.path.isfile(sess_dmtx):
                print('Missing session file: %s' % sess_dmtx)
                return None

            _subject_data['design_matrix%i' % session] = sess_dmtx

        # glob for mask data
        mask = os.path.join(subject_dir, 'mask.nii.gz')
        if not os.path.isfile(mask):
            print('Missing mask image.')
            return None

        _subject_data['mask'] = mask
        return Bunch(**_subject_data)

    # maybe data_dir already contains the data ?
    data = _glob_fiac_data()
    if data is not None:
        return data

    # No. Download the data
    print('Data absent, downloading...')
    url = 'https://nipy.org/data-packages/nipy-data-0.2.tar.gz'

    archive_path = os.path.join(data_dir, os.path.basename(url))
    _fetch_file(url, data_dir)
    try:
        _uncompress_file(archive_path)
    except Exception:
        print('Archive corrupted, trying to download it again.')
        return fetch_fiac_first_level(data_dir=data_dir)

    return _glob_fiac_data()
