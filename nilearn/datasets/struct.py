"""
Downloading NeuroImaging datasets: structural datasets

# License: simplified BSD
"""

import warnings
import os
import functools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import binary_closing
from sklearn.utils import Bunch

from .utils import (_get_dataset_dir, _fetch_files, _get_dataset_descr)

from ..image import new_img_like, get_data, resampling
from .._utils import check_niimg, fill_doc

_package_directory = os.path.dirname(os.path.abspath(__file__))
MNI152_FILE_PATH = os.path.join(
    _package_directory, "data",
    "mni_icbm152_t1_tal_nlin_sym_09a_converted.nii.gz")
GM_MNI152_FILE_PATH = os.path.join(
    _package_directory, "data",
    "mni_icbm152_gm_tal_nlin_sym_09a_converted.nii.gz")
WM_MNI152_FILE_PATH = os.path.join(
    _package_directory, "data",
    "mni_icbm152_wm_tal_nlin_sym_09a_converted.nii.gz")
FSAVERAGE5_PATH = os.path.join(_package_directory, "data", "fsaverage5")

_LEGACY_FORMAT_MSG = (
    "`legacy_format` will default to `False` in release 0.11. "
    "Dataset fetchers will then return pandas dataframes by default "
    "instead of recarrays."
)


@fill_doc
def fetch_icbm152_2009(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the ICBM152 template (dated 2009).

    %(templateflow)s

    For more information, see :footcite:`Fonov2011`,
    :footcite:`Fonov2009`, and :footcite:`Collins1999`.

    Parameters
    ----------
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, interest keys are:

        - "t1": str,
          Path to T1-weighted anatomical image
        - "t2": str,
          Path to T2-weighted anatomical image
        - "t2_relax": str,
          Path to anatomical image obtained with the T2 relaxometry
        - "pd": str,
          Path to the proton density weighted anatomical image
        - "gm": str,
          Path to grey matter segmented image
        - "wm": str,
          Path to white matter segmented image
        - "csf": str,
          Path to cerebrospinal fluid segmented image
        - "eye_mask": str,
          Path to eye mask useful to mask out part of MRI images
        - "face_mask": str,
          Path to face mask useful to mask out part of MRI images
        - "mask": str,
          Path to whole brain mask useful to mask out skull areas

    See Also
    --------
    nilearn.datasets.load_mni152_template: to load MNI152 T1 template.

    nilearn.datasets.load_mni152_gm_template: to load MNI152 grey matter
        template.

    nilearn.datasets.load_mni152_wm_template: to load MNI152 white matter
        template.

    nilearn.datasets.load_mni152_brain_mask: to load MNI152 whole brain mask.

    nilearn.datasets.load_mni152_gm_mask: to load MNI152 grey matter mask.

    nilearn.datasets.load_mni152_wm_mask: to load MNI152 white matter mask.

    nilearn.datasets.fetch_icbm152_brain_gm_mask: to fetch only ICBM grey
        matter mask.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    For more information about this dataset's structure:
    http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009

    The original download URL is
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip

    TemplateFlow repository for ICBM152 2009

    Symmetric: https://github.com/templateflow/tpl-MNI152NLin2009cSym

    Asymmetric: https://github.com/templateflow/tpl-MNI152NLin2009cSAsym


    """
    if url is None:
        # The URL can be retrieved from the nilearn account on OSF (Open
        # Science Framework), https://osf.io/4r3jt/quickfiles/
        # Clicking on the "share" button gives the root of the URL.
        url = "https://osf.io/7pj92/download"
    opts = {'uncompress': True}

    keys = ("csf", "gm", "wm",
            "pd", "t1", "t2", "t2_relax",
            "eye_mask", "face_mask", "mask")
    filenames = [(os.path.join("mni_icbm152_nlin_sym_09a", name), url, opts)
                 for name in (
        "mni_icbm152_csf_tal_nlin_sym_09a.nii.gz",
        "mni_icbm152_gm_tal_nlin_sym_09a.nii.gz",
        "mni_icbm152_wm_tal_nlin_sym_09a.nii.gz",

        "mni_icbm152_pd_tal_nlin_sym_09a.nii.gz",
        "mni_icbm152_t1_tal_nlin_sym_09a.nii.gz",
        "mni_icbm152_t2_tal_nlin_sym_09a.nii.gz",
        "mni_icbm152_t2_relx_tal_nlin_sym_09a.nii.gz",

        "mni_icbm152_t1_tal_nlin_sym_09a_eye_mask.nii.gz",
        "mni_icbm152_t1_tal_nlin_sym_09a_face_mask.nii.gz",
        "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii.gz")]

    dataset_name = 'icbm152_2009'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    sub_files = _fetch_files(data_dir, filenames, resume=resume,
                             verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)

    params = dict([('description', fdescr)] + list(zip(keys, sub_files)))
    return Bunch(**params)


@functools.lru_cache(maxsize=3)
def load_mni152_template(resolution=None):
    """Load the MNI152 skullstripped T1 template.
    This function takes the skullstripped, re-scaled 1mm-resolution version of
    the MNI ICBM152 T1 template and re-samples it using a different resolution,
    if specified.

    For more information, see :footcite:`Fonov2011`,
    and :footcite:`Fonov2009`.

    Parameters
    ----------
    resolution: int, optional, Default = 1
        If resolution is different from 1, the template is re-sampled with the
        specified resolution.

        .. versionadded:: 0.8.1

    Returns
    -------
    mni152_template : Nifti1Image, image representing the re-sampled
        whole-brain template

    See Also
    --------
    nilearn.datasets.fetch_icbm152_2009: for details regarding the difference
        between NiLearn and :term:`fMRIPrep` ICBM152 template.

    nilearn.datasets.load_mni152_gm_template : for details about version of the
        MNI152 grey-matter template.

    nilearn.datasets.load_mni152_wm_template : for details about version of the
        MNI152 white-matter template.

    References
    ----------
    .. footbibliography::

    """

    resolution = resolution or 1

    brain_template = check_niimg(MNI152_FILE_PATH)

    # Typecasting
    brain_data = get_data(brain_template).astype("float32")

    # Re-scale template from 0 to 1
    brain_data /= brain_data.max()
    new_brain_template = new_img_like(brain_template, brain_data)

    # Resample template according to the pre-specified resolution, if different
    # than 1
    if resolution != 1:
        new_brain_template = resampling.resample_img(new_brain_template,
                                                     np.eye(3) * resolution)

    return new_brain_template


def load_mni152_gm_template(resolution=None):
    """Load the MNI152 grey-matter template.
    This function takes the re-scaled 1mm-resolution version of the grey-matter
    MNI ICBM152 template and re-samples it using a different resolution,
    if specified.

    .. versionadded:: 0.8.1

    Parameters
    ----------
    resolution: int, optional, Default = 1
        If resolution is different from 1, the template is re-sampled with the
        specified resolution.

    Returns
    -------
    gm_mni152_template : Nifti1Image, image representing the resampled
        grey-matter template

    See Also
    --------
    nilearn.datasets.load_mni152_template : for details about version of the
        MNI152 T1 template.

    nilearn.datasets.load_mni152_wm_template : for details about version of the
        MNI152 white-matter template.

    """

    resolution = resolution or 1

    gm_template = check_niimg(GM_MNI152_FILE_PATH)

    # Typecasting
    gm_data = get_data(gm_template).astype("float32")

    # Re-scale template from 0 to 1
    gm_data /= gm_data.max()
    new_gm_template = new_img_like(gm_template, gm_data)

    # Resample template according to the pre-specified resolution, if different
    # than 1
    if resolution != 1:
        new_gm_template = resampling.resample_img(new_gm_template,
                                                  np.eye(3) * resolution)

    return new_gm_template


def load_mni152_wm_template(resolution=None):
    """Load the MNI152 white-matter template.
    This function takes the re-scaled 1mm-resolution version of the
    white-matter MNI ICBM152 template and re-samples it using a different
    resolution, if specified.

    .. versionadded:: 0.8.1

    Parameters
    ----------
    resolution: int, optional, Default = 1
        If resolution is different from 1, the template is re-sampled with the
        specified resolution.

    Returns
    -------
    wm_mni152_template : Nifti1Image, image representing the resampled
        white-matter template

    See Also
    --------
    nilearn.datasets.load_mni152_template : for details about version of the
        MNI152 T1 template.

    nilearn.datasets.load_mni152_gm_template : for details about version of the
        MNI152 grey-matter template.

    """

    resolution = resolution or 1

    wm_template = check_niimg(WM_MNI152_FILE_PATH)

    # Typecasting
    wm_data = get_data(wm_template).astype("float32")

    # Re-scale template from 0 to 1
    wm_data /= wm_data.max()
    new_wm_template = new_img_like(wm_template, wm_data)

    # Resample template according to the pre-specified resolution, if different
    # than 1
    if resolution != 1:
        new_wm_template = resampling.resample_img(new_wm_template,
                                                  np.eye(3) * resolution)

    return new_wm_template


def load_mni152_brain_mask(resolution=None, threshold=0.2):
    """Load the MNI152 whole-brain mask.
    This function takes the whole-brain MNI152 T1 template and threshold it,
    in order to obtain the corresponding whole-brain mask.

    .. versionadded:: 0.2.5

    Parameters
    ----------
    resolution: int, optional, Default = 1
        If resolution is different from 1, the template loaded is first
        re-sampled with the specified resolution.

        .. versionadded:: 0.8.1

    threshold : float, optional
        Values of the MNI152 T1 template above this threshold will be included.
        Default=0.2

    Returns
    -------
    mask_img : Nifti1Image, image corresponding to the whole-brain mask.

    Notes
    -----
    Refer to load_mni152_template function for more information about the
    MNI152 T1 template.

    See Also
    --------
    nilearn.datasets.load_mni152_template : for details about version of the
        MNI152 T1 template and related.

    """

    resolution = resolution or 1

    # Load MNI template
    target_img = load_mni152_template(resolution=resolution)
    mask_voxels = (get_data(target_img) > threshold).astype("int8")
    mask_img = new_img_like(target_img, mask_voxels)

    return mask_img


def load_mni152_gm_mask(resolution=None, threshold=0.2, n_iter=2):
    """Load the MNI152 grey-matter mask.
    This function takes the grey-matter MNI152 template and threshold it, in
    order to obtain the corresponding grey-matter mask.

    .. versionadded:: 0.8.1

    Parameters
    ----------
    resolution: int, optional, Default = 1
        If resolution is different from 1, the template loaded is first
        re-sampled with the specified resolution.

    threshold : float, optional
        Values of the grey-matter MNI152 template above this threshold will be
        included. Default=0.2

    n_iter: int, optional, Default = 2
        Number of repetitions of dilation and erosion steps performed in
        scipy.ndimage.binary_closing function.

    Returns
    -------
    gm_mask_img : Nifti1Image, image corresponding to the grey-matter mask.

    Notes
    -----
    Refer to load_mni152_gm_template function for more information about the
    MNI152 grey-matter template.

    See Also
    --------
    nilearn.datasets.load_mni152_gm_template : for details about version of the
        MNI152 grey-matter template and related.

    """

    resolution = resolution or 1

    # Load MNI template
    gm_target = load_mni152_gm_template(resolution=resolution)
    gm_target_img = check_niimg(gm_target)
    gm_target_data = get_data(gm_target_img)

    gm_target_mask = (gm_target_data > threshold).astype("int8")

    gm_target_mask = binary_closing(gm_target_mask, iterations=n_iter)
    gm_mask_img = new_img_like(gm_target_img, gm_target_mask)

    return gm_mask_img


def load_mni152_wm_mask(resolution=None, threshold=0.2, n_iter=2):
    """Load the MNI152 white-matter mask.
    This function takes the white-matter MNI152 template and threshold it, in
    order to obtain the corresponding white-matter mask.

    .. versionadded:: 0.8.1

    Parameters
    ----------
    resolution: int, optional, Default = 1
        If resolution is different from 1, the template loaded is first
        re-sampled with the specified resolution.

    threshold : float, optional
        Values of the white-matter MNI152 template above this threshold will be
        included. Default=0.2

    n_iter: int, optional, Default = 2
        Number of repetitions of dilation and erosion steps performed in
        scipy.ndimage.binary_closing function.

    Returns
    -------
    wm_mask_img : Nifti1Image, image corresponding to the white-matter mask.

    Notes
    -----
    Refer to load_mni152_gm_template function for more information about the
    MNI152 white-matter template.

    See Also
    --------
    nilearn.datasets.load_mni152_wm_template : for details about version of the
        MNI152 white-matter template and related.

    """

    resolution = resolution or 1

    # Load MNI template
    wm_target = load_mni152_wm_template(resolution=resolution)
    wm_target_img = check_niimg(wm_target)
    wm_target_data = get_data(wm_target_img)

    wm_target_mask = (wm_target_data > threshold).astype("int8")

    wm_target_mask = binary_closing(wm_target_mask, iterations=n_iter)
    wm_mask_img = new_img_like(wm_target_img, wm_target_mask)

    return wm_mask_img


@fill_doc
def fetch_icbm152_brain_gm_mask(data_dir=None, threshold=0.2, resume=True,
                                n_iter=2, verbose=1):
    """Downloads ICBM152 template first, then loads the 'gm' mask.

     %(templateflow)s

    .. versionadded:: 0.2.5

    Parameters
    ----------
    %(data_dir)s
    threshold : float, optional
        Values of the ICBM152 grey-matter template above this threshold will be
        included. Default=0.2

    %(resume)s
    n_iter: int, optional, Default = 2
        Number of repetitions of dilation and erosion steps performed in
        scipy.ndimage.binary_closing function.

        .. versionadded:: 0.8.1

    %(verbose)s

    Returns
    -------
    gm_mask_img : Nifti1Image, image corresponding to the brain grey matter
        from ICBM152 template.

    Notes
    -----
    This function relies on ICBM152 templates where we particularly pick
    grey matter template and threshold the template at .2 to take one fifth
    of the values. Then, do a bit post processing such as binary closing
    operation to more compact mask image.

    .. note::
        It is advised to check the mask image with your own data processing.

    See Also
    --------
    nilearn.datasets.fetch_icbm152_2009: for details regarding the ICBM152
        template.

    nilearn.datasets.load_mni152_template: for details about version of MNI152
        template and related.

    """
    # Fetching ICBM152 grey matter mask image
    icbm = fetch_icbm152_2009(data_dir=data_dir, resume=resume,
                              verbose=verbose)
    gm = icbm['gm']
    gm_img = check_niimg(gm)
    gm_data = get_data(gm_img)

    # getting one fifth of the values
    gm_mask = (gm_data > threshold).astype("int8")

    gm_mask = binary_closing(gm_mask, iterations=n_iter)
    gm_mask_img = new_img_like(gm_img, gm_mask)

    return gm_mask_img


@fill_doc
def fetch_oasis_vbm(n_subjects=None, dartel_version=True, data_dir=None,
                    url=None, resume=True, verbose=1, legacy_format=True):
    """Download and load Oasis "cross-sectional MRI" dataset (416 subjects).

    For more information, see :footcite:`OASISbrain`,
    and :footcite:`Marcus2007`.

    Parameters
    ----------
    n_subjects : int, optional
        The number of subjects to load. If None is given, all the
        subjects are used.

    dartel_version : boolean, optional
        Whether or not to use data normalized with DARTEL instead of standard
        SPM8 normalization. Default=True.
    %(data_dir)s
    %(url)s
    %(resume)s
    %(verbose)s
    %(legacy_format)s

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :

        - 'gray_matter_maps': string list
          Paths to nifti gray matter density probability maps
        - 'white_matter_maps' string list
          Paths to nifti white matter density probability maps
        - 'ext_vars': np.recarray
          Data from the .csv file with information about selected subjects
        - 'data_usage_agreement': string
          Path to the .txt file containing the data usage agreement.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    In the DARTEL version, original Oasis data have been preprocessed
    with the following steps:

      1. Dimension swapping (technically required for subsequent steps)
      2. Brain Extraction
      3. Segmentation with SPM8
      4. Normalization using DARTEL algorithm
      5. Modulation
      6. Replacement of NaN values with 0 in gray/white matter density maps.
      7. Resampling to reduce shape and make it correspond to the shape of
         the non-DARTEL data (fetched with dartel_version=False).
      8. Replacement of values < 1e-4 with zeros to reduce the file size.

    In the non-DARTEL version, the following steps have been performed instead:

      1. Dimension swapping (technically required for subsequent steps)
      2. Brain Extraction
      3. Segmentation and normalization to a template with SPM8
      4. Modulation
      5. Replacement of NaN values with 0 in gray/white matter density maps.

    An archive containing the gray and white matter density probability maps
    for the 416 available subjects is provided. Gross outliers are removed and
    filtered by this data fetcher (DARTEL: 13 outliers; non-DARTEL: 1 outlier)
    Externals variates (age, gender, estimated intracranial volume,
    years of education, socioeconomic status, dementia score) are provided
    in a CSV file that is a copy of the original Oasis CSV file. The current
    downloader loads the CSV file and keeps only the lines corresponding to
    the subjects that are actually demanded.

    The Open Access Structural Imaging Series (OASIS) is a project
    dedicated to making brain imaging data openly available to the public.
    Using data available through the OASIS project requires agreeing with
    the Data Usage Agreement that can be found at
    http://www.oasis-brains.org/app/template/UsageAgreement.vm

    """
    # check number of subjects
    if n_subjects is None:
        n_subjects = 403 if dartel_version else 415
    if dartel_version:  # DARTEL version has 13 identified outliers
        if n_subjects > 403:
            warnings.warn('Only 403 subjects are available in the '
                          'DARTEL-normalized version of the dataset. '
                          'All of them will be used instead of the wanted %d'
                          % n_subjects)
            n_subjects = 403
    else:  # all subjects except one are available with non-DARTEL version
        if n_subjects > 415:
            warnings.warn('Only 415 subjects are available in the '
                          'non-DARTEL-normalized version of the dataset. '
                          'All of them will be used instead of the wanted %d'
                          % n_subjects)
            n_subjects = 415
    if n_subjects < 1:
        raise ValueError("Incorrect number of subjects (%d)" % n_subjects)

    # pick the archive corresponding to preprocessings type
    if url is None:
        if dartel_version:
            url_images = ('https://www.nitrc.org/frs/download.php/'
                          '6364/archive_dartel.tgz?i_agree=1&download_now=1')
        else:
            url_images = ('https://www.nitrc.org/frs/download.php/'
                          '6359/archive.tgz?i_agree=1&download_now=1')
        # covariates and license are in separate files on NITRC
        url_csv = ('https://www.nitrc.org/frs/download.php/'
                   '6348/oasis_cross-sectional.csv?i_agree=1&download_now=1')
        url_dua = ('https://www.nitrc.org/frs/download.php/'
                   '6349/data_usage_agreement.txt?i_agree=1&download_now=1')
    else:  # local URL used in tests
        url_csv = url + "/oasis_cross-sectional.csv"
        url_dua = url + "/data_usage_agreement.txt"
        if dartel_version:
            url_images = url + "/archive_dartel.tgz"
        else:
            url_images = url + "/archive.tgz"

    opts = {'uncompress': True}

    # missing subjects create shifts in subjects ids
    missing_subjects = [8, 24, 36, 48, 89, 93, 100, 118, 128, 149, 154,
                        171, 172, 175, 187, 194, 196, 215, 219, 225, 242,
                        245, 248, 251, 252, 257, 276, 297, 306, 320, 324,
                        334, 347, 360, 364, 391, 393, 412, 414, 427, 436]

    if dartel_version:
        # DARTEL produces outliers that are hidden by nilearn API
        removed_outliers = [27, 57, 66, 83, 122, 157, 222, 269, 282, 287,
                            309, 428]
        missing_subjects = sorted(missing_subjects + removed_outliers)
        file_names_gm = [
            (os.path.join(
                    "OAS1_%04d_MR1",
                    "mwrc1OAS1_%04d_MR1_mpr_anon_fslswapdim_bet.nii.gz")
             % (s, s),
             url_images, opts)
            for s in range(1, 457) if s not in missing_subjects][:n_subjects]
        file_names_wm = [
            (os.path.join(
                    "OAS1_%04d_MR1",
                    "mwrc2OAS1_%04d_MR1_mpr_anon_fslswapdim_bet.nii.gz")
             % (s, s),
             url_images, opts)
            for s in range(1, 457) if s not in missing_subjects]
    else:
        # only one gross outlier produced, hidden by nilearn API
        removed_outliers = [390]
        missing_subjects = sorted(missing_subjects + removed_outliers)
        file_names_gm = [
            (os.path.join(
                    "OAS1_%04d_MR1",
                    "mwc1OAS1_%04d_MR1_mpr_anon_fslswapdim_bet.nii.gz")
             % (s, s),
             url_images, opts)
            for s in range(1, 457) if s not in missing_subjects][:n_subjects]
        file_names_wm = [
            (os.path.join(
                    "OAS1_%04d_MR1",
                    "mwc2OAS1_%04d_MR1_mpr_anon_fslswapdim_bet.nii.gz")
             % (s, s),
             url_images, opts)
            for s in range(1, 457) if s not in missing_subjects]
    file_names_extvars = [("oasis_cross-sectional.csv", url_csv, {})]
    file_names_dua = [("data_usage_agreement.txt", url_dua, {})]
    # restrict to user-specified number of subjects
    file_names_gm = file_names_gm[:n_subjects]
    file_names_wm = file_names_wm[:n_subjects]

    file_names = (file_names_gm + file_names_wm +
                  file_names_extvars + file_names_dua)
    dataset_name = 'oasis1'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, file_names, resume=resume,
                         verbose=verbose)

    # Build Bunch
    gm_maps = files[:n_subjects]
    wm_maps = files[n_subjects:(2 * n_subjects)]
    ext_vars_file = files[-2]
    data_usage_agreement = files[-1]

    # Keep CSV information only for selected subjects
    csv_data = pd.read_csv(ext_vars_file)
    # Comparisons to recfromcsv data must be bytes.
    actual_subjects_ids = [("OAS1" +
                            str.split(os.path.basename(x),
                                      "OAS1")[1][:9])
                           for x in gm_maps]
    subject_mask = np.asarray([subject_id in actual_subjects_ids
                               for subject_id in csv_data['ID']])
    csv_data = csv_data[subject_mask]
    csv_data = csv_data.rename(
        columns={c: c.lower().replace("/", "") for c in csv_data.columns}
    )
    fdescr = _get_dataset_descr(dataset_name)

    if legacy_format:
        warnings.warn(_LEGACY_FORMAT_MSG)
        csv_data = csv_data.to_records(index=False)

    return Bunch(
        gray_matter_maps=gm_maps,
        white_matter_maps=wm_maps,
        ext_vars=csv_data,
        data_usage_agreement=data_usage_agreement,
        description=fdescr)


@fill_doc
def fetch_surf_fsaverage(mesh='fsaverage5', data_dir=None):
    """Download a Freesurfer fsaverage surface.
    File names are subject to change and only attribute names
    are guaranteed to be stable across nilearn versions.
    See :footcite:`Fischl1999`.

    Parameters
    ----------
    mesh : str, optional
        Which mesh to fetch. Should be one of the following values:
        %(fsaverage_options)s
        Default='fsaverage5'.
    %(data_dir)s

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
         - 'area_left': Gifti file, left hemisphere area data
         - 'area_right': Gifti file, right hemisphere area data
         - 'curv_left': Gifti file, left hemisphere curvature data
         - 'curv_right': Gifti file, right hemisphere curvature data
         - 'flat_left': Gifti file, left hemisphere flat surface mesh
         - 'flat_right': Gifti file, right hemisphere flat surface mesh
         - 'pial_left': Gifti file, left hemisphere pial surface mesh
         - 'pial_right': Gifti file, right hemisphere pial surface mesh
         - 'infl_left': Gifti file, left hemisphere inflated pial surface mesh
         - 'infl_right': Gifti file, right hemisphere inflated pial
                         surface mesh
         - 'sphere_left': Gifti file, left hemisphere sphere surface mesh
         - 'sphere_right': Gifti file, right hemisphere sphere surface mesh
         - 'sulc_left': Gifti file, left hemisphere sulcal depth data
         - 'sulc_right': Gifti file, right hemisphere sulcal depth data
         - 'thick_left': Gifti file, left hemisphere cortical thickness data
         - 'thick_right': Gifti file, right hemisphere cortical thickness data
         - 'white_left': Gifti file, left hemisphere white surface mesh
         - 'white_right': Gifti file, right hemisphere white surface mesh

    References
    ----------
    .. footbibliography::

    """
    available_meshes = (
        "fsaverage3", "fsaverage4", "fsaverage5",
        "fsaverage6", "fsaverage7", "fsaverage",
    )

    # Call a dataset loader depending on the value of mesh
    if mesh in (
        "fsaverage3", "fsaverage4",
        "fsaverage6", "fsaverage7", "fsaverage",
    ):
        # rename mesh to "fsaverage" to download it once
        # regardless of whether mesh equals "fsaverage" or "fsaverage7"
        if mesh == "fsaverage7":
            mesh = "fsaverage"

        return _fetch_surf_fsaverage(mesh, data_dir=data_dir)
    elif mesh == "fsaverage5":
        return _fetch_surf_fsaverage5()
    else:
        raise ValueError(
            "'mesh' should be one of {}; {!r} was provided".format(
                available_meshes, mesh
            )
        )


def _fetch_surf_fsaverage5():
    """Helper function to ship fsaverage5 surfaces and sulcal information
    with Nilearn.

    The source of the data is coming from nitrc based on this PR #1016.
    Manually downloaded gzipped and shipped with this function.

    Shipping is done with Nilearn based on issue #1705.

    """
    data_dir = Path(FSAVERAGE5_PATH)

    data = {
        "{}_{}".format(part, hemi): str(
            data_dir / "{}_{}.gii.gz".format(part, hemi)
        )
        for part in [
            "area", "curv", "flat", "infl", "pial",
            "sphere", "sulc", "thick", "white"
        ]
        for hemi in ["left", "right"]
    }
    data["description"] = _get_dataset_descr("fsaverage5")

    return Bunch(**data)


def _fetch_surf_fsaverage(dataset_name, data_dir=None):
    """Helper function to ship fsaverage{3,4,6,7} meshes.

    These meshes can be used for visualization purposes, but also to run
    cortical surface-based searchlight decoding.

    The source of the data is downloaded from OSF.
    """
    dataset_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)
    opts = {'uncompress': True}

    url = {
        "fsaverage3": "https://osf.io/azhdf/download",
        "fsaverage4": "https://osf.io/28uma/download",
        "fsaverage6": "https://osf.io/jzxyr/download",
        "fsaverage": "https://osf.io/svf8k/download",  # fsaverage7
    }[dataset_name]

    # List of attributes exposed by the dataset
    dataset_attributes = [
        "{}_{}".format(part, hemi)
        for part in [
            "area", "curv", "flat", "infl", "pial",
            "sphere", "sulc", "thick", "white"
        ]
        for hemi in ["left", "right"]
    ]

    # Note that the file names match the attribute's
    _fetch_files(
        dataset_dir,
        [
            ("{}.gii.gz".format(attribute), url, opts)
            for attribute in dataset_attributes
        ]
    )

    result = {
        attribute: os.path.join(dataset_dir, "{}.gii.gz".format(attribute))
        for attribute in dataset_attributes
    }
    result["description"] = str(_get_dataset_descr(dataset_name))

    return Bunch(**result)
