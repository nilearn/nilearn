"""
Downloading NeuroImaging datasets: structural datasets
"""
import warnings
import os
import numpy as np
from sklearn.datasets.base import Bunch

from .utils import _get_dataset_dir, _fetch_files, _get_dataset_descr

from .._utils import check_niimg


def fetch_icbm152_2009(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the ICBM152 template (dated 2009)

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a non-
        standard location. Default: None (meaning: default)
    url: string, optional
        Download URL of the dataset. Overwrite the default URL.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, interest keys are:
        "t1", "t2", "t2_relax", "pd": anatomical images obtained with the
        given modality (resp. T1, T2, T2 relaxometry and proton
        density weighted). Values are file paths.
        "gm", "wm", "csf": segmented images, giving resp. gray matter,
        white matter and cerebrospinal fluid. Values are file paths.
        "eye_mask", "face_mask", "mask": use these images to mask out
        parts of mri images. Values are file paths.

    References
    ----------
    VS Fonov, AC Evans, K Botteron, CR Almli, RC McKinstry, DL Collins
    and BDCG, "Unbiased average age-appropriate atlases for pediatric studies",
    NeuroImage,Volume 54, Issue 1, January 2011

    VS Fonov, AC Evans, RC McKinstry, CR Almli and DL Collins,
    "Unbiased nonlinear average age-appropriate brain templates from birth
    to adulthood", NeuroImage, Volume 47, Supplement 1, July 2009, Page S102
    Organization for Human Brain Mapping 2009 Annual Meeting.

    DL Collins, AP Zijdenbos, WFC Baare and AC Evans,
    "ANIMAL+INSECT: Improved Cortical Structure Segmentation",
    IPMI Lecture Notes in Computer Science, 1999, Volume 1613/1999, 210-223

    Notes
    -----
    For more information about this dataset's structure:
    http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009
    """
    if url is None:
        url = "http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/" \
              "mni_icbm152_nlin_sym_09a_nifti.zip"
    opts = {'uncompress': True}

    keys = ("csf", "gm", "wm",
            "pd", "t1", "t2", "t2_relax",
            "eye_mask", "face_mask", "mask")
    filenames = [(os.path.join("mni_icbm152_nlin_sym_09a", name), url, opts)
                 for name in ("mni_icbm152_csf_tal_nlin_sym_09a.nii",
                              "mni_icbm152_gm_tal_nlin_sym_09a.nii",
                              "mni_icbm152_wm_tal_nlin_sym_09a.nii",

                              "mni_icbm152_pd_tal_nlin_sym_09a.nii",
                              "mni_icbm152_t1_tal_nlin_sym_09a.nii",
                              "mni_icbm152_t2_tal_nlin_sym_09a.nii",
                              "mni_icbm152_t2_relx_tal_nlin_sym_09a.nii",

                              "mni_icbm152_t1_tal_nlin_sym_09a_eye_mask.nii",
                              "mni_icbm152_t1_tal_nlin_sym_09a_face_mask.nii",
                              "mni_icbm152_t1_tal_nlin_sym_09a_mask.nii")]

    dataset_name = 'icbm152_2009'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    sub_files = _fetch_files(data_dir, filenames, resume=resume,
                             verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)

    params = dict([('description', fdescr)] + list(zip(keys, sub_files)))
    return Bunch(**params)


def load_mni152_template():
    """Load skullstripped 2mm version of the MNI152 originally distributed
    with FSL

    Returns
    -------
    mni152_template: nibabel object corresponding to the template


    References
    ----------

    VS Fonov, AC Evans, K Botteron, CR Almli, RC McKinstry, DL Collins and
    BDCG, Unbiased average age-appropriate atlases for pediatric studies,
    NeuroImage, Volume 54, Issue 1, January 2011, ISSN 1053-8119, DOI:
    10.1016/j.neuroimage.2010.07.033

    VS Fonov, AC Evans, RC McKinstry, CR Almli and DL Collins, Unbiased
    nonlinear average age-appropriate brain templates from birth to adulthood,
    NeuroImage, Volume 47, Supplement 1, July 2009, Page S102 Organization for
    Human Brain Mapping 2009 Annual Meeting, DOI: 10.1016/S1053-8119(09)70884-5

    """
    package_directory = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(package_directory, "data", "avg152T1_brain.nii.gz")

    # XXX Should we load the image here?
    return check_niimg(path)


def fetch_oasis_vbm(n_subjects=None, dartel_version=True, data_dir=None,
                    url=None, resume=True, verbose=1):
    """Download and load Oasis "cross-sectional MRI" dataset (416 subjects).

    Parameters
    ----------
    n_subjects: int, optional
        The number of subjects to load. If None is given, all the
        subjects are used.

    dartel_version: boolean,
        Whether or not to use data normalized with DARTEL instead of standard
        SPM8 normalization.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    resume: bool, optional
        If true, try resuming download if possible

    verbose: int, optional
        verbosity level (0 means no message).

    Returns
    -------
    data: Bunch
        Dictionary-like object, the interest attributes are :
        'gray_matter_maps': string list
            Paths to nifti gray matter density probability maps
        'white_matter_maps' string list
            Paths to nifti white matter density probability maps
        'ext_vars': np.recarray
            Data from the .csv file with information about selected subjects
        'data_usage_agreement': string
            Path to the .txt file containing the data usage agreement.

    References
    ----------
    [1] http://www.oasis-brains.org/

    [2] Open Access Series of Imaging Studies (OASIS): Cross-sectional MRI
        Data in Young, Middle Aged, Nondemented, and Demented Older Adults.
        Marcus, D. S and al., 2007, Journal of Cognitive Neuroscience.

    Notes
    -----
    In the DARTEL version, original Oasis data [1] have been preprocessed
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
    csv_data = np.recfromcsv(ext_vars_file)
    # Comparisons to recfromcsv data must be bytes.
    actual_subjects_ids = [("OAS1" +
                            str.split(os.path.basename(x),
                                      "OAS1")[1][:9]).encode()
                           for x in gm_maps]
    subject_mask = np.asarray([subject_id in actual_subjects_ids
                               for subject_id in csv_data['id']])
    csv_data = csv_data[subject_mask]

    fdescr = _get_dataset_descr(dataset_name)

    return Bunch(
        gray_matter_maps=gm_maps,
        white_matter_maps=wm_maps,
        ext_vars=csv_data,
        data_usage_agreement=data_usage_agreement,
        description=fdescr)
