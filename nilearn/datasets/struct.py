"""
Downloading NeuroImaging datasets: structural datasets
"""
import warnings
import os
from pathlib import Path

import numpy as np
from scipy import ndimage
from sklearn.utils import Bunch

from .utils import (_get_dataset_dir, _fetch_files,
                    _get_dataset_descr, _uncompress_file)

from .._utils import check_niimg, niimg
from ..image import new_img_like, get_data

_package_directory = os.path.dirname(os.path.abspath(__file__))
# Useful for the very simple examples
MNI152_FILE_PATH = os.path.join(_package_directory, "data",
                                "avg152T1_brain.nii.gz")
FSAVERAGE5_PATH = os.path.join(_package_directory, "data", "fsaverage5")


def fetch_icbm152_2009(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the ICBM152 template (dated 2009).

    For more information, see :footcite:`FONOV2011313`,
    :footcite:`Fonov2009`, and :footcite:`Collins1999algorithm`.

    Parameters
    ----------
    data_dir : string, optional
        Path of the data directory. Used to force data storage in a non-
        standard location. Default: None (meaning: default)

    url : string, optional
        Download URL of the dataset. Overwrite the default URL.

    resume : bool, optional
        If True, try resuming partially downloaded data.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, interest keys are:
        "t1", "t2", "t2_relax", "pd": anatomical images obtained with the
        given modality (resp. T1, T2, T2 relaxometry and proton
        density weighted). Values are file paths.
        "gm", "wm", "csf": segmented images, giving resp. gray matter,
        white matter and cerebrospinal fluid. Values are file paths.
        "eye_mask", "face_mask", "mask": use these images to mask out
        parts of mri images. Values are file paths.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    For more information about this dataset's structure:
    http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009

    The original download URL is
    http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip

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


def load_mni152_template():
    """Load skullstripped 2mm version of the MNI152 originally distributed
    with FSL.

    For more information, see :footcite:`FONOV2011313`,
    and :footcite:`Fonov2009`.

    Returns
    -------
    mni152_template : nibabel object corresponding to the template

    References
    ----------
    .. footbibliography::

    """
    return check_niimg(MNI152_FILE_PATH)


def load_mni152_brain_mask():
    """Load brain mask from MNI152 T1 template

    .. versionadded:: 0.2.5

    Returns
    -------
    mask_img : Nifti-like mask image corresponding to grey and white matter.

    Notes
    -----
    Refer to load_mni152_template function for more information about the MNI152
    T1 template

    See Also
    --------
    nilearn.datasets.load_mni152_template : for details about version of the
        MNI152 T1 template and related.

    """
    # Load MNI template
    target_img = load_mni152_template()
    mask_voxels = (get_data(target_img) > 0).astype(int)
    mask_img = new_img_like(target_img, mask_voxels)
    return mask_img


def fetch_icbm152_brain_gm_mask(data_dir=None, threshold=0.2, resume=True,
                                verbose=1):
    """Downloads ICBM152 template first, then loads 'gm' mask image.

    .. versionadded:: 0.2.5

    Parameters
    ----------
    data_dir : str, optional
        Path of the data directory. Used to force storage in a specified
        location. Defaults to None.

    threshold : float, optional
        The parameter which amounts to include the values in the mask image.
        The values lies above than this threshold will be included. Defaults
        to 0.2 (one fifth) of values.
        Default=0.2.

    resume : bool, optional
        If True, try resuming partially downloaded data.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    gm_mask_img : Nifti image
        Corresponding to brain grey matter from ICBM152 template.

    Notes
    -----
    This function relies on ICBM152 templates where we particularly pick
    grey matter template and threshold the template at .2 to take one fifth
    of the values. Then, do a bit post processing such as binary closing
    operation to more compact mask image.

    Note: It is advised to check the mask image with your own data processing.

    See Also
    --------
    nilearn.datasets.fetch_icbm152_2009: for details regarding the ICBM152
        template.

    nilearn.datasets.load_mni152_template: for details about version of MNI152
        template and related.

    """
    # Fetching ICBM152 grey matter mask image
    icbm = fetch_icbm152_2009(data_dir=data_dir, resume=resume, verbose=verbose)
    gm = icbm['gm']
    gm_img = check_niimg(gm)
    gm_data = niimg._safe_get_data(gm_img)

    # getting one fifth of the values
    gm_mask = (gm_data > threshold)

    gm_mask = ndimage.binary_closing(gm_mask, iterations=2)
    gm_mask_img = new_img_like(gm_img, gm_mask)
    return gm_mask_img


def fetch_oasis_vbm(n_subjects=None, dartel_version=True, data_dir=None,
                    url=None, resume=True, verbose=1):
    """Download and load Oasis "cross-sectional MRI" dataset (416 subjects).

    For more information, see :footcite:`OASISbrain`,
    and :footcite:`Marcus2007OASIS`.

    Parameters
    ----------
    n_subjects : int, optional
        The number of subjects to load. If None is given, all the
        subjects are used.

    dartel_version : boolean, optional
        Whether or not to use data normalized with DARTEL instead of standard
        SPM8 normalization. Default=True.

    data_dir : string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url : string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    resume : bool, optional
        If true, try resuming download if possible. Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

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


def fetch_surf_fsaverage(mesh='fsaverage5', data_dir=None):
    """Download a Freesurfer fsaverage surface.
    File names are subject to change and only attribute names
    are guaranteed to be stable across nilearn versions.
    See :footcite:`Fischl1999neurons`.

    Parameters
    ----------
    mesh : str, optional
        Which mesh to fetch. Default='fsaverage5'.

        - 'fsaverage3': the low-resolution fsaverage3 mesh (642 nodes)
        - 'fsaverage4': the low-resolution fsaverage4 mesh (2562 nodes)
        - 'fsaverage5': the low-resolution fsaverage5 mesh (10242 nodes)
        - 'fsaverage5_sphere': this option has been deprecated
            and will be removed in v0.9.0.
            fsaverage5 sphere coordinates can now be accessed through
            attributes sphere_{left, right} using mesh='fsaverage5'
        - 'fsaverage6': the medium-resolution fsaverage6 mesh (40962 nodes)
        - 'fsaverage7': same as 'fsaverage'
        - 'fsaverage': the high-resolution fsaverage mesh (163842 nodes)
            (high-resolution fsaverage will result in more computation time and memory usage)

    data_dir : str, optional
        Path of the data directory. Used to force data storage in a specified
        location.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :
         - 'area_left': Gifti file, left hemisphere area data
         - 'area_right': Gifti file, right hemisphere area data
         - 'curv_left': Gifti file, left hemisphere curvature data
         - 'curv_right': Gifti file, right hemisphere curvature data
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
    available_meshes = [
        "fsaverage3", "fsaverage4", "fsaverage5", "fsaverage5_sphere",
        "fsaverage6", "fsaverage7", "fsaverage",
    ]

    # Call a dataset loader depending on the value of mesh
    if (
        mesh == "fsaverage3"
        or mesh == "fsaverage4"
        or mesh == "fsaverage6"
        or mesh == "fsaverage7"
        or mesh == "fsaverage"
    ):
        # rename mesh to "fsaverage" to download it once
        # regardless of whether mesh equals "fsaverage" or "fsaverage7"
        if mesh == "fsaverage7":
            mesh = "fsaverage"

        return _fetch_surf_fsaverage(mesh, data_dir=data_dir)
    elif mesh == "fsaverage5":
        return _fetch_surf_fsaverage5()
    elif mesh == "fsaverage5_sphere":
        warnings.warn(
            "mesh='fsaverage5_sphere' has been deprecated "
            "and will be removed in v0.9.0.\n"
            "fsaverage5 sphere coordinates can now be accessed through "
            "attributes sphere_{left, right} using mesh='fsaverage5'"
        )
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
            "area", "curv", "infl", "pial",
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
        "fsaverage3": "https://osf.io/asvjk/download",
        "fsaverage4": "https://osf.io/x2j49/download",
        "fsaverage6": "https://osf.io/um5ag/download",
        "fsaverage": "https://osf.io/q7a5k/download",  # fsaverage7
    }[dataset_name]

    # List of attributes exposed by the dataset
    dataset_attributes = [
        "{}_{}".format(part, hemi)
        for part in [
            "area", "curv", "infl", "pial",
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
