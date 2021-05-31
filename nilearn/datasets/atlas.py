"""
Downloading NeuroImaging datasets: atlas datasets
"""
import os
import warnings
import xml.etree.ElementTree
from tempfile import mkdtemp
import json
import shutil

import nibabel as nb
import numpy as np
from numpy.lib import recfunctions
from sklearn.utils import Bunch

from .utils import _get_dataset_dir, _fetch_files, _get_dataset_descr
from .._utils import check_niimg
from ..image import new_img_like, get_data

_TALAIRACH_LEVELS = ['hemisphere', 'lobe', 'gyrus', 'tissue', 'ba']


def fetch_atlas_difumo(dimension=64, resolution_mm=2, data_dir=None, resume=True, verbose=1):
    """Fetch DiFuMo brain atlas

    Dictionaries of Functional Modes, or “DiFuMo”, can serve as atlases to extract
    functional signals with different dimensionalities (64, 128, 256, 512, and 1024).
    These modes are optimized to represent well raw BOLD timeseries,
    over a with range of experimental conditions. 
    See :footcite:`DADI2020117126`.

    Notes
    -----
    Direct download links from OSF:

    - 64: https://osf.io/pqu9r/download
    - 128: https://osf.io/wjvd5/download
    - 256: https://osf.io/3vrct/download
    - 512: https://osf.io/9b76y/download
    - 1024: https://osf.io/34792/download

    Parameters
    ----------
    dimension : int, optional
        Number of dimensions in the dictionary. Valid resolutions
        available are {64, 128, 256, 512, 1024}.
        Default=64.

    resolution_mm : int, optional
        The resolution in mm of the atlas to fetch. Valid options
        available are {2, 3}. Default=2mm.

    data_dir : string, optional
        Path where data should be downloaded. By default,
        files are downloaded in home directory.

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :

        - 'maps': str, 4D path to nifti file containing regions definition.
        - 'labels': Numpy recarray containing the labels of the regions.
        - 'description': str, general description of the dataset.

    References
    ----------
    .. footbibliography::

    """
    dic = {64: 'pqu9r',
           128: 'wjvd5',
           256: '3vrct',
           512: '9b76y',
           1024: '34792',
           }
    valid_dimensions = [64, 128, 256, 512, 1024]
    valid_resolution_mm = [2, 3]
    if dimension not in valid_dimensions:
        raise ValueError("Requested dimension={} is not available. Valid "
                         "options: {}".format(dimension, valid_dimensions))
    if resolution_mm not in valid_resolution_mm:
        raise ValueError("Requested resolution_mm={} is not available. Valid "
                         "options: {}".format(resolution_mm,
                                              valid_resolution_mm))

    url = 'https://osf.io/{}/download'.format(dic[dimension])
    opts = {'uncompress': True}

    csv_file = os.path.join('{0}', 'labels_{0}_dictionary.csv')
    if resolution_mm != 3:
        nifti_file = os.path.join('{0}', '2mm', 'maps.nii.gz')
    else:
        nifti_file = os.path.join('{0}', '3mm', 'maps.nii.gz')

    files = [(csv_file.format(dimension), url, opts),
             (nifti_file.format(dimension), url, opts)]

    dataset_name = 'difumo_atlases'

    data_dir = _get_dataset_dir(dataset_name=dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # Download the zip file, first
    files_ = _fetch_files(data_dir, files, verbose=verbose)
    labels = np.recfromcsv(files_[0])

    # README
    readme_files = [('README.md', 'https://osf.io/4k9bf/download',
                    {'move': 'README.md'})]
    if not os.path.exists(os.path.join(data_dir, 'README.md')):
        _fetch_files(data_dir, readme_files, verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)

    params = dict(description=fdescr, maps=files_[1], labels=labels)

    return Bunch(**params)


def fetch_atlas_craddock_2012(data_dir=None, url=None, resume=True, verbose=1):
    """Download and return file names for the Craddock 2012 parcellation

    The provided images are in MNI152 space.

    See :footcite:`CreativeCommons` for the licence.

    See :footcite:`craddock2012whole` and :footcite:`nitrcClusterROI`
    for more information on this parcellation.

    Parameters
    ----------
    data_dir : string, optional
        Directory where data should be downloaded and unpacked.

    url : string, optional
        url of file to download.

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, keys are:
        scorr_mean, tcorr_mean,
        scorr_2level, tcorr_2level,
        random

    References
    ----------
    .. footbibliography::

    """
    if url is None:
        url = "ftp://www.nitrc.org/home/groups/cluster_roi/htdocs" \
              "/Parcellations/craddock_2011_parcellations.tar.gz"
    opts = {'uncompress': True}

    dataset_name = "craddock_2012"
    keys = ("scorr_mean", "tcorr_mean",
            "scorr_2level", "tcorr_2level",
            "random")
    filenames = [
            ("scorr05_mean_all.nii.gz", url, opts),
            ("tcorr05_mean_all.nii.gz", url, opts),
            ("scorr05_2level_all.nii.gz", url, opts),
            ("tcorr05_2level_all.nii.gz", url, opts),
            ("random_all.nii.gz", url, opts)
    ]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    sub_files = _fetch_files(data_dir, filenames, resume=resume,
                             verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)

    params = dict([('description', fdescr)] + list(zip(keys, sub_files)))

    return Bunch(**params)


def fetch_atlas_destrieux_2009(lateralized=True, data_dir=None, url=None,
                               resume=True, verbose=1):
    """Download and load the Destrieux cortical atlas (dated 2009)
    
    see :footcite:`Fischl2004Automatically`,
    and :footcite:`Destrieux2009sulcal`.

    Parameters
    ----------
    lateralized : boolean, optional
        If True, returns an atlas with distinct regions for right and left
        hemispheres. Default=True.

    data_dir : string, optional
        Path of the data directory. Use to forec data storage in a non-
        standard location. Default: None (meaning: default)

    url : string, optional
        Download URL of the dataset. Overwrite the default URL.

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, contains:

        - Cortical ROIs, lateralized or not (maps)
        - Labels of the ROIs (labels)

    References
    ----------
    .. footbibliography::

    """
    if url is None:
        url = "https://www.nitrc.org/frs/download.php/11942/"

    url += "destrieux2009.tgz"
    opts = {'uncompress': True}
    lat = '_lateralized' if lateralized else ''

    files = [
        ('destrieux2009_rois_labels' + lat + '.csv', url, opts),
        ('destrieux2009_rois' + lat + '.nii.gz', url, opts),
        ('destrieux2009.rst', url, opts)
    ]

    dataset_name = 'destrieux_2009'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files_ = _fetch_files(data_dir, files, resume=resume,
                          verbose=verbose)

    params = dict(maps=files_[1], labels=np.recfromcsv(files_[0]))

    with open(files_[2], 'r') as rst_file:
        params['description'] = rst_file.read()

    return Bunch(**params)


def fetch_atlas_harvard_oxford(atlas_name, data_dir=None,
                               symmetric_split=False,
                               resume=True, verbose=1):
    """Load Harvard-Oxford parcellations from FSL.

    This function downloads Harvard Oxford atlas packaged from FSL 5.0
    and stores atlases in NILEARN_DATA folder in home directory.

    This function can also load Harvard Oxford atlas from your local directory
    specified by your FSL installed path given in `data_dir` argument.
    See documentation for details.

    Parameters
    ----------
    atlas_name : string
        Name of atlas to load. Can be:
        cort-maxprob-thr0-1mm,  cort-maxprob-thr0-2mm,
        cort-maxprob-thr25-1mm, cort-maxprob-thr25-2mm,
        cort-maxprob-thr50-1mm, cort-maxprob-thr50-2mm,
        sub-maxprob-thr0-1mm,  sub-maxprob-thr0-2mm,
        sub-maxprob-thr25-1mm, sub-maxprob-thr25-2mm,
        sub-maxprob-thr50-1mm, sub-maxprob-thr50-2mm,
        cort-prob-1mm, cort-prob-2mm,
        sub-prob-1mm, sub-prob-2mm

    data_dir : string, optional
        Path of data directory where data will be stored. Optionally,
        it can also be a FSL installation directory (which is dependent
        on your installation).
        Example, if FSL is installed in /usr/share/fsl/ then
        specifying as '/usr/share/' can get you Harvard Oxford atlas
        from your installed directory. Since we mimic same root directory
        as FSL to load it easily from your installation.

    symmetric_split : bool, optional
        If True, lateralized atlases of cort or sub with maxprob will be
        returned. For subcortical types (sub-maxprob), we split every
        symmetric region in left and right parts. Effectively doubles the
        number of regions.
        NOTE Not implemented for full probabilistic atlas (*-prob-* atlases).
        Default=False.

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, keys are:

        - "maps": nibabel.Nifti1Image, 4D maps if a probabilistic atlas is
          requested and 3D labels if a maximum probabilistic atlas was
          requested.

        - "labels": string list, labels of the regions in the atlas.

    """
    atlas_items = ("cort-maxprob-thr0-1mm", "cort-maxprob-thr0-2mm",
                   "cort-maxprob-thr25-1mm", "cort-maxprob-thr25-2mm",
                   "cort-maxprob-thr50-1mm", "cort-maxprob-thr50-2mm",
                   "sub-maxprob-thr0-1mm", "sub-maxprob-thr0-2mm",
                   "sub-maxprob-thr25-1mm", "sub-maxprob-thr25-2mm",
                   "sub-maxprob-thr50-1mm", "sub-maxprob-thr50-2mm",
                   "cort-prob-1mm", "cort-prob-2mm",
                   "sub-prob-1mm", "sub-prob-2mm")
    if atlas_name not in atlas_items:
        raise ValueError("Invalid atlas name: {0}. Please chose an atlas "
                         "among:\n{1}".format(
                             atlas_name, '\n'.join(atlas_items)))

    url = 'http://www.nitrc.org/frs/download.php/9902/HarvardOxford.tgz'

    # For practical reasons, we mimic the FSL data directory here.
    dataset_name = 'fsl'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    opts = {'uncompress': True}
    root = os.path.join('data', 'atlases')

    if atlas_name[0] == 'c':
        if 'cort-maxprob' in atlas_name and symmetric_split:
            split_name = atlas_name.split('cort')
            atlas_name = 'cortl' + split_name[1]
            label_file = 'HarvardOxford-Cortical-Lateralized.xml'
            lateralized = True
        else:
            label_file = 'HarvardOxford-Cortical.xml'
            lateralized = False
    else:
        label_file = 'HarvardOxford-Subcortical.xml'
        lateralized = False
    label_file = os.path.join(root, label_file)

    atlas_file = os.path.join(root, 'HarvardOxford',
                              'HarvardOxford-' + atlas_name + '.nii.gz')

    atlas_img, label_file = _fetch_files(
        data_dir,
        [(atlas_file, url, opts), (label_file, url, opts)],
        resume=resume, verbose=verbose)

    names = {}
    from xml.etree import ElementTree
    names[0] = 'Background'
    for label in ElementTree.parse(label_file).findall('.//label'):
        names[int(label.get('index')) + 1] = label.text
    names = list(names.values())

    if not symmetric_split:
        return Bunch(maps=atlas_img, labels=names)

    if atlas_name in ("cort-prob-1mm", "cort-prob-2mm",
                      "sub-prob-1mm", "sub-prob-2mm"):
        raise ValueError("Region splitting not supported for probabilistic "
                         "atlases")

    atlas_img = check_niimg(atlas_img)
    if lateralized:
        return Bunch(maps=atlas_img, labels=names)

    atlas = get_data(atlas_img)

    labels = np.unique(atlas)
    # Build a mask of both halves of the brain
    middle_ind = (atlas.shape[0] - 1) // 2
    # Put zeros on the median plane
    atlas[middle_ind, ...] = 0
    # Split every zone crossing the median plane into two parts.
    left_atlas = atlas.copy()
    left_atlas[middle_ind:, ...] = 0
    right_atlas = atlas.copy()
    right_atlas[:middle_ind, ...] = 0

    new_label = 0
    new_atlas = atlas.copy()
    # Assumes that the background label is zero.
    new_names = [names[0]]
    for label, name in zip(labels[1:], names[1:]):
        new_label += 1
        left_elements = (left_atlas == label).sum()
        right_elements = (right_atlas == label).sum()
        n_elements = float(left_elements + right_elements)
        if (left_elements / n_elements < 0.05 or
                right_elements / n_elements < 0.05):
            new_atlas[atlas == label] = new_label
            new_names.append(name)
            continue
        new_atlas[right_atlas == label] = new_label
        new_names.append(name + ', left part')
        new_label += 1
        new_atlas[left_atlas == label] = new_label
        new_names.append(name + ', right part')

    atlas_img = new_img_like(atlas_img, new_atlas, atlas_img.affine)
    return Bunch(maps=atlas_img, labels=new_names)


def fetch_atlas_msdl(data_dir=None, url=None, resume=True, verbose=1):
    """Download and load the MSDL brain atlas.

    It can be downloaded at :footcite:`atlas_msdl`, and cited
    using :footcite:`Varoquaux2011multisubject`.
    See also :footcite:`VAROQUAUX2013405` for more information.

    Parameters
    ----------
    data_dir : string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url : string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :

        - 'maps': str, path to nifti file containing regions definition.
        - 'labels': string list containing the labels of the regions.
        - 'region_coords': tuple list (x, y, z) containing coordinates
          of each region in MNI space.
        - 'networks': string list containing names of the networks.
        - 'description': description about the atlas.

    References
    ----------
    .. footbibliography::


    """
    url = 'https://team.inria.fr/parietal/files/2015/01/MSDL_rois.zip'
    opts = {'uncompress': True}

    dataset_name = "msdl_atlas"
    files = [(os.path.join('MSDL_rois', 'msdl_rois_labels.csv'), url, opts),
             (os.path.join('MSDL_rois', 'msdl_rois.nii'), url, opts)]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files = _fetch_files(data_dir, files, resume=resume, verbose=verbose)
    csv_data = np.recfromcsv(files[0])
    labels = [name.strip() for name in csv_data['name'].tolist()]
    labels = [label.decode("utf-8") for label in labels]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', module='numpy',
                                category=FutureWarning)
        region_coords = csv_data[['x', 'y', 'z']].tolist()
    net_names = [net_name.strip() for net_name in csv_data['net_name'].tolist()]
    fdescr = _get_dataset_descr(dataset_name)

    return Bunch(maps=files[1], labels=labels, region_coords=region_coords,
                 networks=net_names, description=fdescr)


def fetch_coords_power_2011():
    """Download and load the Power et al. brain atlas composed of 264 ROIs
    
    See :footcite:`Power2011Functional`.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, contains:
        - "rois": coordinates of 264 ROIs in MNI space


    References
    ----------
    .. footbibliography::

    """
    dataset_name = 'power_2011'
    fdescr = _get_dataset_descr(dataset_name)
    package_directory = os.path.dirname(os.path.abspath(__file__))
    csv = os.path.join(package_directory, "data", "power_2011.csv")
    params = dict(rois=np.recfromcsv(csv), description=fdescr)

    return Bunch(**params)


def fetch_atlas_smith_2009(data_dir=None, mirror='origin', url=None,
                           resume=True, verbose=1):
    """Download and load the Smith ICA and BrainMap atlas (dated 2009).

    See :footcite:`Smith200913040` and :footcite:`Laird2011behavioral`.

    Parameters
    ----------
    data_dir : string, optional
        Path of the data directory. Used to force data storage in a non-
        standard location. Default: None (meaning: default)

    mirror : string, optional
        By default, the dataset is downloaded from the original website of the
        atlas. Specifying "nitrc" will force download from a mirror, with
        potentially higher bandwith. Default='origin'.

    url : string, optional
        Download URL of the dataset. Overwrite the default URL.

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, contains:

        - 20-dimensional ICA, Resting-FMRI components:

          - all 20 components (rsn20)
          - 10 well-matched maps from these, as shown in PNAS paper (rsn10)

        - 20-dimensional ICA, BrainMap components:

          - all 20 components (bm20)
          - 10 well-matched maps from these, as shown in PNAS paper (bm10)

        - 70-dimensional ICA, Resting-FMRI components (rsn70)

        - 70-dimensional ICA, BrainMap components (bm70)

    References
    ----------
    .. footbibliography::

    Notes
    -----
    For more information about this dataset's structure:
    http://www.fmrib.ox.ac.uk/datasets/brainmap+rsns/

    """
    if url is None:
        if mirror == 'origin':
            url = "http://www.fmrib.ox.ac.uk/datasets/brainmap+rsns/"
        elif mirror == 'nitrc':
            url = [
                    'https://www.nitrc.org/frs/download.php/7730/',
                    'https://www.nitrc.org/frs/download.php/7729/',
                    'https://www.nitrc.org/frs/download.php/7731/',
                    'https://www.nitrc.org/frs/download.php/7726/',
                    'https://www.nitrc.org/frs/download.php/7728/',
                    'https://www.nitrc.org/frs/download.php/7727/',
            ]
        else:
            raise ValueError('Unknown mirror "%s". Mirror must be "origin" '
                'or "nitrc"' % str(mirror))

    files = [
            'rsn20.nii.gz',
            'PNAS_Smith09_rsn10.nii.gz',
            'rsn70.nii.gz',
            'bm20.nii.gz',
            'PNAS_Smith09_bm10.nii.gz',
            'bm70.nii.gz'
    ]

    if isinstance(url, str):
        url = [url] * len(files)

    files = [(f, u + f, {}) for f, u in zip(files, url)]

    dataset_name = 'smith_2009'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    files_ = _fetch_files(data_dir, files, resume=resume,
                          verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)

    keys = ['rsn20', 'rsn10', 'rsn70', 'bm20', 'bm10', 'bm70']
    params = dict(zip(keys, files_))
    params['description'] = fdescr

    return Bunch(**params)


def fetch_atlas_yeo_2011(data_dir=None, url=None, resume=True, verbose=1):
    """Download and return file names for the Yeo 2011 parcellation.

    The provided images are in MNI152 space.

    For more information on this dataset's structure,
    see :footcite:`CorticalParcellation_Yeo2011`,
    and :footcite:`Yeo2011organization`.

    Parameters
    ----------
    data_dir : string, optional
        Directory where data should be downloaded and unpacked.

    url : string, optional
        Url of file to download.

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, keys are:

        - "thin_7", "thick_7": 7-region parcellations,
          fitted to resp. thin and thick template cortex segmentations.

        - "thin_17", "thick_17": 17-region parcellations.

        - "colors_7", "colors_17": colormaps (text files) for 7- and 17-region
          parcellation respectively.

        - "anat": anatomy image.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Licence: unknown.

    """
    if url is None:
        url = ('ftp://surfer.nmr.mgh.harvard.edu/pub/data/'
               'Yeo_JNeurophysiol11_MNI152.zip')
    opts = {'uncompress': True}

    dataset_name = "yeo_2011"
    keys = ("thin_7", "thick_7",
            "thin_17", "thick_17",
            "colors_7", "colors_17", "anat")
    basenames = (
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_7Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm.nii.gz",
        "Yeo2011_17Networks_MNI152_FreeSurferConformed1mm_LiberalMask.nii.gz",
        "Yeo2011_7Networks_ColorLUT.txt",
        "Yeo2011_17Networks_ColorLUT.txt",
        "FSL_MNI152_FreeSurferConformed_1mm.nii.gz")

    filenames = [(os.path.join("Yeo_JNeurophysiol11_MNI152", f), url, opts)
                 for f in basenames]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
            verbose=verbose)
    sub_files = _fetch_files(data_dir, filenames, resume=resume,
                             verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)

    params = dict([('description', fdescr)] + list(zip(keys, sub_files)))
    return Bunch(**params)


def fetch_atlas_aal(version='SPM12', data_dir=None, url=None, resume=True,
                    verbose=1):
    """Downloads and returns the AAL template for SPM 12.

    This atlas is the result of an automated anatomical parcellation of the
    spatially normalized single-subject high-resolution T1 volume provided by
    the Montreal Neurological Institute (MNI) (D. L. Collins et al., 1998,
    Trans. Med. Imag. 17, 463-468, PubMed).

    For more information on this dataset's structure,
    see :footcite:`AAL_atlas`,
    and :footcite:`TZOURIOMAZOYER2002273`.

    Parameters
    ----------
    version : string {'SPM12', 'SPM5', 'SPM8'}, optional
        The version of the AAL atlas. Must be SPM5, SPM8 or SPM12.
        Default='SPM12'.

    data_dir : string, optional
        Directory where data should be downloaded and unpacked.

    url : string, optional
        Url of file to download.

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, keys are:

        - "maps": str. path to nifti file containing regions.

        - "labels": list of the names of the regions

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Licence: unknown.

    """
    versions = ['SPM5', 'SPM8', 'SPM12']
    if version not in versions:
        raise ValueError('The version of AAL requested "%s" does not exist.'
                         'Please choose one among %s.' %
                         (version, str(versions)))

    if url is None:
        baseurl = "http://www.gin.cnrs.fr/AAL_files/aal_for_%s.tar.gz"
        url = baseurl % version
    opts = {'uncompress': True}

    dataset_name = "aal_" + version
    # keys and basenames would need to be handled for each spm_version
    # for now spm_version 12 is hardcoded.
    basenames = ("AAL.nii", "AAL.xml")
    filenames = [(os.path.join('aal', 'atlas', f), url, opts)
                 for f in basenames]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    atlas_img, labels_file = _fetch_files(data_dir, filenames, resume=resume,
                                          verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)
    # We return the labels contained in the xml file as a dictionary
    xml_tree = xml.etree.ElementTree.parse(labels_file)
    root = xml_tree.getroot()
    labels = []
    indices = []
    for label in root.iter('label'):
        indices.append(label.find('index').text)
        labels.append(label.find('name').text)

    params = {'description': fdescr, 'maps': atlas_img,
              'labels': labels, 'indices': indices}

    return Bunch(**params)


def fetch_atlas_basc_multiscale_2015(version='sym', data_dir=None, url=None,
                                     resume=True, verbose=1):
    """Downloads and loads multiscale functional brain parcellations

    This atlas includes group brain parcellations generated from
    resting-state functional magnetic resonance images from about
    200 young healthy subjects.

    Multiple scales (number of networks) are available, among
    7, 12, 20, 36, 64, 122, 197, 325, 444. The brain parcellations
    have been generated using a method called bootstrap analysis of
    stable clusters called as BASC :footcite:`BELLEC20101126`,
    and the scales have been selected using a data-driven method
    called MSTEPS :footcite:`Bellec2013Mining`.

    Note that two versions of the template are available, 'sym' or 'asym'.
    The 'asym' type contains brain images that have been registered in the
    asymmetric version of the MNI brain template (reflecting that the brain
    is asymmetric), while the 'sym' type contains images registered in the
    symmetric version of the MNI template. The symmetric template has been
    forced to be symmetric anatomically, and is therefore ideally suited to
    study homotopic functional connections in fMRI: finding homotopic regions
    simply consists of flipping the x-axis of the template.

    .. versionadded:: 0.2.3

    Parameters
    ----------
    version : str {'sym', 'asym'}, optional
        Available versions are 'sym' or 'asym'. By default all scales of
        brain parcellations of version 'sym' will be returned.
        Default='sym'.

    data_dir : str, optional
        Directory where data should be downloaded and unpacked.

    url : str, optional
        Url of file to download.

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, Keys are:

        - "scale007", "scale012", "scale020", "scale036", "scale064",
          "scale122", "scale197", "scale325", "scale444": str, path
          to Nifti file of various scales of brain parcellations.

        - "description": details about the data release.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    For more information on this dataset's structure, see
    https://figshare.com/articles/basc/1285615

    """
    versions = ['sym', 'asym']
    if version not in versions:
        raise ValueError('The version of Brain parcellations requested "%s" '
                         'does not exist. Please choose one among them %s.' %
                         (version, str(versions)))

    keys = ['scale007', 'scale012', 'scale020', 'scale036', 'scale064',
            'scale122', 'scale197', 'scale325', 'scale444']

    if version == 'sym':
        url = "https://ndownloader.figshare.com/files/1861819"
    elif version == 'asym':
        url = "https://ndownloader.figshare.com/files/1861820"
    opts = {'uncompress': True}

    dataset_name = "basc_multiscale_2015"
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    folder_name = 'template_cambridge_basc_multiscale_nii_' + version
    basenames = ['template_cambridge_basc_multiscale_' + version +
                 '_' + key + '.nii.gz' for key in keys]

    filenames = [(os.path.join(folder_name, basename), url, opts)
                 for basename in basenames]
    data = _fetch_files(data_dir, filenames, resume=resume, verbose=verbose)

    descr = _get_dataset_descr(dataset_name)

    params = dict(zip(keys, data))
    params['description'] = descr

    return Bunch(**params)


def fetch_coords_dosenbach_2010(ordered_regions=True):
    """Load the Dosenbach et al. 160 ROIs. These ROIs cover
    much of the cerebral cortex and cerebellum and are assigned to 6
    networks.

    See :footcite:`Dosenbach20101358`.

    Parameters
    ----------
    ordered_regions : bool, optional
        ROIs from same networks are grouped together and ordered with respect
        to their names and their locations (anterior to posterior).
        Default=True.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, contains:

        - "rois": coordinates of 160 ROIs in MNI space
        - "labels": ROIs labels
        - "networks": networks names

    References
    ----------
    .. footbibliography::

    """
    dataset_name = 'dosenbach_2010'
    fdescr = _get_dataset_descr(dataset_name)
    package_directory = os.path.dirname(os.path.abspath(__file__))
    csv = os.path.join(package_directory, "data", "dosenbach_2010.csv")
    out_csv = np.recfromcsv(csv)

    if ordered_regions:
        out_csv = np.sort(out_csv, order=['network', 'name', 'y'])

    # We add the ROI number to its name, since names are not unique
    names = out_csv['name']
    numbers = out_csv['number']
    labels = np.array(['{0} {1}'.format(name, number) for (name, number) in
                       zip(names, numbers)])
    params = dict(rois=out_csv[['x', 'y', 'z']],
                  labels=labels,
                  networks=out_csv['network'], description=fdescr)

    return Bunch(**params)


def fetch_coords_seitzman_2018(ordered_regions=True):
    """Load the Seitzman et al. 300 ROIs. These ROIs cover cortical,
    subcortical and cerebellar regions and are assigned to one of 13
    networks (Auditory, CinguloOpercular, DefaultMode, DorsalAttention,
    FrontoParietal, MedialTemporalLobe, ParietoMedial, Reward, Salience,
    SomatomotorDorsal, SomatomotorLateral, VentralAttention, Visual) and
    have a regional label (cortexL, cortexR, cerebellum, thalamus, hippocampus,
    basalGanglia, amygdala, cortexMid).

    See :footcite:`SEITZMAN2020116290`.

    .. versionadded:: 0.5.1

    Parameters
    ----------
    ordered_regions : bool, optional
        ROIs from same networks are grouped together and ordered with respect
        to their locations (anterior to posterior). Default=True.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, contains:

        - "rois": Coordinates of 300 ROIs in MNI space
        - "radius": Radius of each ROI in mm
        - "networks": Network names
        - "regions": Region names

    References
    ----------
    .. footbibliography::

    """
    dataset_name = 'seitzman_2018'
    fdescr = _get_dataset_descr(dataset_name)
    package_directory = os.path.dirname(os.path.abspath(__file__))
    roi_file = os.path.join(package_directory, "data",
                            "seitzman_2018_ROIs_300inVol_MNI_allInfo.txt")
    anatomical_file = os.path.join(package_directory, "data",
                                   "seitzman_2018_ROIs_anatomicalLabels.txt")

    rois = np.recfromcsv(roi_file, delimiter=" ")
    rois = recfunctions.rename_fields(rois, {"netname": "network",
                                             "radiusmm": "radius"})
    rois.network = rois.network.astype(str)

    # get integer regional labels and convert to text labels with mapping
    # from header line
    with open(anatomical_file, 'r') as fi:
        header = fi.readline()
    region_mapping = {}
    for r in header.strip().split(","):
        i, region = r.split("=")
        region_mapping[int(i)] = region

    anatomical = np.genfromtxt(anatomical_file, skip_header=1)
    anatomical_names = np.array([region_mapping[a] for a in anatomical])

    rois = recfunctions.merge_arrays((rois, anatomical_names),
                                     asrecarray=True, flatten=True)
    rois.dtype.names = rois.dtype.names[:-1] + ("region",)

    if ordered_regions:
        rois = np.sort(rois, order=['network', 'y'])

    params = dict(rois=rois[['x', 'y', 'z']],
                  radius=rois['radius'],
                  networks=rois['network'].astype(str),
                  regions=rois['region'], description=fdescr)

    return Bunch(**params)


def fetch_atlas_allen_2011(data_dir=None, url=None, resume=True, verbose=1):
    """Download and return file names for the Allen and MIALAB ICA atlas
    (dated 2011).

    See :footcite:`Allen2011baseline`.

    The provided images are in MNI152 space.

    Parameters
    ----------
    data_dir : str, optional
        Directory where data should be downloaded and unpacked.

    url : str, optional
        Url of file to download.

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, keys are:

        - "maps": T-maps of all 75 unthresholded components.
        - "rsn28": T-maps of 28 RSNs included in E. Allen et al.
        - "networks": string list containing the names for the 28 RSNs.
        - "rsn_indices": dict[rsn_name] -> list of int, indices in the "maps"
          file of the 28 RSNs.
        - "comps": The aggregate ICA Components.
        - "description": details about the data release.

    References
    ----------
    .. footbibliography::

    Notes
    -----
    Licence: unknown

    See http://mialab.mrn.org/data/index.html for more information
    on this dataset.

    """
    if url is None:
        url = "https://osf.io/hrcku/download"

    dataset_name = "allen_rsn_2011"
    keys = ("maps",
            "rsn28",
            "comps")

    opts = {'uncompress': True}
    files = ["ALL_HC_unthresholded_tmaps.nii.gz",
             "RSN_HC_unthresholded_tmaps.nii.gz",
             "rest_hcp_agg__component_ica_.nii.gz"]

    labels = [('Basal Ganglia', [21]),
              ('Auditory', [17]),
              ('Sensorimotor', [7, 23, 24, 38, 56, 29]),
              ('Visual', [46, 64, 67, 48, 39, 59]),
              ('Default-Mode', [50, 53, 25, 68]),
              ('Attentional', [34, 60, 52, 72, 71, 55]),
              ('Frontal', [42, 20, 47, 49])]

    networks = [[name] * len(idxs) for name, idxs in labels]

    filenames = [(os.path.join('allen_rsn_2011', f), url, opts) for f in files]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    sub_files = _fetch_files(data_dir, filenames, resume=resume,
                             verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)

    params = [('description', fdescr),
              ('rsn_indices', labels),
              ('networks', networks)]
    params.extend(list(zip(keys, sub_files)))

    return Bunch(**dict(params))


def fetch_atlas_surf_destrieux(data_dir=None, url=None,
                               resume=True, verbose=1):
    """Download and load Destrieux et al, 2010 cortical atlas
    
    See :footcite:`DESTRIEUX20101`.

    This atlas returns 76 labels per hemisphere based on sulco-gryal patterns
    as distributed with Freesurfer in fsaverage5 surface space.

    .. versionadded:: 0.3

    Parameters
    ----------
    data_dir : str, optional
        Path of the data directory. Use to force data storage in a non-
        standard location. Default: None

    url : str, optional
        Download URL of the dataset. Overwrite the default URL.

    resume : bool, optional
        If True, try resuming download if possible.
        Default=True.

    verbose : int, optional
        Defines the level of verbosity of the output.
        Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, contains:

        - "labels": list
                     Contains region labels

        - "map_left": numpy.ndarray
                      Index into 'labels' for each vertex on the
                      left hemisphere of the fsaverage5 surface

        - "map_right": numpy.ndarray
                       Index into 'labels' for each vertex on the
                       right hemisphere of the fsaverage5 surface

        - "description": str
                         Details about the dataset

    References
    ----------
    .. footbibliography::

    """
    if url is None:
        url = "https://www.nitrc.org/frs/download.php/"

    dataset_name = 'destrieux_surface'
    fdescr = _get_dataset_descr(dataset_name)
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    # Download annot files, fsaverage surfaces and sulcal information
    annot_file = '%s.aparc.a2009s.annot'
    annot_url = url + '%i/%s.aparc.a2009s.annot'
    annot_nids = {'lh annot': 9343, 'rh annot': 9342}

    annots = []
    for hemi in [('lh', 'left'), ('rh', 'right')]:

        annot = _fetch_files(data_dir,
                             [(annot_file % (hemi[1]),
                               annot_url % (annot_nids['%s annot' % hemi[0]],
                                            hemi[0]),
                              {'move': annot_file % (hemi[1])})],
                             resume=resume, verbose=verbose)[0]
        annots.append(annot)

    annot_left = nb.freesurfer.read_annot(annots[0])
    annot_right = nb.freesurfer.read_annot(annots[1])

    return Bunch(labels=annot_left[2],  map_left=annot_left[0],
                 map_right=annot_right[0], description=fdescr)


def _separate_talairach_levels(atlas_img, labels, verbose=1):
    """Separate the multiple annotation levels in talairach raw atlas.

    The Talairach atlas has five levels of annotation: hemisphere, lobe, gyrus,
    tissue, brodmann area. They are mixed up in the original atlas: each label
    in the atlas corresponds to a 5-tuple containing, for each of these levels,
    a value or the string '*' (meaning undefined, background).

    This function disentangles the levels, and stores each on an octet in an
    int64 image (the level with most labels, ba, has 72 labels).
    This way, any subset of these levels can be accessed by applying a bitwise
    mask.

    In the created image, the least significant octet contains the hemisphere,
    the next one the lobe, then gyrus, tissue, and ba. Background is 0.
    The labels contain
    [('level name', ['labels', 'for', 'this', 'level' ...]), ...],
    where the levels are in the order mentionned above.

    The label '*' is replaced by 'Background' for clarity.

    """
    labels = np.asarray(labels)
    if verbose:
        print(
            'Separating talairach atlas levels: {}'.format(_TALAIRACH_LEVELS))
    levels = []
    new_img = np.zeros(atlas_img.shape, dtype=np.int64)
    for pos, level in enumerate(_TALAIRACH_LEVELS):
        if verbose:
            print(level)
        level_img = np.zeros(atlas_img.shape, dtype=np.int64)
        level_labels = {'*': 0}
        for region_nb, region in enumerate(labels[:, pos]):
            level_labels.setdefault(region, len(level_labels))
            level_img[get_data(atlas_img) == region_nb] = level_labels[
                region]
        # shift this level to its own octet and add it to the new image
        level_img <<= 8 * pos
        new_img |= level_img
        # order the labels so that image values are indices in the list of
        # labels for each level
        level_labels = list(list(
            zip(*sorted(level_labels.items(), key=lambda t: t[1])))[0])
        # rename '*' -> 'Background'
        level_labels[0] = 'Background'
        levels.append((level, level_labels))
    new_img = new_img_like(atlas_img, data=new_img)
    return new_img, levels


def _get_talairach_all_levels(data_dir=None, verbose=1):
    """Get the path to Talairach atlas and labels

    The atlas is downloaded and the files are created if necessary.

    The image contains all five levels of the atlas, each encoded on 8 bits
    (least significant octet contains the hemisphere, the next one the lobe,
    then gyrus, tissue, and ba).

    The labels json file contains
    [['level name', ['labels', 'for', 'this', 'level' ...]], ...],
    where the levels are in the order mentionned above.

    """
    data_dir = _get_dataset_dir(
        'talairach_atlas', data_dir=data_dir, verbose=verbose)
    img_file = os.path.join(data_dir, 'talairach.nii')
    labels_file = os.path.join(data_dir, 'talairach_labels.json')
    if os.path.isfile(img_file) and os.path.isfile(labels_file):
        return img_file, labels_file
    atlas_url = 'http://www.talairach.org/talairach.nii'
    temp_dir = mkdtemp()
    try:
        temp_file = _fetch_files(
            temp_dir, [('talairach.nii', atlas_url, {})], verbose=verbose)[0]
        atlas_img = nb.load(temp_file, mmap=False)
        atlas_img = check_niimg(atlas_img)
    finally:
        shutil.rmtree(temp_dir)
    labels = atlas_img.header.extensions[0].get_content()
    labels = labels.strip().decode('utf-8').split('\n')
    labels = [l.split('.') for l in labels]
    new_img, level_labels = _separate_talairach_levels(
        atlas_img, labels, verbose=verbose)
    new_img.to_filename(img_file)
    with open(labels_file, 'w') as fp:
        json.dump(level_labels, fp)
    return img_file, labels_file


def fetch_atlas_talairach(level_name, data_dir=None, verbose=1):
    """Download the Talairach atlas.

    For more information, see :footcite:`talairach_atlas`,
    :footcite:`Lancaster2000Talairach`,
    and :footcite:`Lancaster1997labeling`.

    .. versionadded:: 0.4.0

    Parameters
    ----------
    level_name : string {'hemisphere', 'lobe', 'gyrus', 'tissue', 'ba'}
        Which level of the atlas to use: the hemisphere, the lobe, the gyrus,
        the tissue type or the Brodmann area.

    data_dir : str, optional
        Path of the data directory. Used to force data storage in a specified
        location.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    sklearn.datasets.base.Bunch
        Dictionary-like object, contains:

        - maps: 3D Nifti image, values are indices in the list of labels.
        - labels: list of strings. Starts with 'Background'.
        - description: a short description of the atlas and some references.

    References
    ----------
    .. footbibliography::

    """
    if level_name not in _TALAIRACH_LEVELS:
        raise ValueError('"level_name" should be one of {}'.format(
            _TALAIRACH_LEVELS))
    position = _TALAIRACH_LEVELS.index(level_name)
    atlas_file, labels_file = _get_talairach_all_levels(data_dir, verbose)
    atlas_img = check_niimg(atlas_file)
    with open(labels_file) as fp:
        labels = json.load(fp)[position][1]
    level_data = (get_data(atlas_img) >> 8 * position) & 255
    atlas_img = new_img_like(atlas_img, data=level_data)
    description = _get_dataset_descr(
        'talairach_atlas').decode('utf-8').format(level_name)
    return Bunch(maps=atlas_img, labels=labels, description=description)


def fetch_atlas_pauli_2017(version='prob', data_dir=None, verbose=1):
    """Download the Pauli et al. (2017) atlas with in total
    12 subcortical nodes

    See :footcite:`pauli_atlas` and :footcite:`Pauli2018probabilistic`.

    Parameters
    ----------
    version : str {'prob', 'det'}, optional
        Which version of the atlas should be download. This can be
        'prob' for the probabilistic atlas or 'det' for the
        deterministic atlas. Default='prob'.

    data_dir : str, optional
        Path of the data directory. Used to force data storage in a specified
        location.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    sklearn.datasets.base.Bunch
        Dictionary-like object, contains:

        - maps: 3D Nifti image, values are indices in the list of labels.
        - labels: list of strings. Starts with 'Background'.
        - description: a short description of the atlas and some references.

    References
    ----------
    .. footbibliography::

    """
    if version == 'prob':
        url_maps = 'https://osf.io/w8zq2/download'
        filename = 'pauli_2017_prob.nii.gz'
    elif version == 'det':
        url_maps = 'https://osf.io/5mqfx/download'
        filename = 'pauli_2017_det.nii.gz'
    else:
        raise NotImplementedError('{} is no valid version for '.format(version) + \
                                  'the Pauli atlas')

    url_labels = 'https://osf.io/6qrcb/download'
    dataset_name = 'pauli_2017'

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)

    files = [(filename,
              url_maps,
              {'move':filename}),
             ('labels.txt',
              url_labels,
              {'move':'labels.txt'})]
    atlas_file, labels = _fetch_files(data_dir, files)

    labels = np.loadtxt(labels, dtype=str)[:, 1].tolist()

    fdescr = _get_dataset_descr(dataset_name)

    return Bunch(maps=atlas_file,
                 labels=labels,
                 description=fdescr)



def fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=1,
                              data_dir=None, base_url=None, resume=True,
                              verbose=1):
    """Download and return file names for the Schaefer 2018 parcellation

    .. versionadded:: 0.5.1

    The provided images are in MNI152 space.

    For more information on this dataset, see :footcite:`schaefer_atlas`,
    :footcite:`Schaefer2017parcellation`,
    and :footcite:`Yeo2011organization`.

    Parameters
    ----------
    n_rois : int, optional
        Number of regions of interest {100, 200, 300, 400, 500, 600,
        700, 800, 900, 1000}.
        Default=400.

    yeo_networks : int, optional
        ROI annotation according to yeo networks {7, 17}.
        Default=7.

    resolution_mm : int, optional
        Spatial resolution of atlas image in mm {1, 2}.
        Default=1mm.

    data_dir : string, optional
        Directory where data should be downloaded and unpacked.

    base_url : string, optional
        base_url of files to download (None results in default base_url).

    resume : bool, optional
        Whether to resumed download of a partly-downloaded file.
        Default=True.

    verbose : int, optional
        Verbosity level (0 means no message). Default=1.

    Returns
    -------
    data : sklearn.datasets.base.Bunch
        Dictionary-like object, contains:

        - maps: 3D Nifti image, values are indices in the list of labels.
        - labels: ROI labels including Yeo-network annotation,list of strings.
        - description: A short description of the atlas and some references.

    References
    ----------
    .. footbibliography::


    Notes
    -----
    Release v0.14.3 of the Schaefer 2018 parcellation is used by
    default. Versions prior to v0.14.3 are known to contain erroneous region
    label names. For more details, see
    https://github.com/ThomasYeoLab/CBIG/blob/master/stable_projects/brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/Updates/Update_20190916_README.md

    Licence: MIT.

    """
    valid_n_rois = list(range(100, 1100, 100))
    valid_yeo_networks = [7, 17]
    valid_resolution_mm = [1, 2]
    if n_rois not in valid_n_rois:
        raise ValueError("Requested n_rois={} not available. Valid "
                         "options: {}".format(n_rois, valid_n_rois))
    if yeo_networks not in valid_yeo_networks:
        raise ValueError("Requested yeo_networks={} not available. Valid "
                         "options: {}".format(yeo_networks,valid_yeo_networks))
    if resolution_mm not in valid_resolution_mm:
        raise ValueError("Requested resolution_mm={} not available. Valid "
                         "options: {}".format(resolution_mm,
                                              valid_resolution_mm)
                         )

    if base_url is None:
        base_url = ('https://raw.githubusercontent.com/ThomasYeoLab/CBIG/'
                    'v0.14.3-Update_Yeo2011_Schaefer2018_labelname/'
                    'stable_projects/brain_parcellation/'
                    'Schaefer2018_LocalGlobal/Parcellations/MNI/'
                    )

    files = []
    labels_file_template = 'Schaefer2018_{}Parcels_{}Networks_order.txt'
    img_file_template = ('Schaefer2018_{}Parcels_'
                         '{}Networks_order_FSLMNI152_{}mm.nii.gz')
    for f in [labels_file_template.format(n_rois, yeo_networks),
              img_file_template.format(n_rois, yeo_networks, resolution_mm)]:
        files.append((f, base_url + f, {}))

    dataset_name = 'schaefer_2018'
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    labels_file, atlas_file = _fetch_files(data_dir, files, resume=resume,
                                           verbose=verbose)

    labels = np.genfromtxt(labels_file, usecols=1, dtype="S", delimiter="\t")
    fdescr = _get_dataset_descr(dataset_name)

    return Bunch(maps=atlas_file,
                 labels=labels,
                 description=fdescr)
