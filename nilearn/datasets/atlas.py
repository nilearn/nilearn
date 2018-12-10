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
from sklearn.datasets.base import Bunch

from .utils import _get_dataset_dir, _fetch_files, _get_dataset_descr
from .._utils import check_niimg
from .._utils.compat import _basestring
from ..image import new_img_like

_TALAIRACH_LEVELS = ['hemisphere', 'lobe', 'gyrus', 'tissue', 'ba']


def fetch_atlas_craddock_2012(data_dir=None, url=None, resume=True, verbose=1):
    """Download and return file names for the Craddock 2012 parcellation

    The provided images are in MNI152 space.

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked.

    url: string
        url of file to download.

    resume: bool
        whether to resumed download of a partly-downloaded file.

    verbose: int
        verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:
        scorr_mean, tcorr_mean,
        scorr_2level, tcorr_2level,
        random

    References
    ----------
    Licence: Creative Commons Attribution Non-commercial Share Alike
    http://creativecommons.org/licenses/by-nc-sa/2.5/

    Craddock, R. Cameron, G.Andrew James, Paul E. Holtzheimer, Xiaoping P. Hu,
    and Helen S. Mayberg. "A Whole Brain fMRI Atlas Generated via Spatially
    Constrained Spectral Clustering". Human Brain Mapping 33, no 8 (2012):
    1914-1928. doi:10.1002/hbm.21333.

    See http://www.nitrc.org/projects/cluster_roi/ for more information
    on this parcellation.
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

    Parameters
    ----------
    lateralized: boolean, optional
        If True, returns an atlas with distinct regions for right and left
        hemispheres.
    data_dir: string, optional
        Path of the data directory. Use to forec data storage in a non-
        standard location. Default: None (meaning: default)
    url: string, optional
        Download URL of the dataset. Overwrite the default URL.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, contains:
        - Cortical ROIs, lateralized or not (maps)
        - Labels of the ROIs (labels)

    References
    ----------

    Fischl, Bruce, et al. "Automatically parcellating the human cerebral
    cortex." Cerebral cortex 14.1 (2004): 11-22.

    Destrieux, C., et al. "A sulcal depth-based anatomical parcellation
    of the cerebral cortex." NeuroImage 47 (2009): S151.
    """
    if url is None:
        url = "https://www.nitrc.org/frs/download.php/7739/"

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
    atlas_name: string
        Name of atlas to load. Can be:
        cort-maxprob-thr0-1mm,  cort-maxprob-thr0-2mm,
        cort-maxprob-thr25-1mm, cort-maxprob-thr25-2mm,
        cort-maxprob-thr50-1mm, cort-maxprob-thr50-2mm,
        sub-maxprob-thr0-1mm,  sub-maxprob-thr0-2mm,
        sub-maxprob-thr25-1mm, sub-maxprob-thr25-2mm,
        sub-maxprob-thr50-1mm, sub-maxprob-thr50-2mm,
        cort-prob-1mm, cort-prob-2mm,
        sub-prob-1mm, sub-prob-2mm

    data_dir: string, optional
        Path of data directory where data will be stored. Optionally,
        it can also be a FSL installation directory (which is dependent
        on your installation).
        Example, if FSL is installed in /usr/share/fsl/ then
        specifying as '/usr/share/' can get you Harvard Oxford atlas
        from your installed directory. Since we mimic same root directory
        as FSL to load it easily from your installation.

    symmetric_split: bool, optional, (default False).
        If True, lateralized atlases of cort or sub with maxprob will be
        returned. For subcortical types (sub-maxprob), we split every
        symmetric region in left and right parts. Effectively doubles the
        number of regions.
        NOTE Not implemented for full probabilistic atlas (*-prob-* atlases).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:

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

    atlas = atlas_img.get_data()

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

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    url: string, optional
        Override download URL. Used for test only (or if you setup a mirror of
        the data).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        Dictionary-like object, the interest attributes are :

        - 'maps': str, path to nifti file containing regions definition.
        - 'labels': string list containing the labels of the regions.
        - 'region_coords': tuple list (x, y, z) containing coordinates
          of each region in MNI space.
        - 'networks': string list containing names of the networks.
        - 'description': description about the atlas.


    References
    ----------
    :Download:
        https://team.inria.fr/parietal/files/2015/01/MSDL_rois.zip

    :Paper to cite:
        `Multi-subject dictionary learning to segment an atlas of brain
        spontaneous activity <http://hal.inria.fr/inria-00588898/en>`_
        Gael Varoquaux, Alexandre Gramfort, Fabian Pedregosa, Vincent Michel,
        Bertrand Thirion. Information Processing in Medical Imaging, 2011,
        pp. 562-573, Lecture Notes in Computer Science.

    :Other references:
        `Learning and comparing functional connectomes across subjects
        <http://hal.inria.fr/hal-00812911/en>`_.
        Gael Varoquaux, R.C. Craddock NeuroImage, 2013.

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
    """Download and load the Power et al. brain atlas composed of 264 ROIs.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, contains:
        - "rois": coordinates of 264 ROIs in MNI space


    References
    ----------
    Power, Jonathan D., et al. "Functional network organization of the human
    brain." Neuron 72.4 (2011): 665-678.
    """
    dataset_name = 'power_2011'
    fdescr = _get_dataset_descr(dataset_name)
    package_directory = os.path.dirname(os.path.abspath(__file__))
    csv = os.path.join(package_directory, "data", "power_2011.csv")
    params = dict(rois=np.recfromcsv(csv), description=fdescr)

    return Bunch(**params)


def fetch_atlas_smith_2009(data_dir=None, mirror='origin', url=None,
                           resume=True, verbose=1):
    """Download and load the Smith ICA and BrainMap atlas (dated 2009)

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a non-
        standard location. Default: None (meaning: default)
    mirror: string, optional
        By default, the dataset is downloaded from the original website of the
        atlas. Specifying "nitrc" will force download from a mirror, with
        potentially higher bandwith.
    url: string, optional
        Download URL of the dataset. Overwrite the default URL.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, contains:

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

    S.M. Smith, P.T. Fox, K.L. Miller, D.C. Glahn, P.M. Fox, C.E. Mackay, N.
    Filippini, K.E. Watkins, R. Toro, A.R. Laird, and C.F. Beckmann.
    Correspondence of the brain's functional architecture during activation and
    rest. Proc Natl Acad Sci USA (PNAS), 106(31):13040-13045, 2009.

    A.R. Laird, P.M. Fox, S.B. Eickhoff, J.A. Turner, K.L. Ray, D.R. McKay, D.C
    Glahn, C.F. Beckmann, S.M. Smith, and P.T. Fox. Behavioral interpretations
    of intrinsic connectivity networks. Journal of Cognitive Neuroscience, 2011

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

    if isinstance(url, _basestring):
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

    Parameters
    ----------
    data_dir: string
        directory where data should be downloaded and unpacked.

    url: string
        url of file to download.

    resume: bool
        whether to resumed download of a partly-downloaded file.

    verbose: int
        verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:

        - "thin_7", "thick_7": 7-region parcellations,
          fitted to resp. thin and thick template cortex segmentations.

        - "thin_17", "thick_17": 17-region parcellations.

        - "colors_7", "colors_17": colormaps (text files) for 7- and 17-region
          parcellation respectively.

        - "anat": anatomy image.

    Notes
    -----
    For more information on this dataset's structure, see
    http://surfer.nmr.mgh.harvard.edu/fswiki/CorticalParcellation_Yeo2011

    Yeo BT, Krienen FM, Sepulcre J, Sabuncu MR, Lashkari D, Hollinshead M,
    Roffman JL, Smoller JW, Zollei L., Polimeni JR, Fischl B, Liu H,
    Buckner RL. The organization of the human cerebral cortex estimated by
    intrinsic functional connectivity. J Neurophysiol 106(3):1125-65, 2011.

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

    Parameters
    ----------
    version: string, optional
        The version of the AAL atlas. Must be SPM5, SPM8 or SPM12. Default is
        SPM12.

    data_dir: string
        directory where data should be downloaded and unpacked.

    url: string
        url of file to download.

    resume: bool
        whether to resumed download of a partly-downloaded file.

    verbose: int
        verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:

        - "maps": str. path to nifti file containing regions.

        - "labels": list of the names of the regions

    Notes
    -----
    For more information on this dataset's structure, see
    http://www.gin.cnrs.fr/AAL-217?lang=en

    Automated Anatomical Labeling of Activations in SPM Using a Macroscopic
    Anatomical Parcellation of the MNI MRI Single-Subject Brain.
    N. Tzourio-Mazoyer, B. Landeau, D. Papathanassiou, F. Crivello,
    O. Etard, N. Delcroix, B. Mazoyer, and M. Joliot.
    NeuroImage 2002. 15 :273-28

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
    for label in root.getiterator('label'):
        indices.append(label.find('index').text)
        labels.append(label.find('name').text)

    params = {'description': fdescr, 'maps': atlas_img,
              'labels': labels, 'indices': indices}

    return Bunch(**params)


def fetch_atlas_basc_multiscale_2015(version='sym', data_dir=None,
                                     resume=True, verbose=1):
    """Downloads and loads multiscale functional brain parcellations

    This atlas includes group brain parcellations generated from
    resting-state functional magnetic resonance images from about
    200 young healthy subjects.

    Multiple scales (number of networks) are available, among
    7, 12, 20, 36, 64, 122, 197, 325, 444. The brain parcellations
    have been generated using a method called bootstrap analysis of
    stable clusters called as BASC, (Bellec et al., 2010) and the
    scales have been selected using a data-driven method called MSTEPS
    (Bellec, 2013).

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
    version: str, optional
        Available versions are 'sym' or 'asym'. By default all scales of
        brain parcellations of version 'sym' will be returned.

    data_dir: str, optional
        directory where data should be downloaded and unpacked.

    url: str, optional
        url of file to download.

    resume: bool
        whether to resumed download of a partly-downloaded file.

    verbose: int
        verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, Keys are:

        - "scale007", "scale012", "scale020", "scale036", "scale064",
          "scale122", "scale197", "scale325", "scale444": str, path
          to Nifti file of various scales of brain parcellations.

        - "description": details about the data release.

    References
    ----------
    Bellec P, Rosa-Neto P, Lyttelton OC, Benali H, Evans AC, Jul. 2010.
    Multi-level bootstrap analysis of stable clusters in resting-state fMRI.
    NeuroImage 51 (3), 1126-1139.
    URL http://dx.doi.org/10.1016/j.neuroimage.2010.02.082

    Bellec P, Jun. 2013. Mining the Hierarchy of Resting-State Brain Networks:
    Selection of Representative Clusters in a Multiscale Structure.
    Pattern Recognition in Neuroimaging (PRNI), 2013 pp. 54-57.

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

    Parameters
    ----------
    ordered_regions : bool, optional
        ROIs from same networks are grouped together and ordered with respect
        to their names and their locations (anterior to posterior).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, contains:
        - "rois": coordinates of 160 ROIs in MNI space
        - "labels": ROIs labels
        - "networks": networks names

    References
    ----------
    Dosenbach N.U., Nardos B., et al. "Prediction of individual brain maturity
    using fMRI.", 2010, Science 329, 1358-1361.
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


def fetch_atlas_allen_2011(data_dir=None, url=None, resume=True, verbose=1):
    """Download and return file names for the Allen and MIALAB ICA atlas
    (dated 2011).

    The provided images are in MNI152 space.

    Parameters
    ----------
    data_dir: str, optional
        directory where data should be downloaded and unpacked.
    url: str, optional
        url of file to download.
    resume: bool
        whether to resumed download of a partly-downloaded file.
    verbose: int
        verbosity level (0 means no message).

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, keys are:

        - "maps": T-maps of all 75 unthresholded components.
        - "rsn28": T-maps of 28 RSNs included in E. Allen et al.
        - "networks": string list containing the names for the 28 RSNs.
        - "rsn_indices": dict[rsn_name] -> list of int, indices in the "maps"
          file of the 28 RSNs.
        - "comps": The aggregate ICA Components.
        - "description": details about the data release.

    References
    ----------
    E. Allen, et al, "A baseline for the multivariate comparison of resting
    state networks," Frontiers in Systems Neuroscience, vol. 5, p. 12, 2011.

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
    """Download and load Destrieux et al, 2010 cortical atlas.

    This atlas returns 76 labels per hemisphere based on sulco-gryal pattnerns
    as distributed with Freesurfer in fsaverage5 surface space.

    .. versionadded:: 0.3

    Parameters
    ----------
    data_dir: str, optional
        Path of the data directory. Use to force data storage in a non-
        standard location. Default: None

    url: str, optional
        Download URL of the dataset. Overwrite the default URL.

    resume: bool, optional (default True)
        If True, try resuming download if possible.

    verbose: int, optional (default 1)
        Defines the level of verbosity of the output.

    Returns
    -------
    data: sklearn.datasets.base.Bunch
        dictionary-like object, contains:

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
    Destrieux et al. (2010), Automatic parcellation of human cortical gyri and
    sulci using standard anatomical nomenclature. NeuroImage 53, 1-15.
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
            level_img[atlas_img.get_data() == region_nb] = level_labels[
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

    .. versionadded:: 0.4.0

    Parameters
    ----------
    level_name : {'hemisphere', 'lobe', 'gyrus', 'tissue', 'ba'}
        Which level of the atlas to use: the hemisphere, the lobe, the gyrus,
          the tissue type or the Brodmann area.

    data_dir : str, optional (default=None)
        Path of the data directory. Used to force data storage in a specified
        location.

    verbose : int
        verbosity level (0 means no message).

    Returns
    -------
    sklearn.datasets.base.Bunch
        Dictionary-like object, contains:

        - maps: 3D Nifti image, values are indices in the list of labels.
        - labels: list of strings. Starts with 'Background'.
        - description: a short description of the atlas and some references.

    References
    ----------
    http://talairach.org/about.html#Labels

    `Lancaster JL, Woldorff MG, Parsons LM, Liotti M, Freitas CS, Rainey L,
    Kochunov PV, Nickerson D, Mikiten SA, Fox PT, "Automated Talairach Atlas
    labels for functional brain mapping". Human Brain Mapping 10:120-131,
    2000.`

    `Lancaster JL, Rainey LH, Summerlin JL, Freitas CS, Fox PT, Evans AC, Toga
    AW, Mazziotta JC. Automated labeling of the human brain: A preliminary
    report on the development and evaluation of a forward-transform method. Hum
    Brain Mapp 5, 238-242, 1997.`
    """
    if level_name not in _TALAIRACH_LEVELS:
        raise ValueError('"level_name" should be one of {}'.format(
            _TALAIRACH_LEVELS))
    position = _TALAIRACH_LEVELS.index(level_name)
    atlas_file, labels_file = _get_talairach_all_levels(data_dir, verbose)
    atlas_img = check_niimg(atlas_file)
    with open(labels_file) as fp:
        labels = json.load(fp)[position][1]
    level_data = (atlas_img.get_data() >> 8 * position) & 255
    atlas_img = new_img_like(atlas_img, data=level_data)
    description = _get_dataset_descr(
        'talairach_atlas').decode('utf-8').format(level_name)
    return Bunch(maps=atlas_img, labels=labels, description=description)


def fetch_atlas_pauli_2017(version='prob', data_dir=None, verbose=1):
    """Download the Pauli et al. (2017) atlas with in total
    12 subcortical nodes.

    Parameters
    ----------

    version: str, optional (default='prob')
        Which version of the atlas should be download. This can be 'prob'
        for the probabilistic atlas or 'det' for the deterministic atlas.

    data_dir : str, optional (default=None)
        Path of the data directory. Used to force data storage in a specified
        location.

    verbose : int
        verbosity level (0 means no message).

    Returns
    -------
    sklearn.datasets.base.Bunch
        Dictionary-like object, contains:

        - maps: 3D Nifti image, values are indices in the list of labels.
        - labels: list of strings. Starts with 'Background'.
        - description: a short description of the atlas and some references.

    References
    ----------
    https://osf.io/r2hvk/

    `Pauli, W. M., Nili, A. N., & Tyszka, J. M. (2018). A high-resolution
    probabilistic in vivo atlas of human subcortical brain nuclei.
    Scientific Data, 5, 180063-13. http://doi.org/10.1038/sdata.2018.63``
    """

    if version == 'prob':
        url_maps = 'https://osf.io/w8zq2/download'
        filename = 'pauli_2017_labels.nii.gz'
    elif version == 'labels':
        url_maps = 'https://osf.io/5mqfx/download'
        filename = 'pauli_2017_prob.nii.gz'
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
