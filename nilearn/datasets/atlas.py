"""
Downloading NeuroImaging datasets: atlas datasets
"""
import os
import xml.etree.ElementTree
import numpy as np
from scipy import ndimage

from sklearn.datasets.base import Bunch
from sklearn.utils import deprecated

#from . import utils
from .utils import _get_dataset_dir, _fetch_files, _get_dataset_descr

from .._utils import check_niimg
from ..image import new_img_like
from .._utils.compat import _basestring


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


@deprecated('it has been replace by fetch_atlas_craddock_2012 and '
            'will be removed in nilearn 0.1.5')
def fetch_craddock_2012_atlas(data_dir=None, url=None, resume=True, verbose=1):
    return fetch_atlas_craddock_2012(
        data_dir=data_dir,
        url=url,
        resume=resume,
        verbose=verbose)


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
    """Load Harvard-Oxford parcellation from FSL if installed or download it.

    This function looks up for Harvard Oxford atlas in the system and load it
    if present. If not, it downloads it and stores it in NILEARN_DATA
    directory.

    Parameters
    ==========
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
        Path of data directory. It can be FSL installation directory
        (which is dependent on your installation).

    symmetric_split: bool, optional
        If True, split every symmetric region in left and right parts.
        Effectively doubles the number of regions. Default: False.
        Not implemented for probabilistic atlas (*-prob-* atlases)

    Returns
    =======
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

    url = 'https://www.nitrc.org/frs/download.php/7700/HarvardOxford.tgz'

    # For practical reasons, we mimic the FSL data directory here.
    dataset_name = 'fsl'
    # Environment variables
    default_paths = []
    for env_var in ['FSL_DIR', 'FSLDIR']:
        path = os.getenv(env_var)
        if path is not None:
            default_paths.extend(path.split(':'))
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                default_paths=default_paths, verbose=verbose)
    opts = {'uncompress': True}
    root = os.path.join('data', 'atlases')
    atlas_file = os.path.join(root, 'HarvardOxford',
                              'HarvardOxford-' + atlas_name + '.nii.gz')
    if atlas_name[0] == 'c':
        label_file = 'HarvardOxford-Cortical.xml'
    else:
        label_file = 'HarvardOxford-Subcortical.xml'
    label_file = os.path.join(root, label_file)

    atlas_img, label_file = _fetch_files(
        data_dir,
        [(atlas_file, url, opts), (label_file, url, opts)],
        resume=resume, verbose=verbose)

    names = {}
    from xml.etree import ElementTree
    names[0] = 'Background'
    for label in ElementTree.parse(label_file).findall('.//label'):
        names[int(label.get('index')) + 1] = label.text
    names = np.asarray(list(names.values()))

    if not symmetric_split:
        return Bunch(maps=atlas_img, labels=names)

    if atlas_name in ("cort-prob-1mm", "cort-prob-2mm",
                      "sub-prob-1mm", "sub-prob-2mm"):
        raise ValueError("Region splitting not supported for probabilistic "
                         "atlases")

    atlas_img = check_niimg(atlas_img)
    atlas = atlas_img.get_data()

    labels = np.unique(atlas)
    # ndimage.find_objects output contains None elements for labels
    # that do not exist
    found_slices = (s for s in ndimage.find_objects(atlas)
                    if s is not None)
    middle_ind = (atlas.shape[0] - 1) // 2
    crosses_middle = [s.start < middle_ind and s.stop > middle_ind
                      for s, _, _ in found_slices]

    # Split every zone crossing the median plane into two parts.
    # Assumes that the background label is zero.
    half = np.zeros(atlas.shape, dtype=np.bool)
    half[:middle_ind, ...] = True
    new_label = max(labels) + 1
    # Put zeros on the median plane
    atlas[middle_ind, ...] = 0
    for label, crosses in zip(labels[1:], crosses_middle):
        if not crosses:
            continue
        atlas[np.logical_and(atlas == label, half)] = new_label
        new_label += 1

    # Duplicate labels for right and left
    new_names = [names[0]]
    for n in names[1:]:
        new_names.append(n + ', right part')
    for n in names[1:]:
        new_names.append(n + ', left part')

    atlas_img = new_img_like(atlas_img, atlas, atlas_img.get_affine())
    return Bunch(maps=atlas_img, labels=new_names)


@deprecated('it has been replaced by fetch_atlas_harvard_oxford and '
            'will be removed in nilearn 0.1.5')
def fetch_harvard_oxford(atlas_name, data_dir=None, symmetric_split=False,
                        resume=True, verbose=1):

    atlas = fetch_atlas_harvard_oxford(atlas_name, data_dir=data_dir,
                                       symmetric_split=symmetric_split,
                                       resume=resume, verbose=verbose)

    return atlas.maps, atlas.labels


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
        - 'labels': str. Path to csv file containing labels.
        - 'maps': str. path to nifti file containing regions definition.

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
    fdescr = _get_dataset_descr(dataset_name)

    return Bunch(labels=files[0], maps=files[1], description=fdescr)


@deprecated('it has been replace by fetch_atlas_msdl and '
            'will be removed in nilearn 0.1.5')
def fetch_msdl_atlas(data_dir=None, url=None, resume=True, verbose=1):
    return fetch_atlas_msdl(data_dir=data_dir, url=url,
                            resume=resume, verbose=verbose)


def fetch_atlas_power_2011():
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
        By default, the dataset is downloaded from the original website of the atlas.
        Specifying "nitrc" will force download from a mirror, with potentially
        higher bandwith.
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
    http://www.fmrib.ox.ac.uk/analysis/brainmap+rsns/
    """
    if url is None:
        if mirror == 'origin':
            url = "http://www.fmrib.ox.ac.uk/analysis/brainmap+rsns/"
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


@deprecated('it has been replace by fetch_atlas_smith_2009 and '
            'will be removed in nilearn 0.1.5')
def fetch_smith_2009(data_dir=None, mirror='origin', url=None, resume=True,
                     verbose=1):
    return fetch_atlas_smith_2009(
        data_dir=data_dir,
        mirror=mirror,
        url=url,
        resume=resume,
        verbose=verbose)


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
        url = "ftp://surfer.nmr.mgh.harvard.edu/" \
              "pub/data/Yeo_JNeurophysiol11_MNI152.zip"
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


@deprecated('it has been replace by fetch_atlas_yeo_2011 and '
            'will be removed in nilearn 0.1.5')
def fetch_yeo_2011_atlas(data_dir=None, url=None, resume=True, verbose=1):
    return fetch_atlas_yeo_2011(
        data_dir=data_dir,
        url=url,
        resume=resume,
        verbose=verbose)


def fetch_atlas_aal_spm_12(data_dir=None, url=None, resume=True, verbose=1):
    """Downloads and returns the AAL template for SPM 12.

    This atlas is the result of an automated anatomical parcellation of the
    spatially normalized single-subject high-resolution T1 volume provided by
    the Montreal Neurological Institute (MNI) (D. L. Collins et al., 1998,
    Trans. Med. Imag. 17, 463-468, PubMed).

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

        - "regions": str. path to nifti file containing regions.

        - "labels": dict. labels dictionary with their region id as key and
                    name as value

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
    spm_version = 12
    if url is None:
        baseurl = "http://www.gin.cnrs.fr/AAL_files/aal_for_SPM%i.tar.gz"
        url = baseurl % spm_version
    opts = {'uncompress': True}

    dataset_name = "aal_spm_12"
    # keys and basenames would need to be handled for each spm_version
    # for now spm_version 12 is hardcoded.
    basenames = ("AAL.nii", "AAL.xml")
    filenames = [(os.path.join("aal_for_SPM%i" % spm_version, f), url, opts)
                 for f in basenames]

    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir,
                                verbose=verbose)
    atlas_img, labels_file = _fetch_files(data_dir, filenames, resume=resume,
                                          verbose=verbose)

    fdescr = _get_dataset_descr(dataset_name)

    # We return the labels contained in the xml file as a dictionary
    xml_tree = xml.etree.ElementTree.parse(labels_file)
    root = xml_tree.getroot()
    labels_dict = {}
    for label in root.getiterator('label'):
        labels_dict[label.find('index').text] = label.find('name').text

    params = {'description': fdescr, 'regions': atlas_img,
              'labels': labels_dict}

    return Bunch(**params)
