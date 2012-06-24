""" Utilities to download NeuroImaging datasets
"""

import os
import urllib2
import tarfile
import zipfile
import sys
import shutil
import time

import numpy as np
from scipy import io
from sklearn.datasets.base import Bunch

import nibabel as ni


def _chunk_report_(bytes_so_far, total_size, t0):
    """Show downloading percentage

    Parameters
    ----------
    bytes_so_far: integer
        Number of downloaded bytes

    total_size: integer, optional
        Total size of the file. None is valid

    t0: integer, optional
        The time in seconds (as returned by time.time()) at which the
        download was started.
    """
    if total_size:
        percent = float(bytes_so_far) / total_size
        percent = round(percent * 100, 2)
        dt = time.time() - t0
        # We use a max to avoid a division by zero
        remaining = (100. - percent) / max(0.01, percent) * dt
        sys.stderr.write(
            "Downloaded %d of %d bytes (%0.2f%%, %i seconds remaining)\r"
            % (bytes_so_far, total_size, percent, remaining))
    else:
        sys.stderr.write("Downloaded %d of ? bytes\r" % (bytes_so_far))


def _chunk_read_(response, local_file, chunk_size=8192, report_hook=None):
    """Download a file chunk by chunk and show advancement

    Parameters
    ----------
    response: urllib.addinfourl
        Response to the download request in order to get file size

    local_file: file
        Hard disk file where data should be written

    chunk_size: integer, optional
        Size of downloaded chunks. Default: 8192

    report_hook: boolean
        Whether or not to show downloading advancement. Default: None

    Returns
    -------
    data: string
        The downloaded file.

    """
    total_size = response.info().getheader('Content-Length').strip()
    try:
        total_size = int(total_size)
    except Exception, e:
        print "Total size could not be determined. Error: ", e
        total_size = None
    bytes_so_far = 0

    t0 = time.time()
    while 1:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)

        if not chunk:
            if report_hook:
                sys.stderr.write('\n')
            break

        local_file.write(chunk)
        if report_hook:
            _chunk_report_(bytes_so_far, total_size, t0)

    return


def _get_dataset_dir(dataset_name, data_dir=None):
    """Returns data directory of given dataset

    Parameters
    ----------
    dataset_name: string
        The unique name of the dataset.

    data_dir: string
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data_dir: string
        Path of the given dataset directory.

    Notes
    -----
    This function retrieve the datasets directory (or data directory) using
    the following priority :
    1. the keyword argument data_dir
    2. the environment variable NISL_DATA
    3. "nisl_data" directory into the current working directory
    """
    if not data_dir:
        data_dir = os.getenv("NISL_DATA",  os.path.join(os.getcwd(),
            'nisl_data'))
    data_dir = os.path.join(data_dir, dataset_name)
    return data_dir


def _uncompress_file(file, delete_archive=True):
    """Uncompress files contained in a data_set.

    Parameters
    ----------
    file: string
        path of file to be uncompressed.

    delete_archive: boolean, optional
        Wheteher or not to delete archive once it is uncompressed.
        Default: True

    Notes
    -----
    This handles zip, tar, gzip and bzip files only.
    """
    print 'extracting data from %s...' % file
    data_dir = os.path.dirname(file)
    # We first try to see if it is a zip file
    try:
        if file.endswith('.zip'):
            z = zipfile.Zipfile(file)
            z.extractall(data_dir)
            z.close()
        else:
            tar = tarfile.open(file, "r")
            tar.extractall(path=data_dir)
            tar.close()
        if delete_archive:
            os.remove(file)
        print '   ...done.'
    except Exception as e:
        print 'error: ', e
        raise


def _fetch_file(url, data_dir):
    """Load requested file, downloading it if needed or requested

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    urls: array of strings
        Contains the urls of files to be downloaded.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    files: array of string
        Absolute paths of downloaded files on disk

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded data are
    cleaned.
    """
    # Determine data path
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_name = os.path.basename(url)
    full_name = os.path.join(data_dir, file_name)
    if not os.path.exists(full_name):
        t0 = time.time()
        try:
            # Download data
            print 'Downloading data from %s ...' % url
            req = urllib2.Request(url)
            data = urllib2.urlopen(req)
            local_file = open(full_name, "wb")
            _chunk_read_(data, local_file, report_hook=True)
            dt = time.time() - t0
            print '...done. (%i seconds, %i min)' % (dt, dt/60)
        except urllib2.HTTPError, e:
            print "HTTP Error:", e, url
            return None
        except urllib2.URLError, e:
            print "URL Error:", e, url
            return None
        finally:
            local_file.close()
    return full_name


def _fetch_dataset(dataset_name, urls, data_dir=None, uncompress=True):
    """Load requested dataset, downloading it if needed or requested

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    urls: array of strings
        Contains the urls of files to be downloaded.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    files: array of string
        Absolute paths of downloaded files on disk

    Notes
    -----
    If, for any reason, the download procedure fails, all downloaded data are
    cleaned.
    """
    # Determine data path
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    files = []
    for url in urls:
        full_name = _fetch_file(url, data_dir)
        if not full_name:
            print 'An error occured, abort fetching'
            shutil.rmtree(data_dir)
        if uncompress:
            try:
                _uncompress_file(full_name)
            except Exception:
                # We are giving it a second try, but won't try a third
                # time :)
                print 'archive corrupted, trying to download it again'
                _fetch_file(url, data_dir)
                _uncompress_file(full_name)
        files.append(full_name)

    return files


def _get_dataset(dataset_name, file_names, data_dir=None):
    """Returns absolute paths of a dataset files if exist

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    file_names: array of strings
        File that compose the dataset to be retrieved on the disk.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    files: array of string
        List of dataset files on disk

    Notes
    -----
    If at least one file is missing, an IOError is raised.
    """
    data_dir = _get_dataset_dir(dataset_name, data_dir=data_dir)
    file_paths = []
    for file_name in file_names:
        full_name = os.path.join(data_dir, file_name)
        if not os.path.exists(full_name):
            raise IOError("No such file: '%s'" % full_name)
        file_paths.append(full_name)
    return file_paths


###############################################################################
# Dataset downloading functions

def fetch_star_plus(data_dir=None):
    """Function returning the starplus data, downloading them if needed

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'datas' : a list of 6 numpy arrays representing the data to learn
        'targets' : list
                    targets of the datas
        'masks' : the masks for the datas

    Notes
    -----
    Each element will be of the form :
    PATH/*.npy

    The star plus datasets is composed of n_trials trials.
    Each trial is composed of 13 time units.
    We decided here to average on the time
    /!\ y is not binarized !

    References
    ----------
    Documentation :
    http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/\
            README-data-documentation.txt

    Data :
    http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/
    """

    dataset_files = ['data-starplus-%d-%s.npy' % (i, j) for i in range(0, 6)
            for j in ['mask', 'X', 'y']]
    dataset_dir = _get_dataset_dir("starplus", data_dir=data_dir)

    try:
        _get_dataset("starplus", dataset_files, data_dir=data_dir)
    except IOError:
        file_names = ['data-starplus-0%d-v7.mat' % i for i in
                [4847, 4799, 5710, 4820, 5675, 5680]]
        url1 = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/'
        url2 = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-83/www/'

        url1_files = ["/".join(url1, i) for i in file_names[0:3]]
        url2_files = ["/".join(url2, i) for i in file_names[3:6]]
        urls = url1_files + url2_files

        full_names = _fetch_dataset('starplus', urls, data_dir=data_dir)

        for indice, full_name in enumerate(full_names):
            # Converting data to a more readable format
            print "Converting file %d on 6..." % (indice + 1)
            # General information
            try:
                data = io.loadmat(full_name, struct_as_record=True)
                n_voxels = data['meta']['nvoxels'].flat[0].squeeze()
                n_trials = data['meta']['ntrials'].flat[0].squeeze()
                dim_x = data['meta']['dimx'].flat[0].squeeze()
                dim_y = data['meta']['dimy'].flat[0].squeeze()
                dim_z = data['meta']['dimz'].flat[0].squeeze()

                # Loading X
                X_temp = data['data'][:, 0]

                # Loading y
                y = data['info']
                y = y[0, :]

                # y = np.array([y[i].flat[0]['actionRT'].flat[0]
                y = np.array([y[i].flat[0]['cond'].flat[0]
                              for i in range(n_trials)])

                good_trials = np.where(y > 1)[0]
                n_good_trials = len(good_trials)
                n_times = 16  # 8 seconds

                # sentences
                XS = np.zeros((n_good_trials, dim_x, dim_y, dim_z))
                # pictures
                XP = np.zeros((n_good_trials, dim_x, dim_y, dim_z))
                first_stim = data['info']['firstStimulus']

                # Averaging on the time
                for k, i_trial in enumerate(good_trials):
                    i_first_stim = str(first_stim.flat[i_trial][0])
                    XSk = XS[k]
                    XPk = XP[k]
                    for j in range(n_voxels):
                        # Getting the right coords of the voxels
                        x, y, z = data['meta']['colToCoord'].flat[0][j, :] - 1
                        Xkxyz = X_temp[i_trial][:, j]
                        # Xkxyz -= Xkxyz.mean()  # remove drifts
                        if i_first_stim == 'S':  # sentence
                            XSk[x, y, z] = Xkxyz[:n_times].mean()
                            XPk[x, y, z] = Xkxyz[n_times:2 * n_times].mean()
                        elif i_first_stim == 'P':  # picture
                            XPk[x, y, z] = Xkxyz[:n_times].mean()
                            XSk[x, y, z] = Xkxyz[n_times:2 * n_times].mean()
                        else:
                            raise ValueError('Uknown first_stim : %s'
                                             % first_stim)

                X = np.r_[XP, XS]
                y = np.ones(2 * n_good_trials)
                y[:n_good_trials] = 0

                X = X.astype(np.float)
                y = y.astype(np.float)

                name = "data-starplus-%d-X.npy" % indice
                name = os.path.join(dataset_dir, name)
                np.save(name, X)
                name = "data-starplus-%d-y.npy" % indice
                name = os.path.join(dataset_dir, name)
                np.save(name, y)
                name = "data-starplus-%d-mask.npy" % indice
                name = os.path.join(dataset_dir, name)
                mask = X[0, ...]
                mask = mask.astype(np.bool)
                np.save(name, mask)
                print "...done."

                # Removing the unused data
                os.remove(full_name)
            except Exception, e:
                print "Impossible to convert the file %s:\n %s " % (name, e)
                shutil.rmtree(dataset_dir)
                raise e

    print "...done."

    all_subject = []
    for i in range(0, 6):
        X = np.load(os.path.join(dataset_dir, 'data-starplus-%d-X.npy' % i))
        y = np.load(os.path.join(dataset_dir, 'data-starplus-%d-y.npy' % i))
        mask = np.load(os.path.join(dataset_dir,
            'data-starplus-%d-mask.npy' % i))
        all_subject.append(Bunch(data=X, target=y,
                                 mask=mask.astype(np.bool)))

    return all_subject


def fetch_haxby(data_dir=None):
    """Returns the haxby dataset

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'data' : numpy array : the data to learn
        'target' : numpy array
                    target of the data
        'mask' : the masks for the data
        'session' : the labels for LeaveOneLabelOut cross validation

    References
    ----------
    `Haxby, J., Gobbini, M., Furey, M., Ishai, A., Schouten, J.,
    and Pietrini, P. (2001). Distributed and overlapping representations of
    faces and objects in ventral temporal cortex. Science 293, 2425-2430.`

    Notes
    -----
    PyMVPA provides a tutorial using this dataset :
    http://www.pymvpa.org/tutorial.html

    More informations about its structure :
    http://dev.pymvpa.org/datadb/haxby2001.html
    """

    # definition of dataset files
    file_names = ['attributes.txt', 'bold.nii.gz', 'mask.nii.gz']
    file_names = [os.path.join('pymvpa-exampledata', i) for i in file_names]

    # load the dataset
    try:
        # Try to load the dataset
        files = _get_dataset("haxby2001", file_names, data_dir=data_dir)

    except IOError:
        # If the dataset does not exists, we download it
        url = 'http://www.pymvpa.org/files/pymvpa_exampledata.tar.bz2'
        _fetch_dataset('haxby2001', [url, ], data_dir=data_dir)
        files = _get_dataset("haxby2001", file_names, data_dir=data_dir)

    # preprocess data
    y, session = np.loadtxt(files[0]).astype("int").T
    X = ni.load(files[1]).get_data()
    mask = ni.load(files[2]).get_data().astype(np.bool)

    # Crop a bit. We are copying to loose the reference to the original
    # data
    X = np.copy(X[:, 7:56, 11:52])
    mask = np.copy(mask[:, 7:56, 11:52])

    # return the data
    return Bunch(data=X, target=y, mask=mask, session=session, files=files)


def _fetch_kamitani(data_dir=None):
    """Returns the kamitani dataset

    Parameters
    ----------
    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'data' : numpy array : the data to learn
        'target' : numpy array
                    target of the data
        'mask' : the masks for the data
        'xyz' : index to 3D-coordinate array

    Notes
    -----
    Kamitani dataset cannot be downloaded for the moment because it requires
    registration.

    """

    file_names = ['public_beta_201005015.mat']

    try:
        files = _get_dataset("kamitani", file_names, data_dir=data_dir)
    except IOError:
        url = ''
        tar_name = 'public_beta_20100515.zip'
        urls = ['/'.join(url, tar_name)]
        _fetch_dataset('kamitani', urls, data_dir=data_dir)
        files = _get_dataset("kamitani", file_names, data_dir=data_dir)

    mat = io.loadmat(files[0], struct_as_record=True)

    """
    Matrix content :
    - data : already masked fMRI (20 random and 12 figures, 18064 voxels, 145
             shots)
    - label : picture shown to the patient (20 random and 12 figures, 10x10)
    - roi_name + roi_volInd : ROIs and corresponding coordinates (11x4)
    - volInd : indices of each point of data
    - xyz : mapping of each voxel in 3D-coordinates
    """

    y_random = mat['D']['label'].flat[0]['random'].flat[0].squeeze()
    y_figure = mat['D']['label'].flat[0]['figure'].flat[0].squeeze()
    X_random = mat['D']['data'].flat[0]['random'].flat[0].squeeze()
    X_figure = mat['D']['data'].flat[0]['figure'].flat[0].squeeze()
    roi_name = mat['D']['roi_name'].flat[0]
    roi_volInd = mat['D']['roi_volInd'].flat[0]
    volInd = mat['D']['volInd'].flat[0].squeeze()
    xyz = mat['D']['xyz'].flat[0]
    ijk = xyz / 3 - 0.5 + [[32], [32], [15]]

    return Bunch(files=files, data_random=X_random, data_figure=X_figure,
           target_random=y_random, target_figure=y_figure, roi_name=roi_name,
           roi_volInd=roi_volInd, volInd=volInd, xyz=xyz, ijk=ijk)


def fetch_nyu_rest(n_subjects=None, data_dir=None):
    """Returns the NYU Test Retest dataset

    Parameters
    ----------
    n_subjects: integer optional
        The number of subjects to load. If None is given, all the
        subjects are used.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'data' : numpy array : the data to learn
        'target' : numpy array
                    target of the data
        'mask' : the masks for the data
        'xyz' : index to 3D-coordinate array

    Notes
    ------

    This dataset is composed of 3 sessions of 26 participants (11 males).
    For each session, three sets of data are available:

    - anatomical:

      * anonymized data (defaced thanks to BIRN defacer)
      * skullstripped data (using 3DSkullStrip from AFNI)

    - functional

    For each participant, 3 resting-state scans of 197 continuous EPI functional
    volumes were collected :

    - 39 slices
    - matrix = 64 x 64
    - acquisition voxel size = 3 x 3 x 3 mm

    Sessions 2 and 3 were conducted in a single scan session, 45 min
    apart, and were 5-16 months after Scan 1.

    All details about this dataset can be found here :
    http://cercor.oxfordjournals.org/content/19/10/2209.full

    References
    ----------
    :Documentation:
        http://www.nitrc.org/docman/?group_id=274

    :Download:
        http://www.nitrc.org/frs/?group_id=274

    :Paper to cite:
        `The Resting Brain: Unconstrained yet Reliable
        <http://cercor.oxfordjournals.org/content/19/10/2209>`_
        Z. Shehzad, A.M.C. Kelly, P.T. Reiss, D.G. Gee, K. Gotimer,
        L.Q. Uddin, S.H. Lee, D.S. Margulies, A.K. Roy, B.B. Biswal,
        E. Petkova, F.X. Castellanos and M.P. Milham.

    :Other references:
        * `The oscillating brain: Complex and Reliable
          <http://dx.doi.org/10.1016/j.neuroimage.2009.09.037>`_
          X-N. Zuo, A. Di Martino, C. Kelly, Z. Shehzad, D.G. Gee,
          D.F. Klein, F.X. Castellanos, B.B. Biswal, M.P. Milham

        * `Reliable intrinsic connectivity networks: Test-retest
          evaluation using ICA and dual regression approach
          <http://dx.doi.org/10.1016/j.neuroimage.2009.10.080>`_,
          X-N. Zuo, C. Kelly, J.S. Adelstein, D.F. Klein,
          F.X. Castellanos, M.P. Milham

    """

    file_names = [os.path.join('anat', 'mprage_anonymized.nii.gz'),
                  os.path.join('anat', 'mprage_skullstripped.nii.gz'),
                  os.path.join('func', 'lfo.nii.gz')]

    # Warning : only Session 1 subs for the moment
    sub_names = ['sub05676', 'sub08224', 'sub08889', 'sub09607', 'sub14864',
            'sub18604', 'sub22894', 'sub27641', 'sub33259', 'sub34482',
            'sub36678', 'sub38579', 'sub39529']

    file_names = [os.path.join(sub, f) for sub in sub_names
                  for f in file_names]

    try:
        files = _get_dataset("nyu_rest", file_names, data_dir=data_dir)
    except IOError:
        url = 'http://www.nitrc.org/frs/download.php/'
        tar_prefixes = ['1071']
        tar_names = ['NYU_TRT_session1a.tar.gz']
        """
        tar_prefixes = ['1071', '1072', '1073', '1074', '1075', '1076']
        tar_names = ['NYU_TRT_session1a.tar.gz',
            'NYU_TRT_session1b.tar.gz', 'NYU_TRT_session2a.tar.gz',
            'NYU_TRT_session2b.tar.gz', 'NYU_TRT_session3a.tar.gz',
            'NYU_TRT_session3b.tar.gz']
        """
        tar_full_names = [os.path.join(prefix, name)
                          for prefix, name in zip(tar_prefixes, tar_names)]
        urls = ['/'.join(url, name) for name in tar_full_names]
        _fetch_dataset('nyu_rest', urls, data_dir=data_dir)
        files = _get_dataset("nyu_rest", file_names, data_dir=data_dir)

    anat_anon = []
    anat_skull = []
    func = []

    if n_subjects is None:
        n_subjects = len(sub_names)

    for i in range(n_subjects):
        # We are considering files 3 by 3
        i *= 3
        anat_anon.append(ni.load(files[i]).get_data())
        anat_skull.append(ni.load(files[i + 1]).get_data())
        func.append(ni.load(files[i + 2]).get_data())

    return Bunch(anat_anon=anat_anon, anat_skull=anat_skull, func=func,
                 files=files)
