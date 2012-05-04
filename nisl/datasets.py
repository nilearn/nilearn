"""File to import NeuroImaging datasets
"""

import os
import urllib2
import tarfile
import sys

import numpy as np
from scipy import io
from sklearn.datasets.base import Bunch

import nibabel as ni


def chunk_report_(bytes_so_far, total_size=None):
    """Show downloading percentage

    Parameters
    ----------
    bytes_so_far: integer
        Number of downloaded bytes

    total_size: integer, optional
        Total size of the file. If not given, a question mark will be showed
        instead of it. Default: None
    """
    if total_size:
        percent = float(bytes_so_far) / total_size
        percent = round(percent * 100, 2)
        sys.stdout.write("Downloaded %d of %d bytes (%0.2f%%)\r" %
            (bytes_so_far, total_size, percent))
    else:
        sys.stdout.write("Downloaded %d of ? bytes\r" % (bytes_so_far))


def chunk_read_(response, chunk_size=8192, report_hook=None):
    """Download a file chunk by chunk and show advancement

    Parameters
    ----------
    response: urllib.addinfourl
        Response to the download request in order to get file size

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
    data = []

    while 1:
        chunk = response.read(chunk_size)
        bytes_so_far += len(chunk)

        if not chunk:
            if report_hook:
                sys.stdout.write('\n')
            break

        data += chunk
        if report_hook:
            chunk_report_(bytes_so_far, total_size)

    return "".join(data)


def get_dataset_dir(dataset_name, data_dir=None):
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


def clean_dataset(dataset_name, data_dir=None):
    """Erase the directory of a given dataset

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None
    """
    data_dir = get_dataset_dir(dataset_name, data_dir=data_dir)
    os.removedirs(data_dir)


def uncompress_dataset(dataset_name, files, data_dir=None,
        delete_archive=True):
    """Uncompress files contained in a data_set.

    Parameters
    ----------
    dataset_name: string
        Unique dataset name

    files: array of strings
        Contains the names of files to be uncompressed.

    data_dir: string, optional
        Path of the data directory. Used to force data storage in a specified
        location. Default: None

    delete_archive: boolean, optional
        Wheteher or not to delete archive once it is uncompressed.
        Default: True

    Notes
    -----
    This handles tar, gzip and bzip files only.
    """
    data_dir = get_dataset_dir(dataset_name, data_dir=data_dir)
    for file in files:
        full_name = os.path.join(data_dir, file)
        print 'extracting data from %s...' % full_name
        tar = tarfile.open(full_name, "r")
        tar.extractall(path=data_dir)
        if delete_archive:
            os.remove(full_name)
        print '   ...done.'


def fetch_dataset(dataset_name, urls, data_dir=None,
        force_download=False):
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

    force_download: boolean, optional
        Wheteher or not to force download of data (in case of data corruption
        for example). Default: False

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
    data_dir = get_dataset_dir(dataset_name, data_dir=data_dir)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    files = []
    for url in urls:
        file_name = os.path.basename(url)
        full_name = os.path.join(data_dir, file_name)
        if (not os.path.exists(full_name)) or force_download:
            # Download data
            try:
                print 'Downloading data from %s ...' % url
                req = urllib2.Request(url)
                data = urllib2.urlopen(req)
                chunks = chunk_read_(data, report_hook=True)
                local_file = open(full_name, "wb")
                local_file.write(chunks)
                local_file.close()
                print '...done.'
            except urllib2.HTTPError, e:
                print "HTTP Error:", e, url
                os.removedirs(data_dir)
                return
            except urllib2.URLError, e:
                print "URL Error:", e, url
                os.removedirs(data_dir)
                return
        files.append(full_name)

    return files


def get_dataset(dataset_name, file_names, data_dir=None):
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
    data_dir = get_dataset_dir(dataset_name, data_dir=data_dir)
    file_paths = []
    for file_name in file_names:
        full_name = os.path.join(data_dir, file_name)
        if not os.path.exists(full_name):
            raise IOError("No such file: '%s'" % full_name)
        file_paths.append(full_name)
    return file_paths


def fetch_star_plus_data(data_dir=None, force_download=False):
    """Function returning the starplus data, downloading them if needed

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'datas' : a list of 6 numpy arrays representing the data to learn
        'targets' : list
                    targets of the datas
        'masks' : the masks for the datas

    Note
    ----
    Each element will be of the form :
    PATH/*.npy

    The star plus datasets is composed of n_trials trials.
    Each trial is composed of 13 time units.
    We decided here to average on the time
    /!\ y is not binarized !

    Reference
    ---------
    Documentation :
    http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/\
            README-data-documentation.txt

    Data :
    http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/
    """

    dataset_files = ['data-starplus-%d-%s.npy' % (i, j) for i in range(0, 6)
            for j in ['mask', 'X', 'y']]
    dataset_dir = get_dataset_dir("starplus", data_dir=data_dir)

    try:
        get_dataset("starplus", dataset_files, data_dir=data_dir)
    except IOError:
        file_names = ['data-starplus-0%d-v7.mat' % i for i in
                [4847, 4799, 5710, 4820, 5675, 5680]]
        url1 = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/'
        url2 = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-83/www/'

        url1_files = [os.path.join(url1, i) for i in file_names[0:3]]
        url2_files = [os.path.join(url2, i) for i in file_names[3:6]]
        urls = url1_files + url2_files

        full_names = fetch_dataset('starplus', urls, data_dir=data_dir,
                force_download=force_download)

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
                os.removedirs(dataset_dir)
                raise e

    print "...done."

    all_subject = []
    for i in range(0, 6):
        X = np.load(os.path.join(dataset_dir, 'data-starplus-%d-X.npy' % i))
        y = np.load(os.path.join(dataset_dir, 'data-starplus-%d-y.npy' % i))
        mask = np.load(os.path.join(dataset_dir,
            'data-starplus-%d-mask.npy' % i))
        all_subject.append(Bunch(data=X, target=y, mask=mask))

    return all_subject


def fetch_haxby_data(data_dir=None, force_download=False):
    """Returns the haxby datas

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'data' : numpy array : the data to learn
        'target' : numpy array
                    target of the data
        'mask' : the masks for the data
        'session' : the labels for LeaveOneLabelOut cross validation
    """

    file_names = ['attributes.txt', 'bold.nii.gz', 'mask.nii.gz']
    file_names = [os.path.join('pymvpa-exampledata', i) for i in file_names]

    try:
        files = get_dataset("haxby2001", file_names, data_dir=data_dir)
    except IOError:
        url = 'http://www.pymvpa.org/files'
        tar_name = 'pymvpa_exampledata.tar.bz2'
        urls = [os.path.join(url, tar_name)]
        fetch_dataset('haxby2001', urls, data_dir=data_dir,
                force_download=force_download)
        uncompress_dataset('haxby2001', [tar_name], data_dir=data_dir)
        files = get_dataset("haxby2001", file_names, data_dir=data_dir)

    y, session = np.loadtxt(files[0]).astype("int").T
    X = ni.load(files[1]).get_data()
    mask = ni.load(files[2]).get_data()

    return Bunch(data=X, target=y, mask=mask, session=session)


def fetch_kamitani_data(data_dir=None, force_download=False):
    """Returns the kamitani dataset

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'data' : numpy array : the data to learn
        'target' : numpy array
                    target of the data
        'mask' : the masks for the data
        'xyz' : index to 3D-coordinate array
    """

    file_names = ['public_beta_201005015.mat']

    try:
        files = get_dataset("kamitani", file_names, data_dir=data_dir)
    except IOError:
        url = ''
        tar_name = 'public_beta_20100515.zip'
        urls = [os.path.join(url, tar_name)]
        fetch_dataset('kamitani', urls, data_dir=data_dir,
                force_download=force_download)
        uncompress_dataset('kamitani', [tar_name], data_dir=data_dir)
        files = get_dataset("kamitani", file_names, data_dir=data_dir)

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
    # Convert MNI to convenient 3D coordinates
    xyz = xyz / 3 - 0.5 + [[32], [32], [15]]

    return Bunch(data_random=X_random, data_figure=X_figure,
           target_random=y_random, target_figure=y_figure,
           roi_name=roi_name, roi_volInd=roi_volInd, volInd=volInd, xyz=xyz)
