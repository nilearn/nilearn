"""File to import StarPlus data
"""

import os
from os.path import exists, join
from os import makedirs, getcwd
from urllib2 import Request, urlopen, URLError, HTTPError
import tarfile
import numpy as np
from scipy.io import loadmat
import nibabel as ni


class Bunch(dict):
    """Container object for datasets: dictionary-like object that
       exposes its keys as attributes."""

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def fetch_star_plus_data():
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

    # If the directory for the data doesn't exists we create it
    data_dir = join(getcwd(), 'nisl_data')
    if not exists(data_dir):
        makedirs(data_dir)

    file_names = ['data-starplus-0%d-v7.mat' % i for i in [4847,
                  4799, 5710, 4820, 5675, 5680]]
    url1 = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/'
    url2 = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-83/www/'
    full_names = [join(data_dir, name) for name in file_names]

    for indice, name in enumerate(full_names):
        print "dealing file : ", name
        if not exists(join(data_dir, "data-starplus-%d-X.npy" %indice))\
           or not exists(join(data_dir, "data-starplus-%d-y.npy" %indice)):

            # Retrieving the .mat data and saving it if needed
            if not exists(name):
                if indice >= 3:
                    url = url2
                else:
                    url = url1

                data_url = join(url, file_names[indice])
                try:
                    print 'Downloading data from %s ...' % data_url
                    req = Request(data_url)
                    data = urlopen(req)
                    local_file = open(name, "wb")
                    local_file.write(data.read())
                    local_file.close()
                except HTTPError, e:
                    print "HTTP Error:", e, data_url
                except URLError, e:
                    print "URL Error:", e, data_url
                print '...done.'

            # Converting data to a more readable format
            print "Converting file %d on 6..." % (indice+1)
            # General information
            data = loadmat(name)
            n_voxels = data['meta'][0][0].nvoxels[0][0]
            n_trials = data['meta'][0][0].ntrials[0][0]
            dim_x = data['meta'][0][0].dimx[0][0]
            dim_y = data['meta'][0][0].dimy[0][0]
            dim_z = data['meta'][0][0].dimz[0][0]
            # Loading X
            X_temp = data['data']
            X_temp = X_temp[:, 0]
            X = np.zeros((n_trials, dim_x, dim_y, dim_z))
            # Averaging on the time
            for i in range(n_trials):
                for j in range(n_voxels):
                    # Getting the right coords of the voxels
                    coords = data['meta'][0][0].colToCoord[j, :]
                    X[i, coords[0]-1, coords[1]-1, coords[2]-1] =\
                            X_temp[i][:, j].mean()
            # Removing the unused data
            os.remove(name)

            # Loading y
            y = data['info']
            y = y[0, :]
            y = np.array([y[i].actionRT[0][0] for i in range(n_trials)])
            X = X.astype(np.float)
            y = y.astype(np.float)
            name = "data-starplus-%d-X.npy" % indice
            name = join(data_dir, name)
            np.save(name, X)
            name = "data-starplus-%d-y.npy" % indice
            name = join(data_dir, name)
            np.save(name, y)
            name = "data-starplus-%d-mask.npy" % indice
            name = join(data_dir, name)
            mask = X[0, ...]
            mask = mask.astype(np.bool)
            np.save(name, mask)

            print "...done."
    print "...done."

    Xs = [np.load(join(data_dir, 'data-starplus-%d-X.npy' % i))\
            for i in range(6)]
    ys = [np.load(join(data_dir, 'data-starplus-%d-y.npy' % i))\
            for i in range(6)]
    masks = [np.load(join(data_dir, 'data-starplus-%d-mask.npy' % i))\
            for i in range(6)]

    return Bunch(datas=Xs, targets=ys,
            masks=masks)


def fetch_haxby_data():
    """Returns the haxby datas

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interest attributes are :
        'data' : numpy array : the data to learn
        'target' : numpy array
                    target of the data
        'mask' : the masks for the data
    """
    data_dir = join(getcwd(), 'nisl_data')
    if not exists(data_dir):
        makedirs(data_dir)
    url = 'http://www.pymvpa.org/files/pymvpa_exampledata.tar.bz2'
    file_names = ['attributes.txt', 'bold.nii.gz', 'mask.nii.gz']
    file_names = [join('pymvpa-exampledata', i) for i in file_names]
    download = False
    for name in file_names:
        # if one of those files doesn't exist, we download the archive
        if not exists(join(data_dir, name)):
            download = True

    if download:
        try:
            print 'Downloading data from %s ...' % url
            data = urlopen(url)
            temp_name = join(data_dir, 'temp.tar.bz2')
            if not exists(temp_name):
                local_file = open(temp_name, "wb")
                local_file.write(data.read())
                local_file.close()
        except HTTPError, e:
            print "HTTP Error:", e, url
        except URLError, e:
            print "URL Error:", e, url
        print '...done.'
        print 'extracting data from %s...' % temp_name
        tar = tarfile.open(temp_name, "r:bz2")
        for name in file_names:
            print '   extracting %s...' % name
            tar.extract(name, path=data_dir)
            print '   ...done.'
        os.remove(temp_name)

    file_names = [join(data_dir, i) for i in file_names]

    y, session = np.loadtxt(file_names[0]).astype("int").T
    X = ni.load(file_names[1]).get_data()
    mask = ni.load(file_names[2]).get_data()

    return Bunch(data=X, target=y, mask=mask)

