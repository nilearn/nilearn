"""File to import StarPlus data
"""

import os
import urllib2
import tarfile

import numpy as np
from scipy import io
from sklearn.datasets.base import Bunch

import nibabel as ni


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
    data_dir = os.path.join(os.getcwd(), 'nisl_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_names = ['data-starplus-0%d-v7.mat' % i for i in [4847,
                  4799, 5710, 4820, 5675, 5680]]
    url1 = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/'
    url2 = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-83/www/'
    full_names = [os.path.join(data_dir, name) for name in file_names]

    success_indices = []
    for indice, full_name in enumerate(full_names):
        print "Fetching file : %s" % full_name
        if (os.path.exists(os.path.join(data_dir,
                        "data-starplus-%d-X.npy" % indice))
                and os.path.exists(os.path.join(data_dir,
                            "data-starplus-%d-y.npy" % indice))):
            success_indices.append(indice)
        else:
            # Retrieving the .mat data and saving it if needed
            if not os.path.exists(full_name):
                if indice >= 3:
                    url = url2
                else:
                    url = url1

                data_url = os.path.join(url, file_names[indice])
                try:
                    print 'Downloading data from %s ...' % data_url
                    req = urllib2.Request(data_url)
                    data = urllib2.urlopen(req)
                    local_file = open(full_name, "wb")
                    local_file.write(data.read())
                    local_file.close()
                except urllib2.HTTPError, e:
                    print "HTTP Error: %s, %s" % (e, data_url)
                except urllib2.URLError, e:
                    print "URL Error: %s, %s" % (e, data_url)
                print '...done.'

            # Converting data to a more readable format
            print "Converting file %d on 6..." % (indice + 1)
            # General information
            try:
                data = io.loadmat(full_name)
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
                name = os.path.join(data_dir, name)
                np.save(name, X)
                name = "data-starplus-%d-y.npy" % indice
                name = os.path.join(data_dir, name)
                np.save(name, y)
                name = "data-starplus-%d-mask.npy" % indice
                name = os.path.join(data_dir, name)
                mask = X[0, ...]
                mask = mask.astype(np.bool)
                np.save(name, mask)
                print "...done."
                success_indices.append(indice)

                # Removing the unused data
                os.remove(full_name)
            except Exception, e:
                print "Impossible to convert the file %s:\n %s " % (name, e)

    print "...done."

    all_subject = list()
    for i in success_indices:
        X = np.load(os.path.join(data_dir, 'data-starplus-%d-X.npy' % i))
        y = np.load(os.path.join(data_dir, 'data-starplus-%d-y.npy' % i))
        mask = np.load(os.path.join(data_dir, 'data-starplus-%d-mask.npy' % i))
        all_subject.append(Bunch(data=X, target=y, mask=mask))

    return all_subject


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
        'session' : the labels for LeaveOneLabelOut cross validation
    """
    data_dir = os.path.join(os.getcwd(), 'nisl_data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    url = 'http://www.pymvpa.org/files/pymvpa_exampledata.tar.bz2'
    file_names = ['attributes.txt', 'bold.nii.gz', 'mask.nii.gz']
    file_names = [os.path.join('pymvpa-exampledata', i) for i in file_names]
    download = False
    for name in file_names:
        # if one of those files doesn't exist, we download the archive
        if not os.path.exists(os.path.join(data_dir, name)):
            download = True

    if download:
        try:
            print 'Downloading data from %s ...' % url
            data = urllib2.urlopen(url)
            temp_name = os.path.join(data_dir, 'temp.tar.bz2')
            if not os.path.exists(temp_name):
                local_file = open(temp_name, "wb")
                local_file.write(data.read())
                local_file.close()
        except urllib2.HTTPError, e:
            print "HTTP Error:", e, url
        except urllib2.URLError, e:
            print "URL Error:", e, url
        print '...done.'
        print 'extracting data from %s...' % temp_name
        tar = tarfile.open(temp_name, "r:bz2")
        for name in file_names:
            print '   extracting %s...' % name
            tar.extract(name, path=data_dir)
            print '   ...done.'
        os.remove(temp_name)

    file_names = [os.path.join(data_dir, i) for i in file_names]

    y, session = np.loadtxt(file_names[0]).astype("int").T
    X = ni.load(file_names[1]).get_data()
    mask = ni.load(file_names[2]).get_data()

    return Bunch(data=X, target=y, mask=mask, session=session)
