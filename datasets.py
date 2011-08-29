"""File to import StarPlus data
"""

import os
from os.path import exists, join
from os import makedirs, getcwd
from urllib2 import Request, urlopen, URLError, HTTPError
import tarfile


def fetch_star_plus_data():
    """Function returning a list of the locations of the star_plus_data
    files

    Returns
    -------
    String list : list of the location of the data files

    Note
    ----
    Each element will be of the form :
    PATH/file.mat
    so that you just have to import io from scipy
    and then do io.loadmat(location[i])
    where location[i] is the i-th location returned by
    fetch star_plus_data
    """
    data_dir = join(getcwd(), 'nisl_data')
    if not exists(data_dir):
        makedirs(data_dir)
    file_names = ['data-starplus-0%d-v7.mat' % i for i in [4799,
                  4820, 4847,5675, 5680, 5710]]
    url = 'http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/'
    for name in file_names:
        if not exists(join(data_dir, name)):
            data_url = join(url, name)
            try:
                print 'Downloading data from %s ...' % data_url
                req = Request(data_url)
                data = urlopen(req)
                local_file = open(join(data_dir, name), "wb")
                local_file.write(data.read())
                local_file.close()
            except HTTPError, e:
                print "HTTP Error:", e, data_url
            except URLError, e:
                print "URL Error:", e, data_url
            print '...done.'

    file_names = [join(data_dir, i) for i in file_names]
    return file_names


def fetch_haxby_data():
    """Returns the directory where the data are stored

    Returns
    -------
    String
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
    return file_names
