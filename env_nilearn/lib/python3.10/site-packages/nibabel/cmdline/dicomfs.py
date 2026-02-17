#!python
# emacs: -*- mode: python-mode; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See COPYING file distributed along with the NiBabel package for the
#   copyright and license terms.
#
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
# Copyright (C) 2011 Christian Haselgrove

import errno
import locale
import logging
import os
import stat
import sys
import time


class dummy_fuse:
    """Dummy fuse "module" so that nose does not blow during doctests"""

    Fuse = object


try:
    import fuse  # type: ignore[import]

    uid = os.getuid()
    gid = os.getgid()
except ImportError:
    fuse = dummy_fuse

from optparse import Option, OptionParser

import nibabel as nib
import nibabel.dft as dft

encoding = locale.getlocale()[1]

fuse.fuse_python_api = (0, 2)

logger = logging.getLogger('nibabel.dft')


class FileHandle:
    def __init__(self, fno):
        self.fno = fno
        self.keep_cache = False
        self.direct_io = False

    def __str__(self):
        return f'FileHandle({self.fno})'


class DICOMFS(fuse.Fuse):
    def __init__(self, *args, **kwargs):
        if fuse is dummy_fuse:
            raise RuntimeError('fuse module is not available, install it to use DICOMFS')
        self.followlinks = kwargs.pop('followlinks', False)
        self.dicom_path = kwargs.pop('dicom_path', None)
        fuse.Fuse.__init__(self, *args, **kwargs)
        self.fhs = {}

    def get_paths(self):
        paths = {}
        for study in dft.get_studies(self.dicom_path, self.followlinks):
            pd = paths.setdefault(study.patient_name_or_uid(), {})
            patient_info = 'patient information\n'
            patient_info += f'name: {study.patient_name}\n'
            patient_info += f'ID: {study.patient_id}\n'
            patient_info += f'birth date: {study.patient_birth_date}\n'
            patient_info += f'sex: {study.patient_sex}\n'
            pd['INFO'] = patient_info.encode('ascii', 'replace')
            study_datetime = f'{study.date}_{study.time}'
            study_info = 'study info\n'
            study_info += f'UID: {study.uid}\n'
            study_info += f'date: {study.date}\n'
            study_info += f'time: {study.time}\n'
            study_info += f'comments: {study.comments}\n'
            d = {'INFO': study_info.encode('ascii', 'replace')}
            for series in study.series:
                series_info = 'series info\n'
                series_info += f'UID: {series.uid}\n'
                series_info += f'number: {series.number}\n'
                series_info += f'description: {series.description}\n'
                series_info += f'rows: {series.rows}\n'
                series_info += f'columns: {series.columns}\n'
                series_info += f'bits allocated: {series.bits_allocated}\n'
                series_info += f'bits stored: {series.bits_stored}\n'
                series_info += f'storage instances: {len(series.storage_instances)}\n'
                d[series.number] = {
                    'INFO': series_info.encode('ascii', 'replace'),
                    f'{series.number}.nii': (series.nifti_size, series.as_nifti),
                    f'{series.number}.png': (series.png_size, series.as_png),
                }
            pd[study_datetime] = d
        return paths

    def match_path(self, path):
        wd = self.get_paths()
        if path == '/':
            logger.debug('return root')
            return wd
        for part in path.lstrip('/').split('/'):
            logger.debug(f'path:{path} part:{part}')
            if part not in wd:
                return None
            wd = wd[part]
        logger.debug('return')
        return wd

    def readdir(self, path, fh):
        logger.info(f'readdir {path}')
        matched_path = self.match_path(path)
        if matched_path is None:
            return -errno.ENOENT
        logger.debug(f'matched {matched_path}')
        fnames = [k.encode('ascii', 'replace') for k in matched_path.keys()]
        fnames.extend(('.', '..'))
        return [fuse.Direntry(f) for f in fnames]

    def getattr(self, path):
        logger.debug(f'getattr {path}')
        matched_path = self.match_path(path)
        logger.debug(f'matched: {matched_path}')
        now = time.time()
        st = fuse.Stat()
        if isinstance(matched_path, dict):
            st.st_mode = stat.S_IFDIR | 0o755
            st.st_ctime = now
            st.st_mtime = now
            st.st_atime = now
            st.st_uid = uid
            st.st_gid = gid
            st.st_nlink = len(matched_path)
            return st
        if isinstance(matched_path, str):
            st.st_mode = stat.S_IFREG | 0o644
            st.st_ctime = now
            st.st_mtime = now
            st.st_atime = now
            st.st_uid = uid
            st.st_gid = gid
            st.st_size = len(matched_path)
            st.st_nlink = 1
            return st
        if isinstance(matched_path, tuple):
            st.st_mode = stat.S_IFREG | 0o644
            st.st_ctime = now
            st.st_mtime = now
            st.st_atime = now
            st.st_uid = uid
            st.st_gid = gid
            st.st_size = matched_path[0]()
            st.st_nlink = 1
            return st
        return -errno.ENOENT

    def open(self, path, flags):
        logger.debug(f'open {path}')
        matched_path = self.match_path(path)
        if matched_path is None:
            return -errno.ENOENT
        for i in range(1, 10):
            if i not in self.fhs:
                if isinstance(matched_path, str):
                    self.fhs[i] = matched_path
                elif isinstance(matched_path, tuple):
                    self.fhs[i] = matched_path[1]()
                else:
                    return -errno.EFTYPE
                return FileHandle(i)
        return -errno.ENFILE

    # not done
    def read(self, path, size, offset, fh):
        logger.debug('read')
        logger.debug(path)
        logger.debug(size)
        logger.debug(offset)
        logger.debug(fh)
        return self.fhs[fh.fno][offset : offset + size]

    def release(self, path, flags, fh):
        logger.debug('release')
        logger.debug(path)
        logger.debug(fh)
        del self.fhs[fh.fno]


def get_opt_parser():
    # use module docstring for help output
    p = OptionParser(
        usage=f'{os.path.basename(sys.argv[0])} [OPTIONS] <DIRECTORY CONTAINING DICOMSs> <mount point>',
        version='%prog ' + nib.__version__,
    )

    p.add_options(
        [
            Option(
                '-v',
                '--verbose',
                action='count',
                dest='verbose',
                default=0,
                help='make noise.  Could be specified multiple times',
            ),
        ]
    )

    p.add_options(
        [
            Option(
                '-L',
                '--follow-links',
                action='store_true',
                dest='followlinks',
                default=False,
                help='Follow symbolic links in DICOM directory',
            ),
        ]
    )
    return p


def main(args=None):
    parser = get_opt_parser()
    (opts, files) = parser.parse_args(args=args)

    if opts.verbose:
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.setLevel(opts.verbose > 1 and logging.DEBUG or logging.INFO)

    if len(files) != 2:
        sys.stderr.write(f'Please provide two arguments:\n{parser.usage}\n')
        sys.exit(1)

    fs = DICOMFS(
        dash_s_do='setsingle', followlinks=opts.followlinks, dicom_path=files[0].decode(encoding)
    )
    fs.parse(['-f', '-s', files[1]])
    try:
        fs.main()
    except fuse.FuseError:
        # fuse prints the error message
        sys.exit(1)

    sys.exit(0)
