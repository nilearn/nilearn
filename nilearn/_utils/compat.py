"""
Compatibility layer for Python 3/Python 2 single codebase
"""
import sys
import hashlib

from distutils.version import LooseVersion

import nibabel


if sys.version_info[0] == 3:
    import pickle
    import io
    import urllib

    _basestring = str
    cPickle = pickle
    StringIO = io.StringIO
    BytesIO = io.BytesIO
    _urllib = urllib
    izip = zip

    def md5_hash(string):
        m = hashlib.md5()
        m.update(string.encode('utf-8'))
        return m.hexdigest()
else:
    import cPickle
    import StringIO
    import urllib
    import urllib2
    import urlparse
    import types
    import itertools

    _basestring = basestring
    cPickle = cPickle
    StringIO = BytesIO = StringIO.StringIO
    izip = itertools.izip

    class _module_lookup(object):
        modules = [urlparse, urllib2, urllib]

        def __getattr__(self, name):
            for module in self.modules:
                if hasattr(module, name):
                    attr = getattr(module, name)
                    if not isinstance(attr, types.ModuleType):
                        return attr
            raise NotImplementedError(
                'This function has not been imported properly')

    module_lookup = _module_lookup()

    class _urllib():
        request = module_lookup
        error = module_lookup
        parse = module_lookup

    def md5_hash(string):
        m = hashlib.md5()
        m.update(string)
        return m.hexdigest()


if LooseVersion(nibabel.__version__) >= LooseVersion('2.0.0'):
    def get_affine(img):
        return img.affine

    def get_header(img):
        return img.header
else:
    def get_affine(img):
        return img.get_affine()

    def get_header(img):
        return img.get_header()
