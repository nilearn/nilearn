"""
Compatibility layer for Python 3/Python 2 single codebase
"""
import sys


if sys.version_info[0] == 3:
    import pickle
    import io
    import urllib

    _basestring = str
    cPickle = pickle
    StringIO = io.StringIO
    BytesIO = io.BytesIO
    _urllib = urllib
else:
    import cPickle
    import StringIO
    import urllib
    import urllib2
    import urlparse
    import types

    _basestring = basestring
    cPickle = cPickle
    StringIO = BytesIO = StringIO.StringIO

    class _module_lookup(object):
        modules = [urlparse, urllib2, urllib]

        def __getattr__(self, name):
            for module in self.modules:
                if hasattr(module, name):
                    attr = getattr(module, name)
                    if not isinstance(attr, types.ModuleType):
                        return attr
            raise NotImplemented('This function has not be imported properly')

    module_lookup = _module_lookup()

    class _urllib():
        request = module_lookup
        error = module_lookup
        parse = module_lookup
