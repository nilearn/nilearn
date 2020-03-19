"""
Compatibility layer for Python 3/Python 2 single codebase
"""
import hashlib
import sklearn


import pickle
import io
import urllib
from base64 import encodebytes

_encodebytes = encodebytes
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


if sklearn.__version__ < '0.21':
    from sklearn.externals import joblib
else:
    import joblib

Memory = joblib.Memory
Parallel = joblib.Parallel
hash = joblib.hash
delayed = joblib.delayed
cpu_count = joblib.cpu_count
