"""
Compatibility layer for Python 3/Python 2 single codebase
"""
import hashlib
import sklearn


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
