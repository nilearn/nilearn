"""
Compatibility layer for Python 3/Python 2 single codebase
"""
import hashlib


def md5_hash(string):
    m = hashlib.md5()
    m.update(string.encode('utf-8'))
    return m.hexdigest()
