"""Sphinx Gallery
"""
import os
__version__ = '0.0.6'

def _path_static():
    """Returns path to packaged static files"""
    return os.path.abspath(os.path.dirname(__file__))+'/_static'
