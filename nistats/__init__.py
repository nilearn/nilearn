"""
functional MRI module for NeuroImaging in python
--------------------------------------------------

Documentation is available in the docstrings and online at
http://nistats.github.io.

Contents
--------
Nistats aims at

Submodules
---------
datasets                --- Utilities to download NeuroImaging datasets
"""

import gzip

from .version import _check_module_dependencies, __version__

_check_module_dependencies()

__all__ = ['__version__', 'datasets', 'design_matrix']
