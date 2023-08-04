"""The :mod:`nilearn.interfaces.bids` module includes tools to work with \
:term:`BIDS` format data."""
from .glm import save_glm_to_bids
from .query import get_bids_files, parse_bids_filename

__all__ = ["save_glm_to_bids", "get_bids_files", "parse_bids_filename"]
