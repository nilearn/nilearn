"""
The :mod:`nilearn.regions` class module includes region extraction
procedure on a 4D statistical/atlas maps and its function.
"""
from .region_extractor import connected_regions, RegionExtractor

__all__ = ['connected_regions', 'RegionExtractor']
