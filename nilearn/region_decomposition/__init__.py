"""
The :mod:`nilearn.region_decomposition` class module includes region extraction
procedure which operated on a 4D statistical/atlas maps and its functions adapted
to 3D maps.
"""
from .region_extractor import apply_threshold_to_maps, extract_regions, \
    RegionExtractor

__all__ = ['apply_threshold_to_maps', 'extract_regions', 'RegionExtractor']
