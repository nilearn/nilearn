"""
The :mod:`nilearn.regions` class module includes region extraction
procedure which operated on a 4D statistical/atlas maps and its functions adapted
to 3D maps.
"""
from .region_extractor import estimate_apply_threshold_to_maps,\
    break_connected_components, RegionExtractor

__all__ = ['estimate_apply_threshold_to_maps', 'break_connected_components', 'RegionExtractor']
