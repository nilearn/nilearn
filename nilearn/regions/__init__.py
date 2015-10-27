"""
The :mod:`nilearn.regions` class module includes region extraction
procedure which operated on a 4D statistical/atlas maps and its functions adapted
to 3D maps.
"""
from .region_extractor import foreground_extraction,\
    connected_component_extraction, RegionExtractor

__all__ = ['foreground_extraction', 'connected_component_extraction', 'RegionExtractor']
