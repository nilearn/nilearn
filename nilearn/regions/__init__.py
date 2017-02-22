"""
The :mod:`nilearn.regions` class module includes region extraction
procedure on a 4D statistical/atlas maps and its function.
"""
from .region_extractor import (connected_regions, RegionExtractor,
                               connected_label_regions)
from .signal_extraction import (
    img_to_signals_labels, signals_to_img_labels,
    img_to_signals_maps, signals_to_img_maps,
)

__all__ = [
    'connected_regions', 'RegionExtractor',
    'connected_label_regions',
    'img_to_signals_labels', 'signals_to_img_labels',
    'img_to_signals_maps', 'signals_to_img_maps',
]
