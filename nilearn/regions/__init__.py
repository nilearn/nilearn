"""The :mod:`nilearn.regions` class module includes region extraction \
procedure on a 4D statistical/atlas maps and its function.
"""

from nilearn.regions.hierarchical_kmeans_clustering import HierarchicalKMeans
from nilearn.regions.parcellations import Parcellations
from nilearn.regions.region_extractor import (
    RegionExtractor,
    connected_label_regions,
    connected_regions,
)
from nilearn.regions.rena_clustering import (
    ReNA,
    recursive_neighbor_agglomeration,
)
from nilearn.regions.signal_extraction import (
    img_to_signals_labels,
    img_to_signals_maps,
    signals_to_img_labels,
    signals_to_img_maps,
)

__all__ = [
    "HierarchicalKMeans",
    "Parcellations",
    "ReNA",
    "RegionExtractor",
    "connected_label_regions",
    "connected_regions",
    "img_to_signals_labels",
    "img_to_signals_maps",
    "recursive_neighbor_agglomeration",
    "signals_to_img_labels",
    "signals_to_img_maps",
]
