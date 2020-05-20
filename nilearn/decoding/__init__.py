"""
Decoding tools and algorithms.
"""

from nilearn.decoding.searchlight import SearchLight
from nilearn.decoding.surf_searchlight import SurfSearchLight
from nilearn.decoding.space_net import SpaceNetClassifier, SpaceNetRegressor
from nilearn.decoding.decoder import (Decoder, DecoderRegressor,
                                      fREMClassifier, fREMRegressor)


__all__ = ['SearchLight', 'SurfSearchLight', 'SpaceNetClassifier', 'SpaceNetRegressor',
           'Decoder', 'DecoderRegressor', 'fREMClassifier', 'fREMRegressor']