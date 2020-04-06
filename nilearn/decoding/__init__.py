"""
Decoding tools and algorithms.
"""


from nilearn.decoding.searchlight import SearchLight
from nilearn.decoding.space_net import SpaceNetClassifier, SpaceNetRegressor
from nilearn.decoding.decoder import (Decoder, DecoderRegressor,
                                      fREMClassifier, fREMRegressor)


__all__ = ['SearchLight', 'SpaceNetClassifier', 'SpaceNetRegressor', 'Decoder',
           'DecoderRegressor', 'fREMClassifier', 'fREMRegressor']
