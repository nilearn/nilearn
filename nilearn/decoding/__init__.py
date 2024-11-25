"""Decoding tools and algorithms."""

from nilearn.decoding.decoder import (
    Decoder,
    DecoderRegressor,
    FREMClassifier,
    FREMRegressor,
)
from nilearn.decoding.searchlight import SearchLight
from nilearn.decoding.space_net import SpaceNetClassifier, SpaceNetRegressor

__all__ = [
    "Decoder",
    "DecoderRegressor",
    "FREMClassifier",
    "FREMRegressor",
    "SearchLight",
    "SpaceNetClassifier",
    "SpaceNetRegressor",
]
