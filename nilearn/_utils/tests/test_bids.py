"""Test for _utils.bids module."""

import numpy as np
import pytest

from nilearn._utils import data_gen
from nilearn._utils.bids import (
    check_look_up_table,
    generate_atlas_look_up_table,
)


def test_generate_atlas_look_up_table(shape_3d_default, surf_three_labels_img):
    """Check generation of LUT directly from niimg or surface image."""
    mock_regions = data_gen.generate_labeled_regions(
        shape_3d_default, n_regions=10
    )
    lut = generate_atlas_look_up_table(function="unknown", index=mock_regions)
    check_look_up_table(lut=lut, atlas=mock_regions, strict=True)

    lut = generate_atlas_look_up_table(
        function="unknown", index=surf_three_labels_img
    )
    check_look_up_table(lut=lut, atlas=surf_three_labels_img, strict=True)


def test_generate_atlas_look_up_table_errors():
    with pytest.raises(
        ValueError, match="'index' and 'name' cannot both be None."
    ):
        generate_atlas_look_up_table(function=None, name=None, index=None)

    with pytest.raises(
        TypeError,
        match="must be one of: Niimg-Like, SurfaceIamge, numpy array.",
    ):
        generate_atlas_look_up_table(function=None, name=None, index=[1, 2, 3])

    with pytest.raises(ValueError, match="have different lengths"):
        generate_atlas_look_up_table(
            function=None,
            name=["a", "b"],
            index=np.array([1, 2, 3]),
            strict=True,
        )


def test_check_look_up_table_errors(shape_3d_default):
    mock_regions = data_gen.generate_labeled_regions(
        shape_3d_default, n_regions=10
    )
    lut = generate_atlas_look_up_table(function="unknown", index=mock_regions)

    with pytest.raises(
        ValueError, match="missing from the atlas look-up table"
    ):
        check_look_up_table(
            lut=lut.drop(index=2), atlas=mock_regions, strict=True
        )

    mock_regions_with_missing_labels = data_gen.generate_labeled_regions(
        shape_3d_default, n_regions=8
    )
    with pytest.raises(ValueError, match="missing from the atlas image"):
        check_look_up_table(
            lut=lut, atlas=mock_regions_with_missing_labels, strict=True
        )
