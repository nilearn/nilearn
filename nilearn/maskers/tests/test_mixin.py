import pytest

from nilearn.conftest import _img_3d_rand, _make_surface_img
from nilearn.maskers import (
    MultiNiftiLabelsMasker,
    MultiNiftiMapsMasker,
    MultiNiftiMasker,
    MultiSurfaceLabelsMasker,
    MultiSurfaceMapsMasker,
    MultiSurfaceMasker,
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    SurfaceLabelsMasker,
    SurfaceMapsMasker,
    SurfaceMasker,
)


@pytest.fixture
def masker(request, img_maps, surf_maps_img, img_labels, surf_label_img):
    """Fixture to construct a masker instance with proper input."""
    cls, arg = request.param

    img_generators = {
        "img_maps": img_maps,
        "surf_maps_img": surf_maps_img,
        "img_labels": img_labels,
        "surf_label_img": surf_label_img,
    }
    if arg is None:
        return cls()
    elif isinstance(arg, dict):
        return cls(**arg)
    else:
        img = img_generators[arg]
        return cls(img)


@pytest.mark.parametrize(
    "masker, img_func",
    [
        ((NiftiMasker, None), _img_3d_rand),
        ((SurfaceMasker, None), _make_surface_img),
        ((MultiNiftiMasker, None), _img_3d_rand),
        ((MultiSurfaceMasker, None), _make_surface_img),
        ((NiftiMapsMasker, "img_maps"), _img_3d_rand),
        ((MultiNiftiMapsMasker, "img_maps"), _img_3d_rand),
        ((SurfaceMapsMasker, "surf_maps_img"), _make_surface_img),
        ((MultiSurfaceMapsMasker, "surf_maps_img"), _make_surface_img),
        ((NiftiLabelsMasker, "img_labels"), _img_3d_rand),
        ((MultiNiftiLabelsMasker, "img_labels"), _img_3d_rand),
        ((SurfaceLabelsMasker, "surf_label_img"), _make_surface_img),
        ((MultiSurfaceLabelsMasker, "surf_label_img"), _make_surface_img),
        (
            (NiftiSpheresMasker, {"seeds": [(1, 1, 1)], "radius": 1}),
            _img_3d_rand,
        ),
    ],
    indirect=["masker"],
)
def test_masker_reporting_init(masker, img_func):
    """Test nilearn.maskers._mixin._ReportingMixin on concrete masker
    instances.
    """
    # check estimator at initialization
    assert masker._report_content is not None
    assert masker._report_content["description"] is not None
    assert masker._report_content["warning_message"] is None
    assert masker._has_report_data() is False

    # check report before fit
    masker.generate_report()
    assert masker._report_content["title"] == "Empty Report"

    # check estimator after fit
    input_img = img_func()
    masker.fit(input_img)
    assert hasattr(masker, "_reporting_data")

    masker.generate_report(title="masker report title")

    assert masker._report_content["title"] == "masker report title"
