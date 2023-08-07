import numpy as np

from nilearn.experimental.surface import (
    SurfaceImage,
    SurfaceMasker,
)


def test_f(make_mini_img):
    mini_img = make_mini_img((2, 3))
    print(mini_img.mesh)
    print(mini_img.data)
    print(mini_img)


def test_surface_masker(pial_surface_mesh):
    """Test fit_transform method"""
    masker = SurfaceMasker()
    data_array = np.arange(1, 5 * 10242 + 1).reshape((5, 10242))
    surf_img = SurfaceImage(
        mesh=pial_surface_mesh,
        data={"left_hemisphere": data_array, "right_hemisphere": data_array},
    )
    masked_data = masker.fit_transform(surf_img)
    assert masked_data.ndim == 2
    assert masked_data.shape == (5, 20484)
