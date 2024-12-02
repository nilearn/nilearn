from nilearn.maskers import SurfaceMapsMasker


def test_surface_maps_masker_fit_transform(surf_maps_img, surf_img, surf_mask):
    masker = SurfaceMapsMasker(surf_maps_img, surf_mask()).fit()
    region_signals = masker.transform(surf_img(50))
    assert region_signals.shape == (50, 4)


def test_surface_maps_masker_inverse_transform(
    surf_maps_img, surf_img, surf_mask
):
    masker = SurfaceMapsMasker(surf_maps_img, surf_mask()).fit()
    region_signals = masker.fit_transform(surf_img(50))
    X_inverse_transformed = masker.inverse_transform(region_signals)
    assert X_inverse_transformed.shape == surf_img(50).shape
