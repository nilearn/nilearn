from nilearn.maskers import SurfaceMapsMasker


def test_surface_maps_masker_fit(surf_maps_img, surf_img, surf_mask):
    masker = SurfaceMapsMasker(surf_maps_img, surf_mask())
    masker.fit()
    X = masker.transform(surf_img(50))
    assert X.shape == (50, 4)
