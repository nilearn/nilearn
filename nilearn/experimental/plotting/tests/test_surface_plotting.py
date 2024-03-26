from nilearn.experimental.plotting import (
    plot_surf,
    plot_surf_roi,
    plot_surf_stat_map,
)


def test_plot_surf(mini_surface_img):
    """Smoke test."""
    plot_surf(surf_map=mini_surface_img)
    plot_surf(surf_map=mini_surface_img, bg_map=mini_surface_img)


def test_plot_surf_stat_map(mini_surface_img):
    """Smoke test."""
    plot_surf_stat_map(stat_map=mini_surface_img)
    plot_surf_stat_map(stat_map=mini_surface_img, bg_map=mini_surface_img)


def test_plot_surf_roi(mini_surface_img):
    """Smoke test."""
    plot_surf_roi(roi_map=mini_surface_img)
    plot_surf_roi(roi_map=mini_surface_img, bg_map=mini_surface_img)
