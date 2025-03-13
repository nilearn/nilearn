import pytest

from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed
from nilearn.plotting.surface._backend import get_surface_backend
from nilearn.plotting.surface.surf_plotting import (
    plot_surf,
    plot_surf_stat_map,
    plot_surf_roi,
)


@pytest.mark.skipif(is_matplotlib_installed(), reason="")
def test_get_surface_backend_matplotlib_not_installed():
    with pytest.raises(ImportError, match="Using engine"):
        get_surface_backend("matplotlib")


@pytest.mark.skipif(is_plotly_installed(), reason="")
def test_get_surface_backend_plotly_not_installed():
    with pytest.raises(ImportError, match="Using engine"):
        get_surface_backend("plotly")


class BaseTestSurfaceBackend:

    def test_get_surface_backend(self, engine):
        assert get_surface_backend(engine) is not None

    def test_plot_surf(self, in_memory_mesh, engine):
        plot_surf(in_memory_mesh, engine=engine) is not None

    def test_plot_surf_stat_map(self, in_memory_mesh, bg_map, engine):
        plot_surf_stat_map(in_memory_mesh, stat_map=bg_map, engine=engine) is not None

    def test_plot_surf_roi(self, surface_image_roi, engine):
        plot_surf_roi(surface_image_roi.mesh, roi_map=surface_image_roi, engine=engine) is not None
