import pytest

from nilearn._utils.helpers import is_matplotlib_installed, is_plotly_installed
from nilearn.plotting.surface._backend import get_surface_backend


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="This test is run only if matplotlib is not installed.",
)
def test_get_surface_backend_matplotlib_not_installed():
    """Tests to see if get_surface_backend raises error when matplotlib is not
    installed.
    """
    with pytest.raises(ImportError, match="Using engine"):
        get_surface_backend("matplotlib")


@pytest.mark.skipif(
    is_plotly_installed(),
    reason="This test is run only if plotly is not installed.",
)
def test_get_surface_backend_plotly_not_installed():
    """Tests to see if get_surface_backend raises error when plotly is not
    installed.
    """
    with pytest.raises(ImportError, match="Using engine"):
        get_surface_backend("plotly")


def test_get_surface_backend_unknown_error():
    """Tests to see if get_surface_backend raises error when plotly is not
    installed.
    """
    with pytest.raises(ValueError, match="Unknown plotting"):
        get_surface_backend("unknown")


class BaseTestSurfaceBackend:
    """A base class to test SurfaceBackend methods."""

    def test_get_surface_backend(self, engine):
        """Tests if backend does not return None or raise error if engine is
        installed for implemented engines.
        """
        assert get_surface_backend(engine) is not None

    def test_plot_surf(self, engine, in_memory_mesh):
        """Smoke test for SurfaceBackend.plot_surf for implemented engines."""
        assert (
            get_surface_backend(engine).plot_surf(in_memory_mesh) is not None
        )

    def test_plot_surf_stat_map(self, engine, in_memory_mesh, bg_map):
        """Smoke test for SurfaceBackend.plot_surf_stat_map for implemented
        engines.
        """
        assert (
            get_surface_backend(engine).plot_surf_stat_map(
                in_memory_mesh, stat_map=bg_map
            )
            is not None
        )

    def test_plot_surf_roi(self, engine, surface_image_roi):
        """Smoke test for SurfaceBackend.plot_surf_roi for implemented
        engines.
        """
        assert (
            get_surface_backend(engine).plot_surf_roi(
                surface_image_roi.mesh,
                roi_map=surface_image_roi,
            )
            is not None
        )

    def test_plot_img_on_surf(self, engine, img_3d_mni):
        """Smoke test for SurfaceBackend.plot_img_on_surf."""
        assert (
            get_surface_backend(engine).plot_img_on_surf(img_3d_mni)
            is not None
        )

    def test_plot_surf_contours(self, engine, surf_mesh, surf_mask_1d):
        """Smoke test for SurfaceBackend.plot_surf_contours."""
        assert (
            get_surface_backend(engine).plot_surf_contours(
                surf_mesh, roi_map=surf_mask_1d
            )
            is not None
        )
