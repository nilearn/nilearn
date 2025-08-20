"""Benchmarks for plotting."""

# ruff: noqa: ARG002

from nilearn.datasets import (
    load_fsaverage_data,
    load_sample_motor_activation_image,
)
from nilearn.plotting import (
    plot_anat,
    # plot_bland_altman,
    # plot_carpet,
    # plot_connectome,
    plot_epi,
    plot_glass_brain,
    plot_img,
    # plot_img_comparison,
    # plot_prob_atlas,
    plot_roi,
    plot_stat_map,
    plot_surf,
    plot_surf_roi,
    plot_surf_stat_map,
)

PLOTTING_FUNCS_3D = [
    plot_img,
    plot_anat,
    plot_stat_map,
    plot_roi,
    plot_epi,
    plot_glass_brain,
]

SURFACE_FUNCS = [
    plot_surf,
    plot_surf_stat_map,
    plot_surf_roi,
]


class BenchMarkPlotting3D:
    """Check plotting of 3D images."""

    param_names = "plot_func"
    params = PLOTTING_FUNCS_3D

    def setup(self, plot_func):
        """Set up for all benchmarks."""
        self.img = load_sample_motor_activation_image()

    def time_plotting_3d(self, plot_func):
        """Check time."""
        plot_func(self.img)

    def peakmem_plotting_3d(self, plot_func):
        """Check peak memory."""
        plot_func(self.img)


class BenchMarkPlottingSurface:
    """Check plotting surface."""

    param_names = ("plot_func", "engine")
    params = (SURFACE_FUNCS, ["matplotlib", "plotly"])

    def setup(self, plot_func, engine):
        """Set up for all benchmarks."""
        self.surf_img = load_fsaverage_data()
        self.engine = engine

    def time_plotting_surface(self, plot_func, engine):
        """Check time."""
        plot_func(
            self.surf_img.mesh,
            self.surf_img,
            engine=engine,
        )

    def peakmem_plotting_surface(self, plot_func, engine):
        """Check peak memory."""
        plot_func(
            self.surf_img.mesh,
            self.surf_img,
            engine=engine,
        )
