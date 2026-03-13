"""Benchmarks for plotting."""

# ruff: noqa: ARG002

import numpy as np

from nilearn.datasets import fetch_surf_fsaverage
from nilearn.plotting import (
    plot_anat,
    plot_epi,
    plot_glass_brain,
    plot_img,
    plot_stat_map,
    plot_surf,
    plot_surf_stat_map,
)

from ..utils import generate_fake_fmri

PLOTTING_FUNCS_3D = [
    plot_img,
    plot_anat,
    plot_stat_map,
    plot_epi,
    plot_glass_brain,
]

SURFACE_FUNCS = [
    plot_surf,
    plot_surf_stat_map,
]


class BenchMarkPlotting3D:
    """Check plotting of 3D images."""

    length = 1

    param_names = "plot_func"
    params = PLOTTING_FUNCS_3D

    def setup(self, plot_func):
        """Set up for all benchmarks.

        Use affine and shape adapted from load_mni152_template()
        """
        shape = (200, 230, 190)

        affine = np.asarray(
            [
                [-3.0, -0.0, 0.0, -98.0],
                [-0.0, 3.0, -0.0, -134.0],
                [0.0, 0.0, 3.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        img, _ = generate_fake_fmri(
            shape=shape,
            length=self.length,
            affine=affine,
        )

        self.img = img

    def time_plotting_3d(self, plot_func):
        """Check time."""
        plot_func(self.img)

    def peakmem_plotting_3d(self, plot_func):
        """Check peak memory."""
        plot_func(self.img)


class BenchMarkPlottingSurface:
    """Check plotting surface."""

    param_names = ("plot_func", "engine")
    params = (SURFACE_FUNCS, ["matplotlib"])

    def setup(self, plot_func, engine):
        """Set up for all benchmarks."""
        self.surf_mesh = fetch_surf_fsaverage()["infl_left"]
        self.surf_stat_map = fetch_surf_fsaverage()["curv_left"]
        self.engine = engine

    def time_plotting_surface(self, plot_func, engine):
        """Check time."""
        plot_func(
            self.surf_mesh,
            self.surf_stat_map,
            engine=engine,
        )

    def peakmem_plotting_surface(self, plot_func, engine):
        """Check peak memory."""
        plot_func(
            self.surf_mesh,
            self.surf_stat_map,
            engine=engine,
        )
