"""Benchmarks for plotting."""

# ruff: noqa: ARG002

import string

import numpy as np
import pandas as pd

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

from ..utils import _rng, generate_fake_fmri

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
        shape = (100, 115, 85)

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


class BenchMarkPlotDesignMatrixCorrelation:
    """Check plotting of design matrix correlation."""

    length = 100
    rank = 10

    def setup(self):
        """Set up for all benchmarks."""
        # Imported locally (added in 0.11.0) so that benchmarking older
        # nilearn versions only fails this benchmark instead of making
        # the whole module fail to import for every benchmark in this file.
        from nilearn.plotting import plot_design_matrix_correlation

        self.plot_design_matrix_correlation = plot_design_matrix_correlation

        rng = _rng()
        columns = rng.choice(
            list(string.ascii_lowercase), size=self.rank, replace=False
        )
        self.design_matrix = pd.DataFrame(
            rng.standard_normal((self.length, self.rank)), columns=columns
        )

    def time_plot_design_matrix_correlation(self):
        """Check time."""
        self.plot_design_matrix_correlation(self.design_matrix)

    def peakmem_plot_design_matrix_correlation(self):
        """Check peak memory."""
        self.plot_design_matrix_correlation(self.design_matrix)
