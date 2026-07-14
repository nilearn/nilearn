"""Benchmarks for nilearn.glm.first_level."""
# ruff: noqa: ARG002

from typing import ClassVar

import numpy as np

from nilearn.glm.first_level import FirstLevelModel

from ..glm import BaseBenchMarkFLM


class BenchMarkFirstLevelModel(BaseBenchMarkFLM):
    """Benchmarks for FLM with different image sizes."""

    # try different combinations run length, n_runs, minimze_memory
    param_names = ("length_n_runs", "minimize_memory")
    params: ClassVar[tuple[list[tuple[int, int]], list[bool]]] = (
        [(250, 2), (500, 1)],
        [True, False],
    )

    def _fit(self, n_timepoints_n_runs, minimize_memory):
        model = FirstLevelModel(mask_img=self.mask)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)

    def time_glm_fit(self, n_timepoints_n_runs, minimize_memory):
        """Time FirstLevelModel on large data."""
        self._fit(n_timepoints_n_runs, minimize_memory)

    def peakmem_glm_fit(self, n_timepoints_n_runs, minimize_memory):
        """Measure peak memory for FirstLevelModel on large data."""
        self._fit(n_timepoints_n_runs, minimize_memory)


class BenchMarkFirstLevelModelReport(BaseBenchMarkFLM):
    """Benchmarks for FLM report generation."""

    param_names = "memory"
    params: ClassVar[list[str | None]] = [None, "nilearn_cache"]

    def _fit(self, memory):
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        model.generate_report(np.array([[1, 0, 0]]))

    def time_glm_report(self, memory):
        """Time FirstLevelModel on large data."""
        self._fit(memory)

    def peakmem_glm_report(self, memory):
        """Measure peak memory for FirstLevelModel on large data."""
        self._fit(memory)
