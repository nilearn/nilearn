"""Benchmarks for GLM."""
# ruff: noqa: ARG002

import string
from typing import ClassVar

import numpy as np
import pandas as pd

from nilearn.glm.first_level import FirstLevelModel
from nilearn.interfaces.bids import save_glm_to_bids

from ..utils import _rng, generate_fake_fmri


class BaseBenchMarkFLM:
    """Base benchmark class for GLM."""

    timeout: int = 2400  # 40 mins

    def setup(
        self,
        minimize_memory,
        memory=None,
        shape=(64, 64, 64),
        length_n_runs=(200, 1),
    ):
        """Set up for all benchmarks."""
        (length, n_runs) = length_n_runs

        shape = (200, 230, 190)

        affine = np.asarray(
            [
                [-3.0, -0.0, 0.0, -98.0],
                [-0.0, 3.0, -0.0, -134.0],
                [0.0, 0.0, 3.0, -72.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        rank = 3

        rng = _rng()

        self.fmri_data = []
        self.design_matrices = []
        for _ in range(n_runs):
            img, mask = generate_fake_fmri(
                shape=shape,
                length=length,
                affine=affine,
            )

            self.mask = mask
            self.fmri_data.append(img)

            columns = rng.choice(
                list(string.ascii_lowercase), size=rank, replace=False
            )
            self.design_matrices.append(
                pd.DataFrame(
                    rng.standard_normal((length, rank)), columns=columns
                )
            )


class BenchMarkFirstLevelModel(BaseBenchMarkFLM):
    """Benchmarks for FLM with different image sizes."""

    # try different combinations run length, n_runs, minimze_memory
    param_names = ("length_n_runs", "minimize_memory")
    params: ClassVar[tuple[list[tuple[int, int]], list[bool]]] = (
        [(500, 2), (1000, 1)],
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


class BenchMarkFirstLevelModelSave(BaseBenchMarkFLM):
    """Benchmarks for FLM saving."""

    param_names = ("memory", "n_runs")
    params = ([None, "nilearn_cache"], [1, 4])

    def _fit(self, memory, n_runs):
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory, verbose=0
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        save_glm_to_bids(model, contrasts={"con1": np.array([[1, 0, 0]])})

    def time_glm_save(self, memory, n_runs):
        """Time FirstLevelModel on large data."""
        self._fit(memory, n_runs)

    def peakmem_glm_save(self, memory, n_runs):
        """Measure peak memory for FirstLevelModel on large data."""
        self._fit(memory, n_runs)
