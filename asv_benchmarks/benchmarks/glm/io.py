"""Benchmarks for saving GLM outputs to BIDS."""
# ruff: noqa: ARG002

import numpy as np

from nilearn.glm.first_level import FirstLevelModel

from .glm import BaseBenchMarkFLM


class BenchMarkFirstLevelModelSave(BaseBenchMarkFLM):
    """Benchmarks for FLM saving."""

    param_names = ("memory", "n_runs")
    params = ([None, "nilearn_cache"], [1, 4])

    def setup(self, memory, n_runs):
        """Set up for all benchmarks."""
        super().setup(memory, n_runs)

        # save_glm_to_bids is permanently moving to nilearn.glm.io in
        # nilearn 0.15.0: nilearn.interfaces.bids.save_glm_to_bids (its
        # current, deprecated, location) will be removed then. Import
        # locally so that benchmarking versions on either side of that
        # move only fails this benchmark instead of the whole module.
        try:
            from nilearn.interfaces.bids import save_glm_to_bids
        except ImportError:
            from nilearn.glm.io import save_glm_to_bids

        self.save_glm_to_bids = save_glm_to_bids

    def _fit(self, memory, n_runs):
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory, verbose=0
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        self.save_glm_to_bids(model, contrasts={"con1": np.array([[1, 0, 0]])})

    def time_glm_save(self, memory, n_runs):
        """Time FirstLevelModel on large data."""
        self._fit(memory, n_runs)

    def peakmem_glm_save(self, memory, n_runs):
        """Measure peak memory for FirstLevelModel on large data."""
        self._fit(memory, n_runs)
