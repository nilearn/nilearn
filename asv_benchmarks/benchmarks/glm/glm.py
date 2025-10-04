"""Benchmarks for GLM."""
# ruff: noqa: ARG002

from typing import ClassVar

import numpy as np

from nilearn._utils.data_gen import generate_fake_fmri_data_and_design
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img
from nilearn.interfaces.bids.glm import save_glm_to_bids


class BenchMarkFirstLevelModel:
    """Benchmarks for FLM with different image sizes."""

    # try different combinations run length, n_runs, minimze_memory
    param_names = ("n_timepoints_n_runs", "minimize_memory")
    params: ClassVar[tuple[list[tuple[int, int]], list[bool]]] = (
        [(500, 2), (1000, 1)],
        [True, False],
    )

    def setup(self, n_timepoints_n_runs, minimize_memory):
        """Set up common to all benchmarks."""
        (n_timepoints, n_runs) = n_timepoints_n_runs
        mni_brain_mask = load_mni152_brain_mask(resolution=3)
        shape = [(64, 64, 64, n_timepoints)] * n_runs
        mask, self.fmri_data, self.design_matrices = (
            generate_fake_fmri_data_and_design(
                shapes=shape, affine=mni_brain_mask.affine
            )
        )

        self.mask = resample_to_img(
            source_img=mni_brain_mask,
            target_img=mask,
            interpolation="nearest",
            force_resample=True,
        )

    def time_glm_fit(self, n_timepoints_n_runs, minimize_memory):
        """Time FirstLevelModel on large data."""
        model = FirstLevelModel(mask_img=self.mask)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)

    def peakmem_glm_fit(self, n_timepoints_n_runs, minimize_memory):
        """Measure peak memory for FirstLevelModel on large data."""
        model = FirstLevelModel(mask_img=self.mask)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)


class BenchMarkFirstLevelModelReport:
    """Benchmarks for FLM report generation."""

    param_names = "memory"
    params: ClassVar[list[str | None]] = [None, "nilearn_cache"]

    def setup(self, memory):
        """Set up common to all benchmarks."""
        mni_brain_mask = load_mni152_brain_mask(resolution=3)
        shape = [(64, 64, 64, 200), (64, 64, 64, 200), (64, 64, 64, 200)]
        mask, self.fmri_data, self.design_matrices = (
            generate_fake_fmri_data_and_design(
                shapes=shape, affine=mni_brain_mask.affine
            )
        )

        self.mask = resample_to_img(
            source_img=mni_brain_mask,
            target_img=mask,
            interpolation="nearest",
            force_resample=True,
        )

    def time_glm_report(self, memory):
        """Time FirstLevelModel on large data."""
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        model.generate_report(np.array([[1, 0, 0]]))

    def peakmem_glm_report(self, memory):
        """Measure peak memory for FirstLevelModel on large data."""
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        model.generate_report(np.array([[1, 0, 0]]))


class BenchMarkFirstLevelModelSave:
    """Benchmarks for FLM saving."""

    timeout = 240

    param_names = ("memory", "n_runs")
    params = ([None, "nilearn_cache"], [1, 4])

    def setup(self, memory, n_runs):
        """Set up common to all benchmarks."""
        mni_brain_mask = load_mni152_brain_mask(resolution=3)
        shape = [(30, 30, 30, 100)] * n_runs
        mask, self.fmri_data, self.design_matrices = (
            generate_fake_fmri_data_and_design(
                shapes=shape, affine=mni_brain_mask.affine
            )
        )

        self.mask = resample_to_img(
            source_img=mni_brain_mask,
            target_img=mask,
            interpolation="nearest",
            force_resample=True,
        )

    def time_glm_save(self, memory, n_runs):
        """Time FirstLevelModel on large data."""
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory, verbose=0
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        save_glm_to_bids(model, contrasts={"con1": np.array([[1, 0, 0]])})

    def peakmem_glm_save(self, memory, n_runs):
        """Measure peak memory for FirstLevelModel on large data."""
        model = FirstLevelModel(
            mask_img=self.mask, minimize_memory=False, memory=memory, verbose=0
        )
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        save_glm_to_bids(model, contrasts={"con1": np.array([[1, 0, 0]])})
