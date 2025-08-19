"""Benchmarks for GLM."""

import numpy as np

from nilearn._utils.data_gen import generate_fake_fmri_data_and_design
from nilearn.datasets import load_mni152_brain_mask
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img


class BenchMarkFirstLevelModel:
    """Benchmarks for FLM with different image sizes."""

    # try different combinations of parameters for the NiftiMasker object
    param_names = "n_timepoints"
    params = ([500],)

    def setup(self, n_timepoints):
        """Set up common to all benchmarks."""
        mni_brain_mask = load_mni152_brain_mask(resolution=3)
        shape = [(64, 64, 64, n_timepoints)]
        mask, self.fmri_data, self.design_matrices = (
            generate_fake_fmri_data_and_design(
                shapes=shape, affine=mni_brain_mask.affine
            )
        )

        self.mask = resample_to_img(
            source_img=mni_brain_mask,
            target_img=mask,
            interpolation="nearest",
            copy_header=True,
            force_resample=True,
        )

    def time_glm(self):
        """Time FirstLevelModel on large data."""
        model = FirstLevelModel(mask_img=self.mask)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)

    def mem_glm(self):
        """Measure memory for FirstLevelModel on large data."""
        model = FirstLevelModel(mask_img=self.mask)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)

    def peakmem_glm(self):
        """Measure peak memory for FirstLevelModel on large data."""
        model = FirstLevelModel(mask_img=self.mask)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)


class BenchMarkFirstLevelModelReport:
    """Benchmarks for FLM report generation."""

    def setup(self):
        """Set up common to all benchmarks."""
        mni_brain_mask = load_mni152_brain_mask(resolution=3)
        shape = [(64, 64, 64, 100), (64, 64, 64, 100)]
        mask, self.fmri_data, self.design_matrices = (
            generate_fake_fmri_data_and_design(
                shapes=shape, affine=mni_brain_mask.affine
            )
        )

        self.mask = resample_to_img(
            source_img=mni_brain_mask,
            target_img=mask,
            interpolation="nearest",
            copy_header=True,
            force_resample=True,
        )

    def time_glm(self):
        """Time FirstLevelModel on large data."""
        model = FirstLevelModel(mask_img=self.mask, minimize_memory=False)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        model.generate_report(np.ndarray([1]))

    def mem_glm(self):
        """Measure memory for FirstLevelModel on large data."""
        model = FirstLevelModel(mask_img=self.mask, minimize_memory=False)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        model.generate_report(np.ndarray([1]))

    def peakmem_glm(self):
        """Measure peak memory for FirstLevelModel on large data."""
        model = FirstLevelModel(mask_img=self.mask, minimize_memory=False)
        model.fit(self.fmri_data, design_matrices=self.design_matrices)
        model.generate_report(np.ndarray([1]))
