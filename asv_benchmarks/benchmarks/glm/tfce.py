"""Benchmarks for TFCE."""

import numpy as np
from nibabel import Nifti1Image
from scipy.ndimage import gaussian_filter

from nilearn.maskers import NiftiMasker
from nilearn.mass_univariate import permuted_ols

from ..utils import _rng


class BaseBenchMarkTFCE:
    """Base for benchmarking different TFCE implementations."""

    def setup(
        self,
        n_samples=10,
        shape=(100, 115, 85),  # Shape of regressor map
        sigma=2.0,  # Spatial correlation from gaussian filter
    ):
        """Define a 4D regressor array from TFCE benchmark."""
        rng = _rng()

        data_4d = np.empty((*shape, n_samples))
        for i in range(n_samples):
            # Define a 4D regressor array from a normal
            reg_arr = rng.standard_normal(shape)
            data_4d[..., i] = gaussian_filter(reg_arr, sigma=sigma)

        # mask is obligatory - need to flatten
        mask_img = Nifti1Image(np.ones(shape, dtype="int8"), affine=np.eye(4))
        self.masker = NiftiMasker(mask_img=mask_img).fit()

        # create flatenned version for permuted_ols
        # (n_samples, n_descriptors)
        # Y
        img_4d = Nifti1Image(data_4d, affine=np.eye(4))
        self.target_vars = self.masker.transform(img_4d)

        # X - one sample t-test
        self.tested_vars = np.ones((n_samples, 1))


class BenchMarkTFCE(BaseBenchMarkTFCE):
    """Benchmarking classical TFCE."""

    param_names = ("n_samples", "shape", "sigma")
    params = (
        [10, 20],
        [(20, 20, 20), (60, 60, 60)],
        [1.0, 3.0],
    )

    def setup(self, n_samples, shape=(100, 115, 85), sigma=2):
        """Set up required by asv."""
        return super().setup(n_samples, shape, sigma)

    def _run(self):
        permuted_ols(
            self.tested_vars,
            self.target_vars,
            masker=self.masker,
            tfce=True,
            n_perm=10,
            n_jobs=1,
        )

    # Arguments are not used to we need to drop ruffer for these lines
    def time_tfce(self, n_samples, shape, sigma):  # noqa: ARG002
        """Time calculate_tfce."""
        self._run()

    def peakmem_tfce(self, n_samples, shape, sigma):  # noqa: ARG002
        """Measure peak memory for calculate_tfce."""
        self._run()
