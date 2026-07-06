"""Benchmarks for TFCE."""
import numpy as np
from scipy.ndimage import gaussian_filter

from nilearn.mass_univariate._utils import calculate_tfce

from ..utils import _rng


class BaseBenchMarkTFCE:
    """Base for benchmarking differen TFCE implementations."""

    def setup(
        self,
        shape=(100, 115, 85),  # Shape of regressor map
        sigma=2.0,  # Spatial correlation from gaussian filter
    ):
        """Define a 4D regressor array from TFCE benchmark."""
        rng = _rng()

        # Define a 4D regressor array from a normal
        reg_arr = rng.standard_normal((*shape, 1))
        reg_arr[..., 0] = gaussian_filter(reg_arr[..., 0], sigma=sigma)
        self.reg_arr = reg_arr
        # All adjacent
        self.adjency_struct = np.ones((3, 3, 3), dtype=bool)


class BenchMarkTFCE(BaseBenchMarkTFCE):
    """Benchmarking classical TFCE."""

    param_names = ("shape", "sigma")
    params = (
        [(20, 20, 20), (60, 60, 60)],
        [1.0, 3.0],
    )

    def setup(self, shape=(100, 115, 85), sigma=2):
        """Set up required by asv."""
        return super().setup(shape, sigma)

    def _run(self):
        calculate_tfce(self.reg_arr, self.adjency_struct, two_sided_test=False)

    def time_tfce(self, shape, sigma):
        """Time calculate_tfce."""
        self._run(shape, sigma)

    def peakmem_tfce(self, shape, sigma):
        """Measure peak memory for calculate_tfce."""
        self._run(shape, sigma)
