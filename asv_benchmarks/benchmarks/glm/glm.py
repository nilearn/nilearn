"""Benchmarks for GLM."""

import string

import numpy as np
import pandas as pd

from .._utils import _rng, generate_fake_fmri


class BaseBenchMarkFLM:
    """Base benchmark class for GLM.

    Subclasses must define their own ``setup`` method
    whose signature matches their ``params``
    (asv passes parameters positionally),
    and call ``_generate_data`` from it.
    """

    timeout: int = 2400  # 40 mins

    def _generate_data(self, length: int = 100, n_runs: int = 1) -> None:
        """Generate fake fMRI runs, a mask, and design matrices.

        Parameters
        ----------
        length : :obj:`int`, default=100
            Number of time points per run.

        n_runs : :obj:`int`, default=1
            Number of runs.
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
