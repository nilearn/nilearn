"""Benchmarks for GLM."""
# ruff: noqa: ARG002

import string

import numpy as np
import pandas as pd

from ..utils import _rng, generate_fake_fmri


class BaseBenchMarkFLM:
    """Base benchmark class for GLM."""

    timeout: int = 2400  # 40 mins

    def setup(
        self,
        minimize_memory,
        memory=None,
        shape=(64, 64, 64),
        length_n_runs=(100, 1),
    ):
        """Set up for all benchmarks."""
        (length, n_runs) = length_n_runs

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
