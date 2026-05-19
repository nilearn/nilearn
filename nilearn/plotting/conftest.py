"""Fixtures for tests for plotting."""

import numpy as np
import pytest
from nibabel import Nifti1Image

from nilearn.datasets import load_mni152_template


@pytest.fixture(scope="session")
def mni152_template_res_2() -> Nifti1Image:
    """Return the mni152 template at 2 mm resolution."""
    return load_mni152_template(resolution=2)


@pytest.fixture
def adjacency():
    """Adjacency matrix symmetric up to 1e-3 relative tolerance."""
    return np.array(
        [
            [1.0, -2.0, 0.3, 0.0],
            [-2.002, 1, 0.0, 0.0],
            [0.3, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@pytest.fixture
def node_coords():
    """Array of node coordinates for testing."""
    return np.arange(3 * 4).reshape(4, 3)


@pytest.fixture
def params_plot_connectome():
    """Return basic set of parameters for testing plot_connectome."""
    return {"edge_threshold": 0.38, "title": "threshold=0.38", "node_size": 10}
