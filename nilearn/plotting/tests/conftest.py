import pytest


@pytest.fixture(autouse=True)
def close_all():
    """Close all matplotlib figures."""
    yield
    import matplotlib.pyplot as plt
    plt.close('all')
