from nistats.design_matrix import make_design_matrix
from nistats.reporting import plot_design_matrix
import numpy as np
from numpy.testing import dec

# Set the backend to avoid having DISPLAY problems
from nilearn.plotting import _set_mpl_backend
# Avoid making pyflakes unhappy
_set_mpl_backend
try:
    import matplotlib.pyplot
    # Avoid making pyflakes unhappy
    matplotlib.pyplot
except ImportError:
    have_mpl = False
else:
    have_mpl = True


@dec.skipif(not have_mpl)
def test_show_design_matrix():
    # test that the show code indeed (formally) runs
    frame_times = np.linspace(0, 127 * 1., 128)
    DM = make_design_matrix(
        frame_times, drift_model='polynomial', drift_order=3)
    ax = plot_design_matrix(DM)
    assert (ax is not None)
