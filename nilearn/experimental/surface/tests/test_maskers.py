import warnings

import pytest

from nilearn._utils.helpers import is_matplotlib_installed
from nilearn.experimental.surface import (
    SurfaceLabelsMasker,
)


@pytest.mark.skipif(
    is_matplotlib_installed(),
    reason="Test requires matplotlib not to be installed.",
)
def test_masker_reporting_mpl_warning(mini_label_img):
    """Raise warning after exception if matplotlib is not installed."""
    with warnings.catch_warnings(record=True) as warning_list:
        SurfaceLabelsMasker(mini_label_img).fit().generate_report()

    assert len(warning_list) == 1
    assert issubclass(warning_list[0].category, ImportWarning)
