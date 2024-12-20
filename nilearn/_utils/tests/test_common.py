import numpy as np
import pytest

from nilearn._utils import all_classes, all_functions, all_modules
from nilearn._utils.helpers import is_matplotlib_installed


@pytest.mark.parametrize("func", [all_modules, all_functions, all_classes])
def test_all_modules_error(func):
    with pytest.raises(
        ValueError,
        match=(
            "`modules_to_ignore` and "
            "`modules_to_consider` cannot "
            "be both specified."
        ),
    ):
        func(modules_to_ignore=["foo"], modules_to_consider=["bar"])


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="This test requires matplotlib to be installed.",
)
@pytest.mark.parametrize("func", [all_functions, all_classes])
def test_private_vs_public(func):
    public_only = set(func(return_private=False))
    private_and_public = set(func(return_private=True))
    assert public_only.issubset(private_and_public)
    assert np.all(
        [elt[0].startswith("_") for elt in private_and_public - public_only]
    )


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="This test requires matplotlib to be installed.",
)
def test_number_public_functions():
    """Check that number of public functions is stable.

    If it changes, it means that we have added or removed a public function.
    If this is intentional, then the number should be updated in the test.
    Otherwise it means that the public API of nilearn has changed by mistake.
    """
    assert len({_[0] for _ in all_functions()}) == 246


@pytest.mark.skipif(
    not is_matplotlib_installed(),
    reason="This test requires matplotlib to be installed.",
)
def test_number_public_classes():
    """Check that number of public classes is stable.

    If it changes, it means that we have added or removed a public function.
    If this is intentional, then the number should be updated in the test.
    Otherwise it means that the public API of nilearn has changed by mistake.
    """
    assert len({_[0] for _ in all_classes()}) == 67
