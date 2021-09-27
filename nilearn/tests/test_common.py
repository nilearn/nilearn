import pytest
import numpy as np
from nilearn._utils import (all_modules,
                            all_functions,
                            all_classes)


@pytest.mark.parametrize("func",
                         [all_modules,
                          all_functions,
                          all_classes])
def test_all_modules_error(func):
    with pytest.raises(ValueError,
                       match=("`modules_to_ignore` and "
                              "`modules_to_consider` cannot "
                              "be both specified.")):
        func(modules_to_ignore=["foo"], modules_to_consider=["bar"])


@pytest.mark.parametrize("func",
                         [all_functions,
                          all_classes])
def test_private_vs_public(func):
    set1 = set(func(return_private=False))
    set2 = set(func(return_private=True))
    assert set1.issubset(set2)
    assert np.all([elt[0].startswith("_") for elt in set2 - set1])


def test_all_functions():
    assert(set(
        [_[0] for _ in all_functions(modules_to_consider=["input_data"])]
    ) == set(['filter_and_extract',
              'check_embedded_nifti_masker',
              'filter_and_mask']))


def test_all_classes():
    from nilearn import input_data
    assert(
        (set([_[0] for _ in all_classes(modules_to_consider=["input_data"])])
         - set(["BaseMasker"])) == set(input_data.__all__)
    )
