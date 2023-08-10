import numpy as np
import pytest

from nilearn._utils import all_classes, all_functions, all_modules


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


@pytest.mark.parametrize("func", [all_functions, all_classes])
def test_private_vs_public(func):
    public_only = set(func(return_private=False))
    private_and_public = set(func(return_private=True))
    assert public_only.issubset(private_and_public)
    assert np.all([elt[0].startswith("_") for elt in private_and_public - public_only])


def test_all_functions():
    assert {
        _[0] for _ in all_functions(modules_to_consider=["connectome"])
    } == {
        'prec_to_partial',
        'vec_to_sym_matrix',
        'group_sparse_covariance_path',
        'empirical_covariances',
        'group_sparse_scores',
        'group_sparse_covariance',
        'cov_to_corr',
        'compute_alpha_max',
        'sym_matrix_to_vec'
    }


def test_all_classes():
    from nilearn import maskers

    assert (
        {_[0] for _ in all_classes(modules_to_consider=["maskers"])}
    ) == set(maskers.__all__)
