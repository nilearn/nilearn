import numpy as np
import pytest
from joblib import Memory
from nibabel import Nifti1Image
from sklearn.base import BaseEstimator

from nilearn._utils.masker_validation import check_embedded_masker
from nilearn.maskers import MultiNiftiMasker, NiftiMasker, SurfaceMasker


class OwningClass(BaseEstimator):
    def __init__(
        self,
        mask=None,
        smoothing_fwhm=None,
        standardize=False,
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        target_affine=None,
        target_shape=None,
        mask_strategy="background",
        mask_args=None,
        memory=None,
        memory_level=0,
        n_jobs=1,
        verbose=0,
        dummy=None,
    ):
        if memory is None:
            memory = Memory(location=None)
        self.mask = mask

        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.detrend = detrend
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.t_r = t_r
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.mask_strategy = mask_strategy
        self.mask_args = mask_args
        self.memory = memory
        self.memory_level = memory_level
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.dummy = dummy


class DummyEstimator:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def fit(self, *args, **kwargs):  # noqa: ARG002
        self.masker = check_embedded_masker(self)


def test_check_embedded_masker_defaults():
    dummy = DummyEstimator(memory=None, memory_level=1)
    with pytest.warns(
        Warning, match="Provided estimator has no verbose attribute set."
    ):
        dummy.fit()
    assert dummy.masker.memory_level == 0
    assert dummy.masker.verbose == 0
    dummy = DummyEstimator(verbose=1)
    with pytest.warns(
        Warning, match="Provided estimator has no memory attribute set."
    ):
        dummy.fit()
    assert isinstance(dummy.masker.memory, Memory)
    assert dummy.masker.memory.location is None
    assert dummy.masker.memory_level == 0
    assert dummy.masker.verbose == 1


def test_check_embedded_masker():
    owner = OwningClass()
    masker = check_embedded_masker(owner)
    assert type(masker) is MultiNiftiMasker

    for mask, masker_type in (
        (MultiNiftiMasker(), "multi_nii"),
        (NiftiMasker(), "nii"),
        (SurfaceMasker(), "surface"),
    ):
        owner = OwningClass(mask=mask)
        masker = check_embedded_masker(owner, masker_type=masker_type)
        assert isinstance(masker, type(mask))
        for param_key in masker.get_params():
            if param_key not in [
                "memory",
                "memory_level",
                "n_jobs",
                "verbose",
            ]:
                assert getattr(masker, param_key) == getattr(mask, param_key)
            else:
                assert getattr(masker, param_key) == getattr(owner, param_key)

    # Check use of mask as mask_img
    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    mask = Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
    owner = OwningClass(mask=mask)
    masker = check_embedded_masker(owner)
    assert masker.mask_img is mask

    # Check attribute forwarding
    data = np.zeros((9, 9, 9))
    data[2:-2, 2:-2, 2:-2] = 10
    imgs = Nifti1Image(data, np.eye(4))
    mask = MultiNiftiMasker()
    mask.fit([[imgs]])
    owner = OwningClass(mask=mask)
    masker = check_embedded_masker(owner)
    assert masker.mask_img is mask.mask_img_

    # Check conflict warning
    mask = NiftiMasker(mask_strategy="epi")
    owner = OwningClass(mask=mask)
    with pytest.warns(UserWarning):
        check_embedded_masker(owner)
