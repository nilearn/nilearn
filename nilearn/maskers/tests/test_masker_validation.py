import numpy as np
import pytest
from joblib import Memory
from nibabel import Nifti1Image
from sklearn.base import BaseEstimator

from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn.maskers import MultiNiftiMasker, NiftiMasker, SurfaceMasker
from nilearn.maskers.masker_validation import check_embedded_masker


class OwningClass(BaseEstimator):
    """Dummy class that can have an embedded masker."""

    _estimator_type = "home_made"  # TODO (sklearn  >= 1.8.0) remove

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

    def __sklearn_tags__(self):
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(surf_img=True, niimg_like=False)

        from nilearn._utils.tags import InputTags

        tags = BaseEstimator().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=False)
        return tags


class DummyEstimator:
    """Dummy class that checks embedded masker at fit time."""

    _estimator_type = "home_made"  # TODO (sklearn  >= 1.8.0) remove

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __sklearn_tags__(self):
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(surf_img=True, niimg_like=False)

        from nilearn._utils.tags import InputTags

        tags = BaseEstimator().__sklearn_tags__()
        tags.input_tags = InputTags(surf_img=True, niimg_like=False)
        return tags

    def fit(self, *args, **kwargs):  # noqa: ARG002
        """Fit estimator."""
        self.masker = check_embedded_masker(self, masker_type="nii")


@pytest.mark.parametrize(
    "kwargs, warning_msg, expected_verbose, expected_memory_level",
    [
        ({"memory": None, "memory_level": 1}, "verbose", 0, 0),
        ({"verbose": 1}, "memory", 1, 0),
    ],
)
def test_check_embedded_masker_defaults(
    kwargs, warning_msg, expected_verbose, expected_memory_level
):
    """Check what values are set in embedded_masker."""
    dummy = DummyEstimator(**kwargs)
    with pytest.warns(
        Warning,
        match=f"Provided estimator has no '{warning_msg}' attribute set.",
    ):
        dummy.fit()

    assert isinstance(dummy.masker.memory, Memory)
    assert dummy.masker.memory.location is None
    assert dummy.masker.memory_level == expected_memory_level
    assert dummy.masker.verbose == expected_verbose


def test_check_embedded_masker_default():
    """Check default return value."""
    owner = OwningClass()
    masker = check_embedded_masker(owner, masker_type="multi_nii")
    assert type(masker) is MultiNiftiMasker


@pytest.mark.parametrize(
    "mask, masker_type",
    (
        (MultiNiftiMasker(), "multi_nii"),
        (NiftiMasker(), "nii"),
        (SurfaceMasker(), "surface"),
    ),
)
def test_check_embedded_masker(mask, masker_type):
    """Check return value and attribute of the masker \
        when passing a masker at init.
    """
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


def test_check_embedded_masker_with_mask():
    """Check that mask of owning class is passed to embedded masker."""
    shape = (6, 8, 10, 5)
    affine = np.eye(4)
    mask = Nifti1Image(np.ones(shape[:3], dtype=np.int8), affine)
    owner = OwningClass(mask=mask)

    masker = check_embedded_masker(owner, masker_type="nii")

    assert masker.mask_img is mask


def test_check_embedded_masker_attribute_forwarding():
    """Check attribute forwarding."""
    data = np.zeros((9, 9, 9))
    data[2:-2, 2:-2, 2:-2] = 10
    imgs = Nifti1Image(data, np.eye(4))
    mask = MultiNiftiMasker()
    mask.fit([[imgs]])
    owner = OwningClass(mask=mask)

    masker = check_embedded_masker(owner, masker_type="nii")

    assert masker.mask_img is mask.mask_img_


def test_check_embedded_masker_conflict_warning():
    """Check warning thrown when changing some parameters."""
    mask = NiftiMasker(mask_strategy="epi")
    owner = OwningClass(mask=mask)
    with pytest.warns(
        UserWarning,
        match="overriding estimator parameter background",
    ):
        check_embedded_masker(owner, masker_type="nii")
