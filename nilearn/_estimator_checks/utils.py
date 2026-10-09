import warnings
from functools import wraps

import numpy as np
import pytest
from nibabel import Nifti1Image
from sklearn.base import clone as sklearn_clone
from sklearn.base import is_classifier, is_regressor
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.utils._testing import set_random_state

from nilearn._base import NilearnBaseEstimator
from nilearn._utils.helpers import is_gil_enabled
from nilearn.conftest import (
    _affine_eye,
    _make_surface_img,
    _make_surface_img_and_design,
    _rng,
    _shape_3d_large,
)
from nilearn.decoding.decoder import (
    FREMClassifier,
    FREMRegressor,
)
from nilearn.decoding.searchlight import SearchLight
from nilearn.decoding.tests.test_same_api import to_niimgs
from nilearn.decomposition._base import _BaseDecomposition
from nilearn.decomposition.dict_learning import DictLearning
from nilearn.decomposition.tests.conftest import (
    _canica_components_volume,
    _make_volume_data_from_components,
)
from nilearn.surface import SurfaceImage
from nilearn.utils.tags import (
    accepts_surface,
    accepts_volume,
    is_glm,
    is_masker,
)


def accepts_image(estimator):
    """Check if estimator accepts volume of surface image."""
    return accepts_volume(estimator) or accepts_surface(estimator)


def clone(estimator):
    """Clone estimator, set random_state to and return."""
    estimator = sklearn_clone(estimator)
    # sets random_state to 0 if parameter exists for the estimator
    set_random_state(estimator)
    return estimator


def requires_y(estimator):
    """Check if estimator expects target as input."""
    tags = estimator.__sklearn_tags__()
    return getattr(tags.target_tags, "required", True)


def generate_data_to_fit(estimator: NilearnBaseEstimator):
    """Generate fit data for the specified estimator."""
    if is_glm(estimator):
        data, design_matrices = _make_surface_img_and_design()
        return data, design_matrices

    elif isinstance(estimator, SearchLight):
        n_samples = 30
        data = _rng().random((5, 5, 5, n_samples))
        # Create a condition array, with balanced classes
        y = np.arange(n_samples, dtype=int) >= (n_samples // 2)

        data[2, 2, 2, :] = 0
        data[2, 2, 2, y] = 2
        X = Nifti1Image(data, np.eye(4))

        return X, y

    elif is_classifier(estimator):
        dim = 5
        if isinstance(estimator, FREMClassifier):
            # FREM needs may need more features in some cases
            dim = 10
        X, y = make_classification(
            n_samples=30,
            n_features=dim**3,
            scale=3.0,
            n_informative=5,
            n_classes=2,
            random_state=42,
            shift=100,
        )
        X, _ = to_niimgs(X, [dim, dim, dim])
        return X, y

    elif is_regressor(estimator):
        dim = 5
        if isinstance(estimator, FREMRegressor):
            # FREM needs may need more features in some cases
            dim = 10
        X, y = make_regression(
            n_samples=30,
            n_features=dim**3,
            n_informative=dim,
            noise=1.5,
            bias=1.0,
            random_state=42,
        )
        X = StandardScaler().fit_transform(X)
        X, _ = to_niimgs(X, [dim, dim, dim])
        return X, y

    elif is_masker(estimator):
        imgs: Nifti1Image | SurfaceImage
        if accepts_volume(estimator):
            imgs = Nifti1Image(
                _rng().random(_shape_3d_large()) + 10.0,
                _affine_eye(),
            )
        else:
            imgs = _make_surface_img(10)
        return imgs, None

    elif isinstance(estimator, _BaseDecomposition):
        n_subjects = 2
        n_timepoints = 40
        if isinstance(estimator, DictLearning):
            n_subjects = 1
            n_timepoints = 200

        decomp_input = _make_volume_data_from_components(
            _canica_components_volume(_shape_3d_large()),
            _affine_eye(),
            _shape_3d_large(),
            _rng(),
            n_subjects=n_subjects,
            n_timepoints=n_timepoints,
        )

        return decomp_input[0], None

    elif not (accepts_volume(estimator) or accepts_surface(estimator)):
        return _rng().random((5, 5)), None

    else:
        data = _rng().random(_shape_3d_large()) + 10.0
        imgs = Nifti1Image(data, _affine_eye())
        return imgs, None


def fit_estimator(
    estimator: NilearnBaseEstimator, X=None, y=None
) -> NilearnBaseEstimator:
    """Fit on a nilearn estimator with appropriate input and return it."""
    if X is None and y is None:
        X, y = generate_data_to_fit(estimator)

    if is_glm(estimator):
        # FirstLevel
        if hasattr(estimator, "hrf_model"):
            return estimator.fit(X, design_matrices=y)
        # SecondLevel
        else:
            return estimator.fit(X, design_matrix=y)

    elif (
        isinstance(estimator, SearchLight)
        or is_classifier(estimator)
        or is_regressor(estimator)
    ):
        return estimator.fit(X, y)

    else:
        if not isinstance(estimator, _BaseDecomposition):
            return estimator.fit(X)

        with warnings.catch_warnings():
            # might not converge
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            return estimator.fit(X)


# ------------------------- Decorators ----------------------------


def clone_estimator(check_func):
    """Provide cloned estimator to check function.

    This decorator should be set at the bottom of all decorators.
    """

    @wraps(check_func)
    def wrapper(estimator):
        estimator = clone(estimator)
        return check_func(estimator)

    return wrapper


def skip_if(*conditions):
    """Skip a check if estimator satisfies one of the conditions."""

    def decorator(check_func):

        @wraps(check_func)
        def wrapper(estimator):
            for condition in conditions:
                if isinstance(condition, tuple):
                    condition, reason = condition
                else:
                    reason = ""
                if condition(estimator):
                    print(
                        f"\n'{check_func.__name__}' does not apply to class "
                        f"'{estimator.__class__.__name__}' for {condition}. "
                        f"{reason}"
                    )
                    return estimator
            return check_func(estimator)

        return wrapper

    return decorator


def skip_if_not(*conditions):
    """Skip a check if estimator does not satisfy one of the conditions."""

    def decorator(check_func):

        @wraps(check_func)
        def wrapper(estimator):
            for condition in conditions:
                if isinstance(condition, tuple):
                    condition, reason = condition
                else:
                    reason = ""
                if not condition(estimator):
                    print(
                        f"\n'{check_func.__name__}' does not apply to class "
                        f"'{estimator.__class__.__name__}' for not "
                        f"{condition}. {reason}"
                    )
                    return estimator
            return check_func(estimator)

        return wrapper

    return decorator


def xfail_if_not_gil(classes=None):
    """Skip a check if GIL is not enabled.

    If ``classes`` is specified, it skip the check if also the estimator
    is an instance of one of the classes listed.

    This decorator should be set at the top of all decorators.
    """

    def decorator(check_func):

        @wraps(check_func)
        def wrapper(estimator):
            if not is_gil_enabled() and (
                classes is None
                or (classes and isinstance(estimator, tuple(classes)))
            ):
                pytest.xfail("May fail without the GIL")
            else:
                return check_func(estimator)

        return wrapper

    return decorator
