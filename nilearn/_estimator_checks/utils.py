from functools import wraps

import pytest
from sklearn.base import clone as sklearn_clone
from sklearn.utils._testing import set_random_state

from nilearn._utils.helpers import is_gil_enabled
from nilearn.utils.tags import accepts_surface, accepts_volume


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
                        f"'{estimator.__class__.__name__}'. "
                        f"{reason}"
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
