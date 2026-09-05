.. currentmodule:: nilearn

Version 0.15.0dev
=================

HIGHLIGHTS
----------

.. warning::

 | **Support for Python 3.10 has been dropped.**
 | **We recommend upgrading to Python 3.13 or above.**
 |
 | **Minimum supported versions of the following packages have been bumped up:**
 | - joblib -- 1.5.0
 | - scikit-learn -- 1.6.0
 |

..
    Each changelog entry should begin with one of the following badges:
    - :bdg-primary:`Doc`
    - :bdg-secondary:`Maint`
    - :bdg-success:`API`
    - :bdg-info:`Plotting`
    - :bdg-warning:`Test`
    - :bdg-danger:`Deprecation`
    - :bdg-dark:`Code`


Fixes
-----


Enhancements
------------

- :bdg-success:`API` Add an ``ensure_finite`` parameter to :func:`~image.smooth_img`, and warn when non-finite values are replaced with zeros rather than doing it silently. Replacement now happens in a single place for both the volume and the surface branch, so the two behave identically (:gh:`6530` by `Cedric Conday`_).

- :bdg-success:`API` :func:`~masking.apply_mask` now honors ``ensure_finite`` for surface data. The surface branch previously cleaned non-finite values unconditionally, ignoring the argument. Passing ``smoothing_fwhm`` still forces ``ensure_finite=True``, now on surfaces as well as volumes (:gh:`6530` by `Cedric Conday`_).

- :bdg-success:`API` The warnings raised when non-finite values are detected are now ``RuntimeWarning`` rather than ``UserWarning``. This covers the ``Non-finite values detected. These values will be replaced with zeros.`` message and the one :class:`~maskers.SurfaceMasker` raises when it masks such vertices out. Code that catches them, with ``warnings.catch_warnings`` or ``pytest.warns``, has to be updated (:gh:`6530` by `Cedric Conday`_).


Changes
-------
