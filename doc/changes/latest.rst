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
 | - joblib -- 1.5

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

- :bdg-warning:`Test` Bump minimum supported ``pytest`` version to 8.0.0; earlier versions do not correctly report warnings when several ``pytest.warns`` context managers are combined in a single ``with`` statement, which made ``test_resampling_target`` fail spuriously under the ``min`` dependency set (:gh:`6566` by `Rémi Gau`_).


Enhancements
------------


Changes
-------
