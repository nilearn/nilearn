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


Enhancements
------------


Changes
-------

- :bdg-warning:`Test` Remove keyword arguments in tests that were explicitly set to the default value of the function, method, or class being called, so that the implicit default behavior is exercised instead. A new script ``maint_tools/check_test_default_kwargs.py`` was added to detect (and optionally auto-fix with ``--fix``) such "extra-default" keyword arguments (:gh:`PR_NUMBER` by `Rémi Gau`_).
