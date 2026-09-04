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

- :bdg-danger:`Deprecation` The functions ``nilearn.reporting.make_glm_report`` and ``nilearn.interfaces.bids.glm.save_glm_to_bids`` have been removed: instead now use :meth:`~nilearn.glm.first_level.FirstLevelModel.generate_report` or :meth:`~nilearn.glm.second_level.SecondLevelModel.generate_report`, and :func:`nilearn.glm.save_glm_to_bids` respectively (:gh:`6548` by `Rémi Gau`_).
