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


Changes
-------

- :bdg-danger:`Deprecation` The functions ``nilearn.reporting.make_glm_report`` and ``nilearn.interfaces.bids.glm.save_glm_to_bids`` have been removed: instead now use :meth:`~nilearn.glm.first_level.FirstLevelModel.generate_report` or :meth:`~nilearn.glm.second_level.SecondLevelModel.generate_report`, and :func:`nilearn.glm.save_glm_to_bids` respectively (:gh:`6548` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``keep_masked_maps`` (and respectively the parameter ``keep_masked_labels``) has been removed from :func:`~nilearn.regions.img_to_signals_maps`, :class:`~nilearn.maskers.NiftiMapsMasker` and :class:`~nilearn.maskers.MultiNiftiMapsMasker` (and respectively from :func:`~nilearn.regions.img_to_signals_labels`, :class:`~nilearn.maskers.NiftiLabelsMasker` and :class:`~nilearn.maskers.MultiNiftiLabelsMasker`). In practice, this means that data will not be extracted from maps or labels that are excluded by a mask image (:gh:`6551` by `Rémi Gau`_).
