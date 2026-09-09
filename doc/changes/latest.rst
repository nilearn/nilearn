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

- :bdg-dark:`Code` Use a `rich <https://github.com/Textualize/rich>`_ progress bar to report dataset download progress when it is installed (:gh:`6434` by `Rémi Gau`_).


Changes
-------

- :bdg-danger:`Deprecation` The ``return_label_maps`` parameter of :func:`~reporting.get_clusters_table` is deprecated and will be removed in version 0.17.0, when cluster label maps will always be returned together with the table (:gh:`6376` by `Mohammad Sadeghi Hardengi`_).

- :bdg-danger:`Deprecation` The functions ``nilearn.reporting.make_glm_report`` and ``nilearn.interfaces.bids.glm.save_glm_to_bids`` have been removed: instead now use :meth:`~nilearn.glm.first_level.FirstLevelModel.generate_report` or :meth:`~nilearn.glm.second_level.SecondLevelModel.generate_report`, and :func:`nilearn.glm.save_glm_to_bids` respectively (:gh:`6548` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter name ``filename`` of the :meth:`~nilearn.plotting.displays.BaseSlicer.savefig` will be permanently replaced by ``output_file`` in version 0.17.0 (:gh:`6471` by `Aniket Singh Yadav`_).
