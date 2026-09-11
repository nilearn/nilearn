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

- :bdg-success:`API` Add an ``ensure_finite`` parameter to :func:`~image.smooth_img`, and warn when non-finite values are replaced with zeros rather than doing it silently. Replacement now happens in a single place for both the volume and the surface branch, so the two behave identically (:gh:`6530` by `Cedric Conday`_).

- :bdg-success:`API` :func:`~masking.apply_mask` now honors ``ensure_finite`` for surface data. The surface branch previously cleaned non-finite values unconditionally, ignoring the argument. Passing ``smoothing_fwhm`` still forces ``ensure_finite=True``, now on surfaces as well as volumes (:gh:`6530` by `Cedric Conday`_).

- :bdg-success:`API` The warnings raised when non-finite values are detected are now ``RuntimeWarning`` rather than ``UserWarning``. This covers the ``Non-finite values detected. These values will be replaced with zeros.`` message and the one :class:`~maskers.SurfaceMasker` raises when it masks such vertices out. Code that catches them, with ``warnings.catch_warnings`` or ``pytest.warns``, has to be updated (:gh:`6530` by `Cedric Conday`_).


Changes
-------

- :bdg-danger:`Deprecation` The ``return_label_maps`` parameter of :func:`~reporting.get_clusters_table` is deprecated and will be removed in version 0.17.0, when cluster label maps will always be returned together with the table (:gh:`6376` by `Mohammad Sadeghi Hardengi`_).

- :bdg-danger:`Deprecation` The functions ``nilearn.reporting.make_glm_report`` and ``nilearn.interfaces.bids.glm.save_glm_to_bids`` have been removed: instead now use :meth:`~nilearn.glm.first_level.FirstLevelModel.generate_report` or :meth:`~nilearn.glm.second_level.SecondLevelModel.generate_report`, and :func:`nilearn.glm.save_glm_to_bids` respectively (:gh:`6548` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter ``keep_masked_maps`` (and respectively the parameter ``keep_masked_labels``) has been removed from :func:`~nilearn.regions.img_to_signals_maps`, :class:`~nilearn.maskers.NiftiMapsMasker` and :class:`~nilearn.maskers.MultiNiftiMapsMasker` (and respectively from :func:`~nilearn.regions.img_to_signals_labels`, :class:`~nilearn.maskers.NiftiLabelsMasker` and :class:`~nilearn.maskers.MultiNiftiLabelsMasker`). In practice, this means that data will not be extracted from maps or labels that are excluded by a mask image (:gh:`6551` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Boolean values for the ``standardize`` parameter (for maskers, glm, decoders...) are no longer supported. Use ``standardize="z_score_sample"`` instead of ``True`` and ``None`` instead of ``False`` (:gh:`6553` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The parameter name ``filename`` of the :meth:`~nilearn.plotting.displays.BaseSlicer.savefig` will be permanently replaced by ``output_file`` in version 0.17.0 (:gh:`6471` by `Aniket Singh Yadav`_).
