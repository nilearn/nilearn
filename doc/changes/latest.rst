.. currentmodule:: nilearn

Version 0.15.0dev
=================

..
    Each changelog entry should begin with one of the following badges:
    - :bdg-primary:`Doc`
    - :bdg-secondary:`Maint`
    - :bdg-success:`API`
    - :bdg-info:`Plotting`
    - :bdg-warning:`Test`
    - :bdg-danger:`Deprecation`
    - :bdg-dark:`Code`

NEW
---

Fixes
-----

Enhancements
------------

Changes
-------

- :bdg-success:`API` Rename the ``subjects`` parameter of :func:`~nilearn.datasets.fetch_haxby` to ``n_subjects``, retain ``subjects`` as a deprecated alias, and centralize subject-selection validation used by functional dataset fetchers (:gh:`6462` by `Mohammad Sadeghi Hardengi`_).
