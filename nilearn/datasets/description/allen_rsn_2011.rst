.. _allen_2011_atlas:

Allen 2011 atlas
================

Access
------
See :func:`nilearn.datasets.fetch_atlas_allen_2011`.

Notes
-----
Collection of :term:`resting-state` network templates
extracted from the esting-state networks of
603 healthy adolescents and adults (mean age: 23.4 years, range: 12-71 years)

The provided images are in MNI152 space.

Data were collected on the same scanner, preprocessed using
an automated analysis pipeline based in SPM, and studied using group independent component
analysis. RSNs were identified and evaluated in terms of three primary outcome measures:
time course spectral power, spatial map intensity, and functional network connectivity.

For more information on this dataset's structure, see:
https://trendscenter.org/data/

Slices and label of the RSN ICs:
https://www.frontiersin.org/files/Articles/2093/fnsys-05-00002-HTML/image_m/fnsys-05-00002-g004.jpg

Direct download link from OSF: https://osf.io/hrcku

Content
-------
    :"maps": T-maps of all 75 unthresholded components.
    :"rsn28": T-maps of 28 RSNs included in :footcite:t:`Allen2011`.
    :"networks": string list containing the names for the 28 RSNs.
    :"rsn_indices": dict[rsn_name] -> list of int, indices in the "maps" file of the 28 RSNs.
    :"comps": The aggregate :term:`ICA` Components.

References
----------

.. footbibliography::

License
-------
unknown
