.. _talairach_atlas:

Talairach atlas
===============

Access
------
See :func:`nilearn.datasets.fetch_atlas_talairach`.

Notes
-----

The anatomical region labels were electronically derived from axial sectional images in the 1988 Talairach Atlas.
While 3-D coordinates are precise, quantitative descriptions of locations (and location variances),
more traditional (but less quantitative) anatomical descriptions,
such as surface anatomy (sulci/gyri) and architectonic areas,
are needed to allow comparisons between coordinate-based data
and those not using standardized coordinates.

The Talairach Atlas has been digitized and manually traced into a volume-occupant hierarchy of anatomical regions.
Hemispheres, lobes, lobules, gyri and nuclei have been outlined and labeled.
Gray matter, white matter and CSF regions will also be defined.
For cerebral cortex, all Brodmann areas have been traced and expanded into 3-D volumes.
The Talairach Atlas includes Brodmann area (BA) labels,
but lacks explicit boundaries between BAs and has several inconsistencies.
Explicit boundaries were rule-based and primarily justified by using Brodmann's 1909 monograph.

Brodmann area, gyrus, lobe and hemisphere region labels are available in the Talairach Labels database.

For more information,
see https://www.talairach.org,
:footcite:t:`talairach_atlas`,
:footcite:t:`Lancaster2000`,
and :footcite:t:`Lancaster1997`.

Direct download link from OSF: https://www.talairach.org/talairach.nii

Content
-------
    :'maps': 3D Nifti image, values are integers corresponding to indices in the
             list of labels.

    :'labels': Annotations (see https://www.talairach.org/labels.html)

    :'lut': color look up table

References
----------

.. footbibliography::


License
-------
unknown
