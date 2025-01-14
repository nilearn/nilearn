.. _oasis_maps:

OASIS volume based morphometry maps
===================================

Access
------
See :func:`nilearn.datasets.fetch_oasis_vbm`.

Content
-------
:'gray_matter_maps': Nifti images with gray matter density probability
:'white_matter_maps': Nifti images with white matter density probability maps
:'ext_vars': Behavioral information on the participants
:'data_usage_agreement': Text file containing the data usage agreement

Notes
-----
The Open Access Structural Imaging Series (OASIS) is a project
dedicated to making brain imaging data openly available to the public.

OASIS is made available by the Washington University Alzheimer's Disease
Research Center, Dr. Randy Buckner at the Howard Hughes Medical
Institute (HHMI) at Harvard University, the Neuroinformatics Research
Group (NRG) at Washington University School of Medicine, and the Biomedical
Informatics Research Network (BIRN).

.. admonition:: Data Usage Agreement
   :class: attention

   Using data available through the OASIS project requires agreeing with
   the Data Usage Agreement that can be found at
   https://sites.wustl.edu/oasisbrains/

In the DARTEL version, original Oasis data have been preprocessed
with the following steps:

1. Dimension swapping (technically required for subsequent steps)
2. Brain Extraction
3. Segmentation with SPM8
4. Normalization using DARTEL algorithm
5. Modulation
6. Replacement of NaN values with 0 in gray/white matter density maps.
7. Resampling to reduce shape and make it correspond to the shape of
   the non-DARTEL data (fetched with dartel_version=False).
8. Replacement of values < 1e-4 with zeros to reduce the file size.

In the non-DARTEL version, the following steps have been performed instead:

1. Dimension swapping (technically required for subsequent steps)
2. Brain Extraction
3. Segmentation and normalization to a template with SPM8
4. Modulation
5. Replacement of NaN values with 0 in gray/white matter density maps.

An archive containing the gray and white matter density probability maps
for the 416 available subjects is provided. Gross outliers are removed and
filtered by this data fetcher (DARTEL: 13 outliers; non-DARTEL: 1 outlier)
Externals variates (age, gender, estimated intracranial volume,
years of education, socioeconomic status, dementia score) are provided
in a CSV file that is a copy of the original Oasis CSV file. The current
downloader loads the CSV file and keeps only the lines corresponding to
the subjects that are actually demanded.

For more information this dataset's structure:
https://sites.wustl.edu/oasisbrains/,
:footcite:t:`OASISbrain`,
and :footcite:t:`Marcus2007`.

References
----------
.. footbibliography::

License
-------
Provided under an open access data use agreement (DUA).
