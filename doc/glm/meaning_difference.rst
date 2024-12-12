.. _meaning_difference:

==================================================
Difference in meanings between different toolboxes
==================================================

.. topic:: **Page summary**

    * SPM has the same meaning of analysis levels, compared to Nilearn's models
        * First-level: analyze across runs
        * Second-level: group-level analysis
    * FSL has a different meaning
        * First-level: analyze one session for one subject
        * Second-level: analyze across sessions for one subject
        * Third-level: group-level analysis

These differences can be put in the table below:

+----------------------------------------------+----------------+------------------+---------------+
| Analyze                                      | nilearn        | SPM              | FSL           |
+==============================================+================+==================+===============+
| One session, one subject                     | First-level    | First-level      | First\-level  |
+----------------------------------------------+----------------+------------------+---------------+
| More than one session, one subject           | First\-level   | First\-level     | Second\-level |
+----------------------------------------------+----------------+------------------+---------------+
| More than or equal to one session,           | Second\-level  | Second\-level    | Third\-level  |
| more than one subject                        |                |                  |               |
+----------------------------------------------+----------------+------------------+---------------+

*Table showing the differences of the meaning for level of models between toolboxes\/libraries*

Statistical Parametric Mapping (SPM)
====================================

SPM uses the same notation as Nilearn for analysis levels,
with a note that a session still refers to an imaging session or a run,
and within a run there could be multiple conditions (for example congruent and incongruent).
In this case, `SPM`_ provided `tutorials`_ and documentation, including `lectures`_,
which one could learn to analyze their own fMRI data with the meaning of analysis levels being as follows:

    * `First-level analysis in SPM`_: Analyze across sessions for a subject
      (meaning more than one session of one subject)
    * `Second-level analysis in SPM`_: Analyze across several subjects
       (meaning more than one subject with one or more sessions per subject).
       This is also known as **group-level analysis** which test
       if the average estimate across subjects is statistically significant.

.. _SPM: https://www.fil.ion.ucl.ac.uk/spm/docs/
.. _tutorials: https://www.fil.ion.ucl.ac.uk/spm/docs/tutorials/
.. _lectures: https://www.fil.ion.ucl.ac.uk/spm/docs/courses/fmri_vbm/recordings/glm/
.. _First-level analysis in SPM: https://andysbrainbook.readthedocs.io/en/latest/SPM/SPM_Short_Course/SPM_Statistics/SPM_06_Stats_Running_1stLevel_Analysis.html
.. _Second-level analysis in SPM: https://andysbrainbook.readthedocs.io/en/latest/SPM/SPM_Short_Course/SPM_08_GroupAnalysis.html

FMRIB Software Library (FSL)
============================

FSL uses a slightly different set of meanings for analysis levels,
but with a note that a session still refers to an imaging session or a run.
Specifically, `FEAT`_, FSL software tool for model-based fMRI data analysis,
provides GUI to run analysis on imaging data for first-level and higher-level analysis
with the `terminology meaning`_ as follows:

    * `First-level analysis in FSL`_: Analyze each session's data by getting the parameter and contrast estimates
       (meaning one session of one subject)
    * `Second-level analysis in FSL`_: Analyze across sessions for a subject
      via averaging the parameter and contrast estimates
      within each subject (i.e., more than one session of one subject)
    * `Third-level analysis in FSL`_: Analyze across several subjects or group-level analysis
      on the averaged contrast estimates for all subjects within the group
      (meaning more than one subject with one or more sessions per subject)

.. _FEAT: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT/UserGuide#Appendix_A:_Brief_Overview_of_GLM_Analysis
.. _terminology meaning: https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FEAT/UserGuide#First-level_or_Higher-level_Analysis.3F
.. _First-level analysis in FSL: https://andysbrainbook.readthedocs.io/en/latest/fMRI_Short_Course/Statistics/06_Stats_Running_1stLevel_Analysis.html
.. _Second-level analysis in FSL: https://andysbrainbook.readthedocs.io/en/latest/fMRI_Short_Course/fMRI_07_2ndLevelAnalysis.html
.. _Third-level analysis in FSL: https://andysbrainbook.readthedocs.io/en/latest/fMRI_Short_Course/fMRI_08_3rdLevelAnalysis.html
