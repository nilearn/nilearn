.. _meaning_difference:

==================================================
Difference in meanings between different toolboxes
==================================================

.. topic:: **Page summary**

    * SPM has the same meaning of analysis levels, compared to Nilearn's models
        * First level: analyze across runs
        * Second level: group level analysis
    * FSL has a different meaning
        * First level: analyze one run for one subject
        * Second level: analyze across runs for one subject
        * Third level: group level analysis

These differences can be put in the table below:

+--------------------------------+----------------+------------------+---------------+
| Analyze                        | nilearn        | SPM              | FSL           |
+================================+================+==================+===============+
| One run, one subject           | First level    | First level      | First level   |
+--------------------------------+----------------+------------------+---------------+
| More than one run, one subject | First level    | First level      | Second level  |
+--------------------------------+----------------+------------------+---------------+
| More than or equal to one run, | Second level   | Second level     | Third level   |
| more than one subject          |                |                  |               |
+--------------------------------+----------------+------------------+---------------+

*Table showing the differences of the meaning for level of models between toolboxes\/libraries*

Statistical Parametric Mapping (SPM)
====================================

SPM uses the same notation as Nilearn for analysis levels,
with a note that in SPM terminology ``session`` refers to an imaging run,
and within a run there could be multiple conditions (for example congruent and incongruent).
`SPM`_ provides `tutorials`_ and documentation, including `lectures`_,
to help users analyze their own fMRI data with the meaning of analysis levels being as follows:

- `First level analysis in SPM`_: Analyze across runs for a subject
  (meaning more than one run of one subject)
- `Second level analysis in SPM`_: Analyze across several subjects
  (meaning more than one subject with one or more run per subject).
  This is also known as **group level analysis** which test
  if the average estimate across subjects is statistically significant.

.. admonition:: Fixed effects analyses
    :class: hint

    One important difference between SPM and Nilearn,
    is that the typical first level workflow in SPM
    will create a single design matrix for all runs
    and thus run a single model at the subject level
    (see `First level analysis in SPM`_).
    Nilearn will instead create one design matrix per run,
    and run one model per run
    (see for example :ref:`this report with 2 runs <two_runs_glm>`).
    In Nilearn, to compute summary statistics across runs
    you can use the method :meth:`~nilearn.glm.first_level.FirstLevelModel.compute_contrast`
    or the function :func:`~nilearn.glm.compute_fixed_effects`.

    .. seealso::

        :ref:`sphx_glr_auto_examples_04_glm_first_level_plot_two_runs_model.py`

.. _SPM: https://www.fil.ion.ucl.ac.uk/spm/docs/
.. _tutorials: https://www.fil.ion.ucl.ac.uk/spm/docs/tutorials/
.. _lectures: https://www.fil.ion.ucl.ac.uk/spm/docs/courses/fmri_vbm/recordings/glm/
.. _First level analysis in SPM: https://andysbrainbook.readthedocs.io/en/latest/SPM/SPM_Short_Course/SPM_Statistics/SPM_06_Stats_Running_1stLevel_Analysis.html
.. _Second level analysis in SPM: https://andysbrainbook.readthedocs.io/en/latest/SPM/SPM_Short_Course/SPM_08_GroupAnalysis.html


FMRIB Software Library (FSL)
============================

FSL uses a slightly different set of meanings for analysis levels,
but with a note that a ``session`` still refers to an imaging run.
Specifically, `FEAT`_, FSL software tool for model-based fMRI data analysis,
provides a CLI and GUI to run analysis on imaging data
for first level and higher level analysis
with the `terminology meaning`_ as follows:

- `First level analysis in FSL`_: Analyze each run's data by getting the parameter and contrast estimates
  (meaning one run of one subject)
- `Second level analysis in FSL`_: Analyze across runs for a subject
  via averaging the parameter and contrast estimates
  within each subject (meaning more than one run of one subject)
- `Third level analysis in FSL`_: Analyze across several subjects or group level analysis
  on the averaged contrast estimates for all subjects within the group
  (meaning more than one subject with one or more runs per subject)

.. _FEAT: https://fsl.fmrib.ox.ac.uk/fsl/docs/#/task_fmri/feat/overview_of_glm_analysis
.. _terminology meaning: https://fsl.fmrib.ox.ac.uk/fsl/docs/#/task_fmri/feat/user_guide?id=feat-user-guide
.. _First level analysis in FSL: https://andysbrainbook.readthedocs.io/en/latest/fMRI_Short_Course/Statistics/06_Stats_Running_1stLevel_Analysis.html
.. _Second level analysis in FSL: https://andysbrainbook.readthedocs.io/en/latest/fMRI_Short_Course/fMRI_07_2ndLevelAnalysis.html
.. _Third level analysis in FSL: https://andysbrainbook.readthedocs.io/en/latest/fMRI_Short_Course/fMRI_08_3rdLevelAnalysis.html
