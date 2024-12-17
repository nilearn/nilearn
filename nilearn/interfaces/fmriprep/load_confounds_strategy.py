"""Predefined denoising strategies.

Authors: Hao-Ting Wang, Pierre Bellec
"""

import warnings

from . import load_confounds

# defining a preset strategy with python dictionary:
# key:
#   name for the strategy
# value:
#   a dictionary containing the parameters from `load_confounds`
#   and associated values.
#   compulsory:
#       strategy (as the value defines the other relevant parameters)
preset_strategies = {
    "simple": {
        "strategy": ("high_pass", "motion", "wm_csf"),
        "motion": "full",
        "wm_csf": "basic",
        "global_signal": None,
        "demean": True,
    },
    "scrubbing": {
        "strategy": ("high_pass", "motion", "wm_csf", "scrub"),
        "motion": "full",
        "wm_csf": "full",
        "scrub": 5,
        "fd_threshold": 0.2,  # updated here and doc to 0.5 in v0.13
        "std_dvars_threshold": 3,  # updated here and doc to 1.5 in v0.13
        "global_signal": None,
        "demean": True,
    },
    "compcor": {
        "strategy": ("high_pass", "motion", "compcor"),
        "motion": "full",
        "n_compcor": "all",
        "compcor": "anat_combined",
        "global_signal": None,
        "demean": True,
    },
    "ica_aroma": {
        "strategy": ("high_pass", "wm_csf", "ica_aroma"),
        "wm_csf": "basic",
        "ica_aroma": "full",
        "global_signal": None,
        "demean": True,
    },
}


def load_confounds_strategy(img_files, denoise_strategy="simple", **kwargs):
    """
    Use preset strategy to load confounds from :term:`fMRIPrep`.

    `load_confounds_strategy` provides an interface to select confounds
    based on past literature with limited parameters for user customization.

    .. versionadded:: 0.9.0

    Parameters
    ----------
    img_files : :obj:`str` or :obj:`list` of :obj:`str`.
        Processed nii.gz/dtseries.nii/func.gii file reside in a
        :term:`fMRIPrep` generated functional derivative directory (i.e.The
        associated confound files should be in the same directory as the image
        file). As long as the image file, confound related tsv and json are in
        the same directory with BIDS-compliant names,
        :func:`nilearn.interfaces.fmriprep.load_confounds` can retrieve the
        relevant files correctly.

        - `nii.gz` or `dtseries.nii`: path to files, optionally as a list.
        - `func.gii`: list of a pair of paths to files, optionally as a list
          of lists.

    denoise_strategy : :obj:`str`, default="simple"
        Name of preset denoising strategies. Each strategy has a set of
        associated configurable parameters. For customiseable parameters,
        please see the table in Notes.

        - 'simple': Load confounds for a simple denoising strategy commonly
          used in resting state functional connectivity, described in
          :footcite:t:`Fox2005`. With the global signal regression,
          this approach can remove confounds
          without compromising the temporal degrees of freedom.
        - 'scrubbing': Load confounds for scrubbing described in
          :footcite:t:`Power2012`. This approach can reliably remove the
          impact of high motion volumes in functional connectome, however, it
          might not be suitable with subjects with high motion (more than 50%
          timeseries flagged as high motion). One should adjust the threshold
          based on the characteristics of the dataset, or remove high motion
          subjects from the dataset.
        - 'compcor': Load confounds using the CompCor strategy from
          :footcite:t:`Behzadi2007`. CompCor estimates noise through principal
          component analysis on regions that are unlikely to contain signal.
          Thus it might not be a suitable approach for researchers who want
          explicit description of the source of noise. Empirically, Compcor
          has shown similar effect of removing physiological noise as methods
          that explicitly model and remove physiology signals. Compcor can
          suffer from loss of temporal degrees of freedom when using explained
          variance as the noise component estimation as the number of compcor
          component can be really high. Please refer to :term:`fMRIPrep`
          documentation for more details.

          .. versionadded:: 0.10.3
            `golobal_signal` is now a tunable parameter for compcor.

        - 'ica_aroma': Load confounds for non-aggresive ICA-AROMA strategy
          described in :footcite:t:`Pruim2015`. The strategy requires
          :term:`fMRIPrep` outputs generated with `--use-aroma` suffixed with
          `desc-smoothAROMAnonaggr_bold`. ICA-AROMA increases the run time of
          :term:`fMRIPrep`, however, the strategy performs well in various
          benchmarks (:footcite:t:`Ciric2017`, :footcite:t:`Parker2018`).
          See Notes for more details about this option.

    Other keyword arguments:
        See additional parameters associated with `denoise_strategy` in
        Notes and refer to the documentation of
        :func:`nilearn.interfaces.fmriprep.load_confounds`.

    Returns
    -------
    confounds : :class:`pandas.DataFrame`, or :obj:`list` of \
        :class:`pandas.DataFrame`
        A reduced version of :term:`fMRIPrep` confounds based on selected
        strategy and flags.
        The columns contains the labels of the regressors.

    sample_mask : None, :class:`numpy.ndarray` or, :obj:`list` of \
        :class:`numpy.ndarray` or None
        When no volume requires removal, the value is None.
        Otherwise, shape: (number of scans - number of volumes removed, )
        The index of the niimgs along time/fourth dimension for valid volumes
        for subsequent analysis.
        This attribute should be passed to parameter `sample_mask` of
        :class:`nilearn.maskers.NiftiMasker` or
        :func:`nilearn.signal.clean`.
        Volumes are removed if flagged as following:

        - Non-steady-state volumes (if present)
        - Motion outliers detected by scrubbing

    Notes
    -----
    1. The following table details the default options of each preset
       strategies. Parameters with `*` denote customizable parameters. Please
       see :func:`nilearn.interfaces.fmriprep.load_confounds`.

        ========= ========= ====== ====== ============= ===== ============ \
        =================== ============== ========= ========= ======
        strategy  high_pass motion wm_csf global_signal scrub fd_threshold \
        std_dvars_threshold compcor        n_compcor ica_aroma demean
        ========= ========= ====== ====== ============= ===== ============ \
        =================== ============== ========= ========= ======
        simple    True      full*  basic* None*         N/A   N/A          \
        N/A                 N/A            N/A       N/A       True*
        scrubbing True      full*  full   None*         5*    0.2*         \
        3*                  N/A            N/A       N/A       True*
        compcor   True      full*  N/A    None*         N/A   N/A          \
        N/A                 anat_combined* all*      N/A       True*
        ica_aroma True      N/A    basic* None*         N/A   N/A          \
        N/A                 N/A            N/A       full      True*
        ========= ========= ====== ====== ============= ===== ============ \
        =================== ============== ========= ========= ======

    2. ICA-AROMA is implemented in two steps in :footcite:t:`Pruim2015`:

        i. A non-aggressive denoising immediately after :term:`ICA`
        classification.
        A linear regression estimates signals with all independent
        components as predictors. A partial regression is then applied to
        remove variance associated with noise independent components.
        :term:`fMRIPrep` performs this step and generates files in
        `MNI152NLin6Asym` template, suffixed with
        `desc-smoothAROMAnonaggr_bold`.

        One can produce `desc-smoothAROMAnonaggr_bold` in other spatial
        templates, please refer to :term:`fMRIPrep` documentation on ICA-AROMA
        `<https://fmriprep.org/en/latest/usage.html#fmriprep.cli.parser-_build_parser-[deprecated]-options-for-running-ica_aroma>`_

        ii. Confound regression step (mean signals from WM and CSF).
        Confound regressors generated by this function with
        `denoise_strategy="ica_aroma"`.

        For more discussion regarding choosing the nuisance regressors before
        or after denoising with ICA-AROMA has a detriment on outcome measures,
        please see notebook 5.
        `<https://github.com/nipreps/fmriprep-notebooks/>`_

    See Also
    --------
    :func:`nilearn.interfaces.fmriprep.load_confounds`

    References
    ----------
    .. footbibliography::

    """
    default_parameters = preset_strategies.get(denoise_strategy, False)
    if not default_parameters:
        raise KeyError(
            f"Provided strategy '{denoise_strategy}' is not a "
            "preset strategy. Valid strategy: "
            f"{preset_strategies.keys()}"
        )

    check_parameters = list(default_parameters.keys())
    check_parameters.remove("strategy")
    # ICA-AROMA only accept the non-aggressive strategy
    # ignore user passed value
    if "ica_aroma" in default_parameters:
        check_parameters.remove("ica_aroma")

    user_parameters, not_needed = _update_user_inputs(
        kwargs, default_parameters, check_parameters
    )

    # raise warning about parameters not needed
    if not_needed:
        warnings.warn(
            "The following parameters are not needed for the "
            f"selected strategy '{denoise_strategy}': {not_needed}; "
            f"parameters accepted: {check_parameters}",
            stacklevel=2,
        )
    return load_confounds(img_files, **user_parameters)


def _update_user_inputs(kwargs, default_parameters, check_parameters):
    """Update keyword parameters with user inputs if applicable.

    Parameters
    ----------
    kwargs : :obj:`dict`
        Keyword parameters passed to `load_confounds_strategy`.

    default_parameters : :obj:`dict`
        Default parameters for the selected pre-set strategy.

    check_parameters : :obj:`list`
        List of parameters that are applicable to the selected pre-set
        strategy.

    Returns
    -------
    parameters : :obj:`dict`
        Updated valid parameters for `load_confounds`.

    not_needed : :obj:`list`
        List of parameters that are not applicable to the selected
        pre-set strategy.
    """
    parameters = default_parameters.copy()
    # update the parameter with user input
    not_needed = []
    for key in check_parameters:
        value = kwargs.pop(key, None)  # get user input
        if value is not None:
            parameters[key] = value
        # global_signal parameter is not in default strategy, but
        # applicable to every strategy other than compcor
        # global signal strategy will only be added if user has passed a
        # recognizable value to the global_signal parameter
        if key == "global_signal":
            if isinstance(value, str):
                parameters["strategy"] += ("global_signal",)
            else:  # remove global signal if not updated
                parameters.pop("global_signal", None)
    # collect remaining parameters in kwargs that are not needed
    not_needed = list(kwargs.keys())
    return parameters, not_needed
