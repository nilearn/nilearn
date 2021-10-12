"""Predefined denoising strategies.

Authors: Hao-Ting Wang, Pierre Bellec
"""
import warnings
from . import fmriprep_confounds

# defining a preset strategy with python dictionary:
# key:
#   name for the strategy
# value:
#   a dictionary containing the parameters from `fmriprep_confounds`
#   and associated values.
#   compulsory:
#       strategy (as the value defines the other relevant parameters)
preset_strategies = {
    "simple": {
        "strategy":
            ["high_pass", "non_steady_state", "motion", "wm_csf"],
        "motion": "basic",
        "wm_csf": "basic",
        "global_signal": None,
        "demean": True
    },
    "scrubbing": {
        "strategy":
            ["high_pass", "non_steady_state", "motion", "wm_csf", "scrub"],
        "motion": "full",
        "wm_csf": "full",
        "scrub": 5,
        "fd_thresh": 0.2,
        "std_dvars_thresh": 3,
        "global_signal": None,
        "demean": True
    },
    "compcor": {
        "strategy":
            ["high_pass", "non_steady_state", "motion", "compcor"],
        "motion": "full",
        "n_compcor": "all",
        "compcor": "anat_combined",
        "demean": True
    },
    "ica_aroma": {
        "strategy":
            ["high_pass", "non_steady_state", "wm_csf", "ica_aroma"],
        "wm_csf": "basic",
        "ica_aroma": "full",
        "global_signal": None,
        "demean": True
    }
}


def fmriprep_confounds_strategy(img_files, denoise_strategy="simple",
                                **kwargs):
    """
    Use preset strategy to load confounds from :term:`fMRIPrep`.

    `fmriprep_confounds_strategy` provides an interface to select confounds
    based on past literature with limited parameters for user customisation.

    .. versionadded:: 0.8.2

    Parameters
    ----------
    img_files : path to processed image files, optionally as a list.
        Processed nii.gz/dtseries.nii/func.gii file reside in a
        :term:`fMRIPrep` generated functional derivative directory (i.e.The
        associated confound files should be in the same directory as the image
        file). As long as the image file, confound related tsv and json are in
        the same directory with BIDS-complied names, `fmriprep_confounds` can
        retrieve the relevant files correctly.

        - `nii.gz` or `dtseries.nii`: path to files, optionally as a list.
        - `func.gii`: list of a pair of paths to files, optionally as a list
          of lists.

    denoise_strategy : {'simple', 'srubbing', 'compcor', 'ica_aroma'}
        Name of preset denoising strategies. Each strategy has a set of
        associated configurable parameters. For the documentation on
        additional parameters, please refer to
        :func:`nilearn.input_data.fmriprep_confounds`.

        - 'simple': Load confounds for a simple denosing strategy commonly
          used in resting state functional connectivity, described in
          :footcite:`Fox2005`. Default as: full motion parameters,
          full WM/CSF signals, and  high pass filter, with an option to
          extract global signal confounds.
          Additional parameters: motion, wm_csf, global_signal, demean
        - 'srubbing': Load confounds for scrubbing describbed in
          :footcite:`Power2012`.Default as: full motion parameters,
          full WM/CSF signals, remove segment smaller than 5 continuous
          volumes (see docstring of
          :func:`nilearn.input_data.fmriprep_confounds`),
          high pass filter, with an option to extract global signal confounds.
          Additional parameters: motion, wm_csf, scrub, fd_thresh,
          std_dvars_thresh, global_signal, demean
        - 'compcor': Load confounds using the CompCor strategy from
          :footcite:`BEHZADI200790`.Default with full motion parameters,
          high pass filter, and anatomical compcor with combined mask.
          Additional parameters: motion, n_compcor, compcor, demean
        - 'ica_aroma': Load confounds for non-aggresive ICA-AROMA strategy
          described in :footcite:`Pruim2015`. The strategy requires
          :term:`fMRIPrep` outputs generated with `--use-aroma` suffixed with
          `desc-smoothAROMAnonaggr_bold`. See notes for more details about
          this option.
          Additional parameters: wm_csf, global_signal, demean

    Other keyword arguments:
        See additional parameters associated with denoise_strategy and refer
        to the documentation of :func:`nilearn.input_data.fmriprep_confounds`

    Returns
    -------
    confounds : pandas.DataFrame, or list of
        A reduced version of :term:`fMRIPrep` confounds based on selected
        strategy and flags.
        An intercept is automatically added to the list of confounds.
        The columns contains the labels of the regressors.

    sample_mask : None, numpy.ndarray, or list of
        When no volumns require removal, the value is None.
        Otherwise, shape: (number of scans - number of volumes removed, )
        The index of the niimgs along time/fourth dimension for valid volumes
        for subsequent analysis.
        This attribute should be passed to parameter `sample_mask` of
        :class:`nilearn.input_data.NiftiMasker` or
        :func:`nilearn.signal.clean`.
        Volumns are removed if flagged as following:

        - Non-steady-state volumes (if present)
        - Motion outliers detected by scrubbing

    Notes
    -----
    ICA-AROMA is implemented in two steps in :footcite:`Pruim2015`:

    1. A non-aggressive denoising immediately after ICA classification.
      A linear regression estimates signals with all independent
      components as predictors. A partial regression is then applied to
      remove variance associated with noise independent components.
      :term:`fMRIPrep` performs this step and generates files in
      `MNI152NLin6Asym` template, suffixed with
      `desc-smoothAROMAnonaggr_bold`.

      One can produce `desc-smoothAROMAnonaggr_bold` in other spatial
      templates, please refer to :term:`fMRIPrep` documentation on ICA-AROMA
      `<https://fmriprep.org/en/latest/workflows.html#ica-aroma>`_

    2. Confound regression step (mean signals from WM and CSF).
      Confound regressors generated by this function with
      `denoise_strategy="ica_aroma"`.

    For more discussion regarding choosing the nuisance regressors before or
    after denoising with ICA-AROMA has a detriment on outcome measures,
    please see notebook 5.
    `<https://github.com/nipreps/fmriprep-notebooks/>`_

    References
    -----------
    .. footbibliography::

    """
    default_parameters = preset_strategies[denoise_strategy].copy()
    check_parameters = list(default_parameters.keys())
    check_parameters.remove("strategy")
    # ICA-AROMA only accept the non-aggressive strategy
    # ignore user passed value
    if "ica_aroma" in default_parameters:
        check_parameters.remove("ica_aroma")

    user_parameters, not_needed = _update_user_inputs(kwargs,
                                                      default_parameters,
                                                      check_parameters)

    # raise warning about parameters not needed
    if not_needed:
        warnings.warn("The following parameters are not needed for the "
                      f"selected strategy '{denoise_strategy}': {not_needed}; "
                      f"parameters accepted: {check_parameters}"
                      )
    return fmriprep_confounds(img_files, **user_parameters)


def _update_user_inputs(kwargs, default_parameters, check_parameters):
    """Update keyword parameters with user inputs if applicable."""
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
        # recognisable value to the global_signal parameter
        if key == "global_signal":
            if isinstance(value, str):
                parameters["strategy"].append("global")
            else:  # remove global signal if not updated
                parameters.pop("global_signal", None)
    # collect remaining parameters in kwargs that are not needed
    not_needed = list(kwargs.keys())
    return parameters, not_needed
