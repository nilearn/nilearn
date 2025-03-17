"""Functions for generating BIDS-compliant GLM outputs."""

import inspect
import json
import warnings
from pathlib import Path

import numpy as np

from nilearn import __version__
from nilearn._utils import logger
from nilearn._utils.glm import coerce_to_dict, make_stat_maps
from nilearn._utils.helpers import is_matplotlib_installed


def _generate_model_metadata(out_file, model):
    """Generate a sidecar JSON file containing model metadata.

    .. versionadded:: 0.9.2

    Parameters
    ----------
    out_file : :obj:`str`
        Output JSON filename, to be created by the function.
    model : :obj:`~nilearn.glm.first_level.FirstLevelModel` or
            :obj:`~nilearn.glm.second_level.SecondLevelModel`
        First- or second-level model from which to save outputs.
    """
    # Define which FirstLevelModel attributes are BIDS compliant and which
    # should be bundled in a new "ModelParameters" field.

    model_metadata = {
        "Description": "A statistical map generated by Nilearn.",
        "ModelParameters": model._attributes_to_dict(),
    }

    with Path(out_file).open("w") as f_obj:
        json.dump(model_metadata, f_obj, indent=4, sort_keys=True)


def _generate_dataset_description(out_file, model_level):
    """Generate a BIDS dataset_description.json file with relevant metadata.

    .. versionadded:: 0.9.2

    If the dataset_description already exists only the GeneratedBy section
    is extended.

    Parameters
    ----------
    out_file : :obj:`pathlib.Path`
        Output JSON filename, to be created by the function.
    model_level : str
        The level of the model.
    """
    repo_url = "https://github.com/nilearn/nilearn"

    GeneratedBy = {
        "Name": "nilearn",
        "Version": __version__,
        "Description": (f"A Nilearn {model_level}-level GLM."),
        "CodeURL": (f"{repo_url}/releases/tag/{__version__}"),
    }

    if out_file.exists():
        with out_file.open() as f_obj:
            dataset_description = json.load(f_obj)
        if dataset_description.get("GeneratedBy"):
            dataset_description["GeneratedBy"].append(GeneratedBy)
    else:
        dataset_description = {
            "BIDSVersion": "1.9.0",
            "DatasetType": "derivative",
            "GeneratedBy": [GeneratedBy],
        }

    with out_file.open("w") as f_obj:
        json.dump(dataset_description, f_obj, indent=4, sort_keys=True)


def save_glm_to_bids(
    model, contrasts, contrast_types=None, out_dir=".", prefix=None, **kwargs
):
    """Save :term:`GLM` results to :term:`BIDS`-like files.

    .. versionadded:: 0.9.2

    Parameters
    ----------
    model : :obj:`~nilearn.glm.first_level.FirstLevelModel` or \
            :obj:`~nilearn.glm.second_level.SecondLevelModel`
        First- or second-level model from which to save outputs.

    contrasts : :obj:`str` or array of shape (n_col) or :obj:`list` \
                of (:obj:`str` or array of shape (n_col)) or :obj:`dict`
        Contrast definitions.

        If a dictionary is passed then it must be a dictionary of
        'contrast name': 'contrast weight' key-value pairs.
        The contrast weights may be strings, lists, or arrays.

        Arrays may be 1D or 2D, with 1D arrays typically being
        t-contrasts and 2D arrays typically being F-contrasts.

    contrast_types : None or :obj:`dict` of :obj:`str`, default=None
        An optional dictionary mapping some
        or all of the :term:`contrast` names to
        specific contrast types ('t' or 'F').
        If None, all :term:`contrast` types will
        be automatically inferred based on the :term:`contrast` arrays
        (1D arrays are t-contrasts, 2D arrays are F-contrasts).
        Keys in this dictionary must match the keys in the ``contrasts``
        dictionary, but only those contrasts
        for which :term:`contrast` type must be
        explicitly set need to be included.

    out_dir : :obj:`str` or :obj:`pathlib.Path`, default="."
        Output directory for files. Default is current working directory.

    prefix : :obj:`str` or None, default=None
        String to prepend to generated filenames.
        If a string is provided, '_' will be added to the end.

    kwargs : extra keywords arguments to pass to ``model.generate_report``
        See :func:`nilearn.reporting.make_glm_report` for more details.
        Can be any of the following: ``title``, ``bg_img``, ``threshold``,
        ``alpha``, ``cluster_threshold``, ``height_control``,
        ``min_distance``, ``plot_type``, ``display_mode``,
        ``two_sided``, ``cut_coords``.

    Warnings
    --------
    The files generated by this function are a best approximation of
    appropriate names for GLM-based BIDS derivatives.
    However, BIDS does not currently have GLM-based derivatives supported in
    the specification, and there is no guarantee that the files created by
    this function will be BIDS-compatible if and when the specification
    supports model derivatives.

    Notes
    -----
    This function writes files for the following:

    - Modeling software information (``dataset_description.json``)
    - Model-level metadata (``statmap.json``)
    - Model design matrix (``design.tsv``)
    - Model design metadata (``design.json``)
    - Model design matrix figure (``design.svg``)
    - Model error (``stat-errorts_statmap.nii.gz``)
    - Model r-squared (``stat-rsquared_statmap.nii.gz``)
    - Contrast :term:`'parameter estimates'<Parameter Estimate>`
      (``contrast-[name]_stat-effect_statmap.nii.gz``)
    - Variance of the contrast parameter estimates
      (``contrast-[name]_stat-variance_statmap.nii.gz``)
    - Contrast test statistics
      (``contrast-[name]_stat-[F|t]_statmap.nii.gz``)
    - Contrast p- and z-values
      (``contrast-[name]_stat-[p|z]_statmap.nii.gz``)
    - Contrast weights figure (``contrast-[name]_design.svg``)

    """
    # Import here to avoid circular imports
    if is_matplotlib_installed():
        from nilearn._utils.plotting import (
            generate_constrat_matrices_figures,
            generate_design_matrices_figures,
        )
    else:
        warnings.warn(
            ("No plotting back-end detected. Output will be missing figures."),
            UserWarning,
            stacklevel=2,
        )

    # fail early if invalid paramaeters to pass to generate_report()
    allowed_extra_kwarg = [
        x
        for x in inspect.signature(model.generate_report).parameters
        if x not in ["contrasts", "input"]
    ]
    for key in kwargs:
        if key not in allowed_extra_kwarg:
            raise ValueError(
                f"Extra key-word arguments must be one of: "
                f"{allowed_extra_kwarg}\n"
                f"Got: {key}"
            )

    if not isinstance(prefix, str):
        prefix = ""
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    contrasts = coerce_to_dict(contrasts)
    for k, v in contrasts.items():
        if not isinstance(k, str):
            raise ValueError(f"contrast names must be strings, not {type(k)}")

        if not isinstance(v, (str, np.ndarray, list)):
            raise ValueError(
                "contrast definitions must be strings or array_likes, "
                f"not {type(v)}"
            )

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    dset_desc_file = out_dir / "dataset_description.json"
    _generate_dataset_description(dset_desc_file, model.__str__())

    model._generate_filenames_output(
        prefix, contrasts, contrast_types, out_dir
    )

    out_dir = model._reporting_data["filenames"]["dir"]
    out_dir.mkdir(exist_ok=True, parents=True)

    verbose = model.verbose

    if hasattr(model, "design_matrices_"):
        design_matrices = model.design_matrices_
    else:
        design_matrices = [model.design_matrix_]

    if is_matplotlib_installed():
        logger.log("Generating design matrices figures...", verbose=verbose)
        # TODO: Assuming that cases of multiple design matrices correspond to
        # different runs. Not sure if this is correct. Need to check.
        generate_design_matrices_figures(
            design_matrices, output=model._reporting_data["filenames"]
        )

    if is_matplotlib_installed():
        logger.log("Generating contrast matrices figures...", verbose=verbose)
        generate_constrat_matrices_figures(
            design_matrices,
            contrasts,
            output=model._reporting_data["filenames"],
        )

    for i_run, design_matrix in enumerate(design_matrices, start=1):
        run_str = f"run-{i_run}_" if len(design_matrices) > 1 else ""

        # Save design matrix and associated figure
        design_matrix.to_csv(
            out_dir / f"{prefix}{run_str}design.tsv",
            sep="\t",
            index=False,
        )

        if model.__str__() == "First Level Model":
            with (out_dir / f"{prefix}{run_str}design.json").open(
                "w"
            ) as f_obj:
                json.dump(
                    {"RepetitionTime": model.t_r},
                    f_obj,
                    indent=4,
                    sort_keys=True,
                )

    # Model metadata
    # TODO: Determine optimal mapping of model metadata to BIDS fields.
    metadata_file = out_dir / f"{prefix}statmap.json"
    _generate_model_metadata(metadata_file, model)

    logger.log(
        "Generating contrast-level statistical maps...", verbose=verbose
    )
    statistical_maps = make_stat_maps(model, contrasts, output_type="all")
    for contrast_name, contrast_maps in statistical_maps.items():
        for output_type in contrast_maps:
            img = statistical_maps[contrast_name][output_type]
            filename = model._reporting_data["filenames"]["statistical_maps"][
                contrast_name
            ][output_type]
            img.to_filename(out_dir / filename)

    logger.log("Saving contrast-level statistical maps...", verbose=verbose)
    _write_model_level_statistical_maps(model, prefix, out_dir)

    logger.log("Generating HTML...", verbose=verbose)
    glm_report = model.generate_report(
        contrasts=contrasts, verbose=verbose - 1, **kwargs
    )
    glm_report.save_as_html(out_dir / f"{prefix}report.html")


def _write_model_level_statistical_maps(model, prefix, out_dir):
    if model.__str__() == "Second Level Model":
        model_level_mapping = {
            "residuals": f"{prefix}stat-errorts_statmap.nii.gz",
            "r_square": f"{prefix}stat-rsquared_statmap.nii.gz",
        }
        for attr, map_name in model_level_mapping.items():
            stat_map_to_save = getattr(model, attr)
            stat_map_to_save.to_filename(out_dir / map_name)

    else:
        if hasattr(model, "design_matrices_"):
            design_matrices = model.design_matrices_
        else:
            design_matrices = [model.design_matrix_]

        for i_run, _ in enumerate(design_matrices):
            run_str = f"run-{i_run + 1}_" if len(design_matrices) > 1 else ""
            model_level_mapping = {
                "residuals": f"{prefix}{run_str}stat-errorts_statmap.nii.gz",
                "r_square": f"{prefix}{run_str}stat-rsquared_statmap.nii.gz",
            }
            for attr, map_name in model_level_mapping.items():
                img = getattr(model, attr)
                stat_map_to_save = img[i_run]
                stat_map_to_save.to_filename(out_dir / map_name)
