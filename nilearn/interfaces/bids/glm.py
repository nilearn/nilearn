"""Functions for generating BIDS-compliant GLM outputs."""

import inspect
import json
import warnings
from collections.abc import Iterable
from pathlib import Path

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

    Returns
    -------
    model : :obj:`~nilearn.glm.first_level.FirstLevelModel` or \
            :obj:`~nilearn.glm.second_level.SecondLevelModel`

            .. versionadded:: 0.11.2dev

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
            ("No plotting backend detected. Output will be missing figures."),
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

    contrasts = coerce_to_dict(contrasts)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    dset_desc_file = out_dir / "dataset_description.json"
    _generate_dataset_description(dset_desc_file, model.__str__())

    model = model._generate_filenames_output(
        prefix, contrasts, contrast_types, out_dir
    )

    filenames = model._reporting_data["filenames"]

    out_dir = filenames["dir"]
    out_dir.mkdir(exist_ok=True, parents=True)

    verbose = model.verbose

    if model.__str__() == "Second Level Model":
        design_matrices = [model.design_matrix_]
    else:
        design_matrices = model.design_matrices_

    if not isinstance(prefix, str):
        prefix = ""
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    if is_matplotlib_installed():
        logger.log("Generating design matrices figures...", verbose=verbose)
        # TODO: Assuming that cases of multiple design matrices correspond to
        # different runs. Not sure if this is correct. Need to check.
        generate_design_matrices_figures(design_matrices, output=filenames)

        logger.log("Generating contrast matrices figures...", verbose=verbose)
        generate_constrat_matrices_figures(
            design_matrices,
            contrasts,
            output=filenames,
        )

    for i_run, design_matrix in enumerate(design_matrices):
        filename = Path(
            filenames["design_matrices_dict"][i_run]["design_matrix_tsv"]
        )

        # Save design matrix and associated figure
        design_matrix.to_csv(
            out_dir / filename,
            sep="\t",
            index=False,
        )

        if model.__str__() == "First Level Model":
            with (out_dir / filename.with_suffix(".json")).open("w") as f_obj:
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

    logger.log("Saving contrast-level statistical maps...", verbose=verbose)
    statistical_maps = make_stat_maps(model, contrasts, output_type="all")
    for contrast_name, contrast_maps in statistical_maps.items():
        for output_type in contrast_maps:
            img = statistical_maps[contrast_name][output_type]
            filename = filenames["statistical_maps"][contrast_name][
                output_type
            ]
            img.to_filename(out_dir / filename)

    logger.log("Saving model level statistical maps...", verbose=verbose)
    _write_model_level_statistical_maps(model, out_dir)

    logger.log("Generating HTML...", verbose=verbose)
    # generate_report can just rely on the name of the files
    # stored in the model instance.
    # temporarily drop verbosity to avoid generate_report
    # logging the same thing
    model.verbose -= 1
    glm_report = model.generate_report(**kwargs)
    model.verbose += 1
    glm_report.save_as_html(out_dir / f"{prefix}report.html")

    return model


def _write_model_level_statistical_maps(model, out_dir):
    for i_run, model_level_mapping in model._reporting_data["filenames"][
        "model_level_mapping"
    ].items():
        for attr, map_name in model_level_mapping.items():
            img = getattr(model, attr)
            stat_map_to_save = img[i_run] if isinstance(img, Iterable) else img
            stat_map_to_save.to_filename(out_dir / map_name)
