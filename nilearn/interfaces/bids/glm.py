"""Functions for generating BIDS-compliant GLM outputs."""

import json
import warnings
from pathlib import Path

import numpy as np

from nilearn import __version__


def _clean_contrast_name(contrast_name):
    """Remove prohibited characters from name and convert to camelCase.

    .. versionadded:: 0.9.2

    BIDS filenames, in which the contrast name will appear as a
    contrast-<name> key/value pair, must be alphanumeric strings.

    Parameters
    ----------
    contrast_name : :obj:`str`
        Contrast name to clean.

    Returns
    -------
    new_name : :obj:`str`
        Contrast name converted to alphanumeric-only camelCase.
    """
    new_name = contrast_name[:]

    # Some characters translate to words
    new_name = new_name.replace("-", " Minus ")
    new_name = new_name.replace("+", " Plus ")
    new_name = new_name.replace(">", " Gt ")
    new_name = new_name.replace("<", " Lt ")

    # Others translate to spaces
    new_name = new_name.replace("_", " ")

    # Convert to camelCase
    new_name = new_name.split(" ")
    new_name[0] = new_name[0].lower()
    new_name[1:] = [c.title() for c in new_name[1:]]
    new_name = " ".join(new_name)

    # Remove non-alphanumeric characters
    new_name = "".join(ch for ch in new_name if ch.isalnum())

    # Let users know if the name was changed
    if new_name != contrast_name:
        warnings.warn(
            f'Contrast name "{contrast_name}" changed to "{new_name}"',
            stacklevel=3,
        )
    return new_name


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
    PARAMETER_ATTRIBUTES = [
        "drift_model",
        "hrf_model",
        "standardize",
        "high_pass",
        "target_shape",
        "signal_scaling",
        "drift_order",
        "scaling_axis",
        "smoothing_fwhm",
        "target_affine",
        "slice_time_ref",
    ]

    if hasattr(model, "hrf_model") and model.hrf_model == "fir":
        PARAMETER_ATTRIBUTES.append("fir_delays")

    # Fields for a nested section of the dictionary
    # The ModelParameters field is an ad-hoc way to retain useful info.
    PARAMETER_ATTRIBUTES.sort()
    model_attributes = {
        attr_name: getattr(model, attr_name)
        for attr_name in PARAMETER_ATTRIBUTES
        if hasattr(model, attr_name)
    }

    for key, value in model_attributes.items():
        if isinstance(value, (np.ndarray)):
            model_attributes[key] = value.tolist()

    model_metadata = {
        "Description": "A statistical map generated by Nilearn.",
        "ModelParameters": model_attributes,
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
    model_level : {1, 2}
        The level of the model. 1 means a first-level model.
        2 means a second-level model.
    """
    repo_url = "https://github.com/nilearn/nilearn"

    GeneratedBy = {
        "Name": "nilearn",
        "Version": __version__,
        "Description": (
            "A Nilearn "
            f"{'first' if model_level == 1 else 'second'}"
            "-level GLM."
        ),
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

    out_dir : :obj:`str` or :obj:`pathlib.Path`, optional
        Output directory for files. Default is current working directory.

    prefix : :obj:`str` or None, default=None
        String to prepend to generated filenames.
        If a string is provided, '_' will be added to the end.

    kwargs : extra keywords arguments to pass to ``model.generate_report``
        See :func:`nilearn.reporting.make_glm_report` for more details.
        Can be any of the following: ``title``, ``bg_img``, ``threshold``,
        ``alpha``, ``cluster_threshold``, ``height_control``,
        ``min_distance``, ``plot_type``, ``display_mode``.

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
    from nilearn.plotting.matrix_plotting import (
        plot_contrast_matrix,
        plot_design_matrix,
    )
    from nilearn.reporting.glm_reporter import _make_stat_maps

    allowed_extra_kwarg = [
        "title",
        "bg_img",
        "threshold",
        "alpha",
        "cluster_threshold",
        "height_control",
        "min_distance",
        "plot_type",
        "display_mode",
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

    if isinstance(contrasts, list):
        contrasts = {c: c for c in contrasts}
    elif isinstance(contrasts, str):
        contrasts = {contrasts: contrasts}

    for k, v in contrasts.items():
        if not isinstance(k, str):
            raise ValueError(f"contrast names must be strings, not {type(k)}")

        if not isinstance(v, (str, np.ndarray, list)):
            raise ValueError(
                "contrast definitions must be strings or array_likes, "
                f"not {type(v)}"
            )

    model_level = _model_level(model)

    if model_level == 2:
        sub_directory = "group"
    else:
        sub_directory = (
            prefix.split("_")[0] if prefix.startswith("sub-") else ""
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    dset_desc_file = out_dir / "dataset_description.json"
    _generate_dataset_description(dset_desc_file, model_level)

    out_dir = out_dir / sub_directory
    out_dir.mkdir(exist_ok=True, parents=True)

    if not isinstance(contrast_types, dict):
        contrast_types = {}

    # Write out design matrices to files.
    if hasattr(model, "design_matrices_"):
        design_matrices = model.design_matrices_
    else:
        design_matrices = [model.design_matrix_]

    # TODO: Assuming that cases of multiple design matrices correspond to
    # different runs. Not sure if this is correct. Need to check.
    for i_run, design_matrix in enumerate(design_matrices):
        run_str = f"run-{i_run + 1}_" if len(design_matrices) > 1 else ""

        # Save design matrix and associated figure
        design_matrix.to_csv(
            out_dir / f"{prefix}{run_str}design.tsv",
            sep="\t",
            index=False,
        )

        dm_fig = plot_design_matrix(design_matrix)
        dm_fig.figure.savefig(out_dir / f"{prefix}{run_str}design.svg")

        if model_level == 1:
            with (out_dir / f"{prefix}{run_str}design.json").open(
                "w"
            ) as f_obj:
                json.dump(
                    {"RepetitionTime": model.t_r},
                    f_obj,
                    indent=4,
                    sort_keys=True,
                )

        # Save contrast plots as well
        for contrast_name, contrast_data in contrasts.items():
            contrast_plot = plot_contrast_matrix(
                contrast_data,
                design_matrix,
                colorbar=True,
            )
            contrast_plot.set_xlabel(contrast_name)
            contrast_plot.figure.set_figheight(2)
            contrast_name = _clean_contrast_name(contrast_name)
            constrast_fig_file = (
                out_dir
                / f"{prefix}{run_str}contrast-{contrast_name}_design.svg"
            )
            contrast_plot.figure.savefig(constrast_fig_file)

    # Model metadata
    # TODO: Determine optimal mapping of model metadata to BIDS fields.
    metadata_file = out_dir / f"{prefix}statmap.json"
    _generate_model_metadata(metadata_file, model)

    # Write out contrast-level statistical maps
    statistical_maps = _make_stat_maps(model, contrasts, output_type="all")
    for contrast_name, contrast_maps in statistical_maps.items():
        # Extract stat_type
        contrast_matrix = contrasts[contrast_name]
        # Strings and 1D arrays are assumed to be t-contrasts
        if isinstance(contrast_matrix, str) or (contrast_matrix.ndim == 1):
            stat_type = "t"
        else:
            stat_type = "F"

        # Override automatic detection with explicit type if provided
        stat_type = contrast_types.get(contrast_name, stat_type)

        # Convert the contrast name to camelCase
        contrast_name = _clean_contrast_name(contrast_name)

        # Contrast-level images
        contrast_level_mapping = {
            "effect_size": (
                f"{prefix}contrast-{contrast_name}_stat-effect_statmap.nii.gz"
            ),
            "stat": (
                f"{prefix}contrast-{contrast_name}_stat-{stat_type}_statmap"
                ".nii.gz"
            ),
            "effect_variance": (
                f"{prefix}contrast-{contrast_name}_stat-variance_statmap"
                ".nii.gz"
            ),
            "z_score": (
                f"{prefix}contrast-{contrast_name}_stat-z_statmap.nii.gz"
            ),
            "p_value": (
                f"{prefix}contrast-{contrast_name}_stat-p_statmap.nii.gz"
            ),
        }
        # Rename keys
        renamed_contrast_maps = {
            contrast_level_mapping.get(k, k): v
            for k, v in contrast_maps.items()
        }

        for map_name, img in renamed_contrast_maps.items():
            img.to_filename(out_dir / map_name)

    _write_model_level_statistical_maps(model, prefix, out_dir)

    # Add html report
    glm_report = model.generate_report(contrasts=contrasts, **kwargs)
    glm_report.save_as_html(out_dir / f"{prefix}report.html")


def _model_level(model):
    from nilearn.glm.first_level import FirstLevelModel

    return 1 if isinstance(model, FirstLevelModel) else 2


def _write_model_level_statistical_maps(model, prefix, out_dir):
    if _model_level(model) == 2:
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
