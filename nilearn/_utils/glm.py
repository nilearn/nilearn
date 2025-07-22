from collections.abc import Iterable


def coerce_to_dict(input_arg):
    """Construct a dict from the provided arg.

    If input_arg is:
      dict or None then returns it unchanged.

      string or collection of Strings or Sequence[int],
      returns a dict {str(value): value, ...}

    Parameters
    ----------
    input_arg : String or Collection[str or Int or Sequence[Int]]
     or Dict[str, str or np.array] or None
        Can be of the form:
         'string'
         ['string_1', 'string_2', ...]
         list/array
         [list/array_1, list/array_2, ...]
         {'string_1': list/array1, ...}

    Returns
    -------
    input_args: Dict[str, np.array or str] or None

    """
    if input_arg is None:
        return None
    if not isinstance(input_arg, dict):
        if (
            isinstance(input_arg, Iterable)
            and not isinstance(input_arg[0], Iterable)
        ) or isinstance(input_arg, str):
            input_arg = [input_arg]
        input_arg = {str(contrast_): contrast_ for contrast_ in input_arg}
    return input_arg


def make_stat_maps(
    model, contrasts, output_type="z_score", first_level_contrast=None
):
    """Given a model and contrasts, return the corresponding z-maps.

    Parameters
    ----------
    model : FirstLevelModel or SecondLevelModel object
        Must have a fitted design matrix(ces).

    contrasts : Dict[str, ndarray or str]
        Dict of contrasts for a first or second level model.
        Corresponds to the contrast_def for the FirstLevelModel
        (nilearn.glm.first_level.FirstLevelModel.compute_contrast)
        & second_level_contrast for a SecondLevelModel
        (nilearn.glm.second_level.SecondLevelModel.compute_contrast)

    output_type : :obj:`str`, default='z_score'
        The type of statistical map to retain from the contrast.

        .. versionadded:: 0.9.2

    %(first_level_contrast)s

        .. versionadded:: 0.12.0

    Returns
    -------
    statistical_maps : Dict[str, img] or Dict[str, Dict[str, img]]
        Dict of statistical z-maps keyed to contrast names/titles.

    See Also
    --------
    nilearn.glm.first_level.FirstLevelModel.compute_contrast
    nilearn.glm.second_level.SecondLevelModel.compute_contrast

    """
    # for second level flm
    if not hasattr(model, "hrf_model"):
        return {
            contrast_name: model.compute_contrast(
                contrast_data,
                output_type=output_type,
                first_level_contrast=first_level_contrast,
            )
            for contrast_name, contrast_data in contrasts.items()
        }

    return {
        contrast_name: model.compute_contrast(
            contrast_data,
            output_type=output_type,
        )
        for contrast_name, contrast_data in contrasts.items()
    }
