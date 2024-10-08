"""Helper function for load_compcor."""

prefix_compcor = {
    "temporal_anat_combined": ["t", "a"],
    "temporal_anat_separated": ["t", "a", "w", "c"],
    "temporal": ["t"],
    "anat_combined": ["a"],
    "anat_separated": ["a", "w", "c"],
}
anat_masker = {
    "combined": ["combined"],
    "separated": ["WM", "CSF"],
    None: None,
}


def find_compcor(confounds_json, compcor, n_compcor):
    """Build list for the number of compcor components.

    Parameters
    ----------
    confounds_json : :obj:`dict`
        Dictionary of confounds from the confounds.json file.
    compcor : :obj:`str`
        Compcor strategy to use. Options are "temporal_anat", "temporal",
        "anat", or "combined".
    n_compcor : :obj:`int` or :obj:`str`
        Number of compcor components to retain. If "all", all components
        are retained.

    Returns
    -------
    collector : :obj:`list`
        List of compcor components to retain.
    """
    prefix_set, anat_mask = _check_compcor_method(compcor)

    collector = []
    for prefix in prefix_set:
        # all possible compcor confounds in order
        all_compcor_name = [
            comp for comp in confounds_json if f"{prefix}_comp_cor" in comp
        ]
        # filter by prefix first (anat vs temporal)
        compcor_cols_filt = _prefix_confound_filter(prefix, all_compcor_name)
        if prefix == "a":
            # apply acompor mask option if relevant, and select top components
            compcor_cols_filt = _acompcor_mask(
                confounds_json, anat_mask, compcor_cols_filt, n_compcor
            )
        else:
            # select top components
            compcor_cols_filt = _select_compcor(compcor_cols_filt, n_compcor)
        # Aggregate components across all masks
        collector += compcor_cols_filt
    return collector


def _select_compcor(compcor_cols, n_compcor):
    """Retain a specified number of compcor components.

    Parameters
    ----------
    compcor_cols : :obj:`list`
        List of compcor components column names filtered by user
        selected CompCor strategy.
    n_compcor : :obj:`int` or :obj:`str`
        Number of compcor components to retain. If "all", all components
        are retained.

    Returns
    -------
    compcor_cols : :obj:`list`
        List of compcor components column names to retain.
    """
    # only select if not "all", or less components are requested than there
    # actually is
    if (n_compcor != "all") and (n_compcor < len(compcor_cols)):
        compcor_cols = compcor_cols[:n_compcor]
    return compcor_cols


def _check_compcor_method(compcor):
    """Load compcor options and check if method is acceptable.

    Parameters
    ----------
    compcor : :obj: `str`
        Compcor strategy to use. Options are "temporal_anat", "temporal",
        "anat", or "combined".

    Returns
    -------
    prefix_set : :obj:`list`
        List of prefixes to use for compcor components.
    anat_mask : :obj:`list`
        List of anatomical masks to use for acompcor.
    """
    # get relevant prefix from compcor strategy
    prefix_set = prefix_compcor[compcor]
    # get relevant compcor mask
    check_masktype = compcor.split("_")
    anat_mask_type = None if len(check_masktype) == 1 else check_masktype[-1]
    anat_mask = anat_masker[anat_mask_type]
    return prefix_set, anat_mask


def _acompcor_mask(confounds_json, anat_mask, compcor_cols_filt, n_compcor):
    """Filter according to acompcor mask(s) and select top components.

    Parameters
    ----------
    confounds_json : :obj: `dict`
        Dictionary of confounds from the confounds.json file.
    anat_mask : :obj:`list`
        List of anatomical masks to use for acompcor.
    compcor_cols_filt : :obj:`list`
        List of compcor components column names filtered by user
        selected CompCor strategy.
    n_compcor : :obj:`int` or :obj:`str`
        Number of compcor components to retain. If "all", all components
        are retained.

    Returns
    -------
    collector : :obj:`list`
        List of compcor components column names to retain.
    """
    collector = []
    for mask in anat_mask:
        cols = _json_mask(compcor_cols_filt, confounds_json, mask)
        cols = _select_compcor(cols, n_compcor)
        collector += cols
    return collector


def _json_mask(compcor_cols_filt, confounds_json, mask):
    """Extract anat compcor components with a given mask.

    Parameters
    ----------
    compcor_cols_filt : :obj:`list`
        List of compcor components column names filtered by user
        selected CompCor strategy.
    confounds_json : :obj:`dict`
        Dictionary of confounds from the confounds.json file.
    mask : :obj:`str`
        Mask to use for acompcor.

    Returns
    -------
    compcor_cols_filt : :obj:`list`
        List of compcor components column names filtered by type of
        acompcor mask.
    """
    return [
        compcor_col
        for compcor_col in compcor_cols_filt
        if confounds_json[compcor_col]["Mask"] in mask
    ]


def _prefix_confound_filter(prefix, all_compcor_name):
    """Get confound columns by prefix and acompcor mask.

    Parameters
    ----------
    prefix : :obj:`str`
        Prefix to use for compcor components.
    all_compcor_name : :obj:`list`
        List of all compcor components column names.

    Returns
    -------
    compcor_cols_filt : :obj:`list`
        List of compcor components column names filtered by prefix.
    """
    compcor_cols_filt = []
    for nn in range(len(all_compcor_name)):
        nn_str = str(nn).zfill(2)
        compcor_col = f"{prefix}_comp_cor_{nn_str}"
        compcor_cols_filt.append(compcor_col)
    return compcor_cols_filt
