"""Helper function for _load_compcor."""


prefix_compcor = {"full": ["t", "a"], "temp": ["t"], "anat": ["a"]}
anat_masker = {True: ["combined"], False: ["WM", "CSF"], None: None}


def _find_compcor(confounds_json, compcor, n_compcor, acompcor_combined):
    """Builds list for the number of compcor components."""
    prefix_set, anat_mask = _check_compcor_method(compcor, acompcor_combined)

    collector = []
    for prefix in prefix_set:
        # all possible compcor confounds in order, mixing different types of mask
        all_compcor_name = [
            comp for comp in confounds_json.keys() if f"{prefix}_comp_cor" in comp
        ]
        # filter by prefix first (anat vs temp)
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
    """Retain a specified number of compcor components."""
    # only select if not "auto", or less components are requested than there actually is
    if (n_compcor != "auto") and (n_compcor < len(compcor_cols)):
        compcor_cols = compcor_cols[0:n_compcor]
    return compcor_cols


def _check_compcor_method(compcor, acompcor_combined):
    """Load compcor options and check if method is acceptable."""
    # get relevant prefix from compcor strategy
    prefix_set = prefix_compcor[compcor]
    # get relevant compcore mask
    anat_mask = anat_masker[acompcor_combined]
    if ("a" in prefix_set) and (anat_mask is None):
        raise ValueError(
            f"acompcor_combined must set to True or False. Got {acompcor_combined}"
        )
    return prefix_set, anat_mask


def _acompcor_mask(confounds_json, anat_mask, compcor_cols_filt, n_compcor):
    """Filter according to acompcor mask(s) and select top components."""
    collector = []
    for mask in anat_mask:
        cols = _json_mask(compcor_cols_filt, confounds_json, mask)
        cols = _select_compcor(cols, n_compcor)
        collector += cols
    return collector


def _json_mask(compcor_cols_filt, confounds_json, mask):
    """Extract anat compcor components from a given mask."""
    cols = []
    for compcor_col in compcor_cols_filt:
        if confounds_json[compcor_col]["Mask"] in mask:
            cols.append(compcor_col)
    return cols


def _prefix_confound_filter(prefix, all_compcor_name):
    """Get confound columns by prefix and acompcor mask."""
    compcor_cols_filt = []
    for nn in range(len(all_compcor_name)):
        nn_str = str(nn).zfill(2)
        compcor_col = f"{prefix}_comp_cor_{nn_str}"
        compcor_cols_filt.append(compcor_col)
    return compcor_cols_filt
