import warnings

warnings.warn(f"Backend changed to {new_backend}...", stacklevel=2)
warnings.warn(
    "Affine is different across subjects. Realignment on first subject",
    stacklevel=2,
)
warnings.warn(
    "Random walker only segments unlabeled areas, where labels == 0. No zero valued areas in labels were found.",
    stacklevel=2,
)
warnings.warn(
    "Random walker only segments unlabeled areas, where labels == 0. No zero valued areas in labels were found.",
    stacklevel=2,
)
warnings.warn(
    "Random walker received no seed label. Returning provided labels.",
    stacklevel=2,
)
warnings.warn(
    "Random walker only segments unlabeled areas, where labels == 0. Data provided only contains isolated seeds.",
    stacklevel=2,
)
warnings.warn(f"Couldn't find {mni152_brain_mask} (for testing)", stacklevel=2)
warnings.warn(
    f"Maximum number of iterations {max_iter} reached without getting to the requested tolerance level {tol}.",
    stacklevel=2,
)
warnings.warn(
    "input signals do not all have unit variance. This can lead to numerical instability.",
    stacklevel=2,
)
warnings.warn(msg, RuntimeWarning, stacklevel=2)
warnings.warn(
    "Newton-Raphson step did not converge. This may indicate a badly conditioned model.",
    stacklevel=2,
)
warnings.warn(
    "Maximum number of iterations reached without getting to the requested tolerance level.",
    stacklevel=2,
)
warnings.warn("Could not find dataset description.", stacklevel=2)
warnings.warn(f"An error occurred while fetching {file_}.", stacklevel=2)
warnings.warn(
    "Deprecation warning for fetcher. In release 0.13, this fetcher will return a dictionary.",
    stacklevel=2,
)
warnings.warn(
    "In release 0.13, this fetcher will return a dictionary. Please update your code.",
    category=DeprecationWarning,
    stacklevel=2,
)
warnings.warn(message, DeprecationWarning, stacklevel=2)
warnings.warn(
    "In release 0.13, this fetcher will return a dictionary. Please update your code.",
    category=DeprecationWarning,
    stacklevel=2,
)
warnings.warn(
    f"Warning: there are only {max_subjects} subjects.", stacklevel=2
)
warnings.warn(
    "Wrong value for 'n_subjects'. The maximum value will be used instead ('n_subjects=94').",
    stacklevel=2,
)
warnings.warn(
    "Some dependencies of nilearn.plotting package seem to be missing."
    "\nThey can be installed with:\n",
    stacklevel=2,
)
warnings.warn(f"Backend changed to {new_backend}...", stacklevel=2)
warnings.warn(
    "Affine is different across subjects." " Realignement on first subject ",
    stacklevel=2,
)
warnings.warn(
    "Random walker only segments unlabeled areas, where "
    "labels == 0. No zero valued areas in labels were "
)


# datasets/atlas.py
warnings.warn(
    message="In release 0.13, this fetcher will return a dictionary",
    category=DeprecationWarning,
    stacklevel=2,
)

# datasets/func.py
warnings.warn(f"Warning: there are only {max_subjects} subjects", stacklevel=2)
warnings.warn(
    "Wrong value for 'n_subjects' (%d). The maximum value will be used instead ('n_subjects=94')",
    stacklevel=2,
)
warnings.warn("Warning: there are only 16 subjects!", stacklevel=2)
warnings.warn(f"Warning: there are only {max_subjects} subjects", stacklevel=2)
warnings.warn(
    f"Wrong value for n_subjects={n_subjects}. The maximum value (for age_group={age_group}) will be used.",
    stacklevel=2,
)
warnings.warn(
    'If `dataset_version` is not "ds000030_R1.0.4", `urls` must be specified. Downloading "ds000030_R1.0.4".',
    stacklevel=2,
)
warnings.warn(
    f"{mat_err!s}. An events.tsv file cannot be generated.", stacklevel=2
)

# datasets/neurovault.py
warnings.warn(
    "No word weight could be loaded. Vectorizing Neurosynth words failed.",
    stacklevel=2,
)
warnings.warn(
    f"Could not update metadata for image {image_info['id']}. Most likely because you do not have the required permissions.",
    stacklevel=2,
)
warnings.warn(
    "Neurovault download stopped early: "
    f"too many downloads failed in a row ({n_consecutive_fails}).",
    stacklevel=2,
)

warnings.warn(
    "You specified contradictory collection ids, "
    "one in the image filters and one in the ",
    stacklevel=2,
)
warnings.warn(
    "You don't have write access to neurovault dir: "
    f"{neurovault_data_dir}. ",
    stacklevel=2,
)
warnings.warn(
    "You specified a value for `image_filter` but the "
    "default filters in `image_terms` still apply. ",
    stacklevel=2,
)
warnings.warn(
    "You specified a value for `collection_filter` but the "
    "default filters in `collection_terms` still apply. ",
    stacklevel=2,
)
warnings.warn(
    "Only 403 subjects are available in the "
    "DARTEL-normalized version of the dataset. ",
    stacklevel=2,
)
warnings.warn(
    "Only 415 subjects are available in the "
    "non-DARTEL-normalized version of the dataset. ",
    stacklevel=2,
)
warnings.warn(f"{_GENERAL_MESSAGE}", stacklevel=2)
warnings.warn(
    f"parameter '{param_name}' should be a sequence of iterables "
    f"(e.g., {{param_name: [[1, 10, 100]]}}) to benefit from",
    stacklevel=2,
)
warnings.warn(
    "Use a custom estimator at your own risk "
    "of the process not working as intended.",
    stacklevel=2,
)
warnings.warn(
    "groups parameter is specified but "
    "cv parameter is not set to custom CV splitter. ",
    stacklevel=2,
)
warnings.warn(
    "After clustering and screening, the decoding model will "
    f"be trained only on {n_final_features} features. ",
    stacklevel=2,
)
warnings.warn(
    f"'slice_time_ref' provided ({slice_time_ref}) is different "
    f"from the value found in the BIDS dataset ",
    stacklevel=2,
)
warnings.warn(
    '"verbose" option requires the package Matplotlib. '
    "Please install it using `pip install matplotlib`.",
    stacklevel=2,
)
warnings.warn(
    "Resampling binary images with continuous or "
    "linear interpolation. This might lead to ",
    stacklevel=2,
)


# base_masker.py corrections
warnings.warn(
    "Starting in version 0.12, 3D images will be transformed to 1D arrays.",
    stacklevel=2,
)

warnings.warn(
    f"[{self.__class__.__name__}.fit] Generation of a mask has been",
    stacklevel=2,
)

# multi_nifti_masker.py corrections
warnings.warn(
    "Masking strategy 'template' is deprecated. Please use 'whole-brain-template' instead.",
    stacklevel=2,
)

# nifti_labels_masker.py corrections
warnings.warn(mpl_unavail_msg, category=ImportWarning, stacklevel=2)

warnings.warn(
    f"After resampling the label image to the data image, the following labels were removed: {labels_diff}.",
    stacklevel=2,
)

# nifti_maps_masker.py corrections
warnings.warn(mpl_unavail_msg, category=ImportWarning, stacklevel=2)

warnings.warn(
    f"Setting number of displayed maps to {n_maps}.",
    category=UserWarning,
    stacklevel=2,
)

# nifti_masker.py corrections
warnings.warn(
    "Masking strategy 'template' is deprecated. Please use 'whole-brain-template' instead.",
    stacklevel=2,
)

warnings.warn(
    "Starting in version 0.12, 3D images will be transformed to 1D arrays.",
    stacklevel=2,
)

warnings.warn(
    "imgs are being resampled to the mask_img resolution. "
    "This process is memory intensive. You might want to provide a mask "
    "of the same resolution as imgs to save memory.",
    stacklevel=2,
)

warnings.warn(mpl_unavail_msg, category=ImportWarning, stacklevel=2)

# nifti_spheres_masker.py corrections
warnings.warn(
    "The imgs you have fed into fit_transform() contains NaN values which will be converted to zeroes.",
    stacklevel=2,
)

warnings.warn(mpl_unavail_msg, category=ImportWarning, stacklevel=2)
