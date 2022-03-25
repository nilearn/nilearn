"""Functions for working with BIDS datasets."""
import json
import glob
import os
import warnings

import nilearn
from nilearn import glm
from nilearn.plotting.matrix_plotting import (
    plot_contrast_matrix,
    plot_design_matrix,
)
from nilearn.reporting.glm_reporter import _make_stat_maps


def get_bids_files(
    main_path,
    file_tag='*',
    file_type='*',
    sub_label='*',
    modality_folder='*',
    filters=None,
    sub_folder=True,
):
    """Search for files in a :term:`BIDS` dataset following given constraints.

    This utility function allows to filter files in the :term:`BIDS` dataset by
    any of the fields contained in the file names. Moreover it allows to search
    for specific types of files or particular tags.

    The provided filters have to correspond to a file name field, so
    any file not containing the field will be ignored. For example the filter
    ('sub', '01') would return all files corresponding to the first
    subject that specifically contain in the file name 'sub-01'. If more
    filters are given then we constraint the possible files names accordingly.

    Notice that to search in the derivatives folder, it has to be given as
    part of the main_path. This is useful since the current convention gives
    exactly the same inner structure to derivatives than to the main
    :term:`BIDS` dataset folder, so we can search it in the same way.

    Parameters
    ----------
    main_path : :obj:`str`
        Directory of the :term:`BIDS` dataset.

    file_tag : :obj:`str` accepted by glob, optional
        The final tag of the desired files. For example 'bold' if one is
        interested in the files related to the neuroimages.
        Default='*'.

    file_type : :obj:`str` accepted by glob, optional
        The type of the desired files. For example to be able to request only
        'nii' or 'json' files for the 'bold' tag.
        Default='*'.

    sub_label : :obj:`str` accepted by glob, optional
        Such a common filter is given as a direct option since it applies also
        at the level of directories. the label is what follows the 'sub' field
        in the :term:`BIDS` convention as 'sub-label'.
        Default='*'.

    modality_folder : :obj:`str` accepted by glob, optional
        Inside the subject and optional session folders a final level of
        folders is expected in the :term:`BIDS` convention that groups files
        according to different neuroimaging modalities and any other additions
        of the dataset provider. For example the 'func' and 'anat' standard
        folders. If given as the empty string '', files will be searched
        inside the sub-label/ses-label directories.
        Default='*'.

    filters : :obj:`list` of :obj:`tuple` (:obj:`str`, :obj:`str`), optional
        Filters are of the form (field, label). Only one filter per field
        allowed. A file that does not match a filter will be discarded.
        Filter examples would be ('ses', '01'), ('dir', 'ap') and
        ('task', 'localizer').

    sub_folder : :obj:`bool`, optional
        Determines if the files searched are at the level of
        subject/session folders or just below the dataset main folder.
        Setting this option to False with other default values would return
        all the files below the main directory, ignoring files in subject
        or derivatives folders.
        Default=True.

    Returns
    -------
    files : :obj:`list` of :obj:`str`
        List of file paths found.

    """
    filters = filters if filters else []
    if sub_folder:
        files = os.path.join(main_path, 'sub-*', 'ses-*')
        if glob.glob(files):
            files = os.path.join(
                main_path,
                'sub-%s' % sub_label,
                'ses-*',
                modality_folder,
                'sub-%s*_%s.%s' % (sub_label, file_tag, file_type),
            )
        else:
            files = os.path.join(
                main_path,
                'sub-%s' % sub_label,
                modality_folder,
                'sub-%s*_%s.%s' % (sub_label, file_tag, file_type),
            )
    else:
        files = os.path.join(main_path, '*%s.%s' % (file_tag, file_type))

    files = glob.glob(files)
    files.sort()
    if filters:
        files = [parse_bids_filename(file_) for file_ in files]
        for key, value in filters:
            files = [
                file_ for file_ in files if
                (key in file_ and file_[key] == value)
            ]
        return [ref_file['file_path'] for ref_file in files]

    return files


def parse_bids_filename(img_path):
    r"""Return dictionary with parsed information from file path.

    Parameters
    ----------
    img_path : :obj:`str`
        Path to file from which to parse information.

    Returns
    -------
    reference : :obj:`dict`
        Returns a dictionary with all key-value pairs in the file name
        parsed and other useful fields like 'file_path', 'file_basename',
        'file_tag', 'file_type' and 'file_fields'.

        The 'file_tag' field refers to the last part of the file under the
        :term:`BIDS` convention that is of the form \*_tag.type.
        Contrary to the rest of the file name it is not a key-value pair.
        This notion should be revised in the case we are handling derivatives
        since so far the convention will keep the tag prepended to any fields
        added in the case of preprocessed files that also end with another tag.
        This parser will consider any tag in the middle of the file name as a
        key with no value and will be included in the 'file_fields' key.

    """
    reference = {}
    reference['file_path'] = img_path
    reference['file_basename'] = os.path.basename(img_path)
    parts = reference['file_basename'].split('_')
    tag, type_ = parts[-1].split('.', 1)
    reference['file_tag'] = tag
    reference['file_type'] = type_
    reference['file_fields'] = []
    for part in parts[:-1]:
        field = part.split('-')[0]
        reference['file_fields'].append(field)
        # In derivatives is not clear if the source file name will
        # be parsed as a field with no value.
        if len(part.split('-')) > 1:
            value = part.split('-')[1]
            reference[field] = value
        else:
            reference[field] = None
    return reference


def save_glm_to_bids(
    model,
    contrasts,
    contrast_types=None,
    out_dir='.',
    prefix=None,
):
    """Save GLM results to BIDS-like files.

    Parameters
    ----------
    model : :obj:`~nilearn.glm.first_level.FirstLevelModel` or
            :obj:`~nilearn.glm.second_level.SecondLevelModel`
        First- or second-level model from which to save outputs.
    contrasts : :obj:`dict` of :obj:`str`-:obj:`numpy.ndarray` pairs
        A dictionary containing contrasts. Keys are contrast names,
        while values are arrays containing contrast weights.
        The arrays may be 1D or 2D.
    contrast_types : None or :obj:`dict` of :obj:`str`s, optional
        An optional dictionary mapping some or all of the contrast names to
        specific contrast types ('t' or 'F'). If None, all contrast types will
        be automatically inferred based on the contrast arrays
        (1D arrays are t-contrasts, 2D arrays are F-contrasts).
        Keys in this dictionary must match the keys in the ``contrasts``
        dictionary, but only those contrasts for which contrast type must be
        explicitly set need to be included.
        Default is None.
    out_dir : :obj:`str`, optional
        Output directory for files. Default is current working directory.
    prefix : :obj:`str` or None, optional
        String to prepend to generated filenames.
        If a string is provided, '_' will be added to the end.
        Default is None.

    Warning
    -------
    The files generated by this function are a best approximation of
    appropriate names for GLM-based BIDS derivatives.
    However, BIDS does not currently have GLM-based derivatives supported in
    the specification, and there is no guarantee that the files created by
    this function will be BIDS-compatible if and when the specification
    supports model derivatives.

    Notes
    -----
    This function writes files for the following:
    - Modeling software information (dataset_description.json)
    - Model-level metadata (statmap.json)
    - Model design matrix (design.tsv)
    - Model design matrix figure (design.svg)
    - Model error (stat-errorts_statmap.nii.gz)
    - Model r-squared (stat-rSquare_statmap.nii.gz)
    - Contrast parameter estimates (contrast-[name]_stat-effect_statmap.nii.gz)
    - Variance of the contrast parameter estimates
      (contrast-[name]_stat-variance_statmap.nii.gz)
    - Contrast test statistics (contrast-[name]_stat-[F|t]_statmap.nii.gz)
    - Contrast p- and z-values (contrast-[name]_stat-[p|z]_statmap.nii.gz)
    - Contrast weights figure (contrast-[name]_design.svg)
    """
    # Define which FirstLevelModel attributes are BIDS compliant and which
    # should be bundled in a new "ModelParameters" field.
    DATA_ATTRIBUTES = [
        't_r',
    ]
    PARAMETER_ATTRIBUTES = [
        'drift_model',
        'hrf_model',
        'standardize',
        'high_pass',
        'target_shape',
        'signal_scaling',
        'drift_order',
        'scaling_axis',
        'smoothing_fwhm',
        'target_affine',
        'slice_time_ref',
        'fir_delays',
    ]
    ATTRIBUTE_RENAMING = {
        't_r': 'RepetitionTime',
    }

    if isinstance(prefix, str) and not prefix.endswith('_'):
        prefix += '_'
    elif not isinstance(prefix, str):
        prefix = ''

    out_dir = os.path.abspath(out_dir)

    model_level = (
        1 if isinstance(model, glm.first_level.FirstLevelModel) else 2
    )

    if not isinstance(contrast_types, dict):
        contrast_types = {}

    # Write out design matrices to files.
    if hasattr(model, 'design_matrices_'):
        design_matrices = model.design_matrices_
    else:
        design_matrices = [model.design_matrix_]

    # TODO: Assuming that cases of multiple design matrices correspond to
    # different runs. Not sure if this is correct. Need to check.
    for i_run, design_matrix in enumerate(design_matrices):
        run_str = f'run-{i_run + 1}_' if len(design_matrices) > 1 else ''

        # Save design matrix and associated figure
        dm_file = os.path.join(
            out_dir,
            '{}{}design.tsv'.format(prefix, run_str),
        )
        design_matrix.to_csv(
            dm_file,
            sep='\t',
            line_terminator='\n',
            index=False,
        )

        dm_fig_file = os.path.join(
            out_dir,
            '{}{}design.svg'.format(prefix, run_str),
        )
        dm_fig = plot_design_matrix(design_matrix)
        dm_fig.figure.savefig(dm_fig_file)

        # Save contrast plots as well
        for contrast_name, contrast_data in contrasts.items():
            contrast_plot = plot_contrast_matrix(
                contrast_data,
                design_matrix,
                colorbar=True,
            )
            contrast_plot.set_xlabel(contrast_name)
            contrast_plot.figure.set_figheight(2)
            contrast_plot.figure.set_tight_layout(True)
            contrast_name = _clean_contrast_name(contrast_name)
            constrast_fig_file = os.path.join(
                out_dir,
                f'{prefix}{run_str}contrast-{contrast_name}_design.svg',
            )
            contrast_plot.figure.savefig(constrast_fig_file)

    statistical_maps = _make_stat_maps(
        model,
        contrasts,
        output_type='all',
    )

    # Model metadata
    # TODO: Determine optimal mapping of model metadata to BIDS fields.
    model_metadata_file = os.path.join(
        out_dir,
        '{}statmap.json'.format(prefix),
    )
    dataset_description_file = os.path.join(
        out_dir,
        'dataset_description.json',
    )

    # Fields for the top level of the dictionary
    DATA_ATTRIBUTES.sort()
    data_attributes = {
        attr_name: getattr(model, attr_name)
        for attr_name in DATA_ATTRIBUTES
        if hasattr(model, attr_name)
    }
    data_attributes = {
        ATTRIBUTE_RENAMING.get(k, k): v for k, v in data_attributes.items()
    }

    # Fields for a nested section of the dictionary
    # The ModelParameters field is an ad-hoc way to retain useful info.
    PARAMETER_ATTRIBUTES.sort()
    model_attributes = {
        attr_name: getattr(model, attr_name)
        for attr_name in PARAMETER_ATTRIBUTES
        if hasattr(model, attr_name)
    }
    model_attributes = {
        ATTRIBUTE_RENAMING.get(k, k): v for k, v in model_attributes.items()
    }

    model_metadata = {
        'Description': 'A statistical map generated by nilearn.',
        **data_attributes,
        'ModelParameters': model_attributes,
    }

    dataset_description = {
        'GeneratedBy': {
            'Name': 'nilearn',
            'Version': nilearn.__version__,
            'Description': 'A nilearn {} GLM.'.format(
                'first-level' if model_level == 1 else 'second-level'
            ),
            'CodeURL': (
                'https://github.com/nilearn/nilearn/releases/tag/'
                '{}'.format(nilearn.__version__)
            )
        }
    }

    with open(model_metadata_file, 'w') as fo:
        json.dump(model_metadata, fo, indent=4, sort_keys=True)

    with open(dataset_description_file, 'w') as fo:
        json.dump(dataset_description, fo, indent=4, sort_keys=True)

    # Write out statistical maps
    for contrast_name, contrast_maps in statistical_maps.items():
        # Extract stat_type
        contrast_matrix = contrasts[contrast_name]
        if contrast_matrix.ndim == 2:
            stat_type = 'F'
        else:
            stat_type = 't'

        # Override automatic detection with explicit type if provided
        stat_type = contrast_types.get(contrast_name, stat_type)

        contrast_name = _clean_contrast_name(contrast_name)

        # Contrast-level images
        contrast_level_mapping = {
            'effect_size':
                '{}contrast-{}_stat-effect_statmap.nii.gz'.format(
                    prefix,
                    contrast_name,
                ),
            'stat':
                '{}contrast-{}_stat-{}_statmap.nii.gz'.format(
                    prefix,
                    contrast_name,
                    stat_type,
                ),
            'effect_variance':
                '{}contrast-{}_stat-variance_statmap.nii.gz'.format(
                    prefix,
                    contrast_name,
                ),
            'z_score':
                '{}contrast-{}_stat-z_statmap.nii.gz'.format(
                    prefix,
                    contrast_name,
                ),
            'p_value':
                '{}contrast-{}_stat-p_statmap.nii.gz'.format(
                    prefix,
                    contrast_name,
                ),
        }
        # Rename keys
        renamed_contrast_maps = {
            contrast_level_mapping.get(k, k): v
            for k, v in contrast_maps.items()
        }

        for map_name, img in renamed_contrast_maps.items():
            out_file = os.path.join(out_dir, map_name)
            img.to_filename(out_file)

    # Model-level images
    model_level_mapping = {
        'residuals': '{}stat-errorts_statmap.nii.gz'.format(prefix),
        'r_square': '{}stat-rSquare_statmap.nii.gz'.format(prefix),
    }
    for attr, map_name in model_level_mapping.items():
        print('Extracting and saving {}'.format(attr))
        img = getattr(model, attr)
        if isinstance(img, list):
            img = img[0]

        out_file = os.path.join(out_dir, map_name)
        img.to_filename(out_file)


def _clean_contrast_name(contrast_name):
    """Remove prohibited characters from name and convert to camelCase.

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
    new_name = new_name.replace('-', ' Minus ')
    new_name = new_name.replace('+', ' Plus ')
    new_name = new_name.replace('>', ' Gt ')
    new_name = new_name.replace('<', ' Lt ')

    # Others translate to spaces
    new_name = new_name.replace('_', ' ')

    # Convert to camelCase
    new_name = new_name.split(' ')
    new_name[0] = new_name[0].lower()
    new_name[1:] = [c.title() for c in new_name[1:]]
    new_name = ' '.join(new_name)

    # Remove non-alphanumeric characters
    new_name = ''.join(ch for ch in new_name if ch.isalnum())

    # Let users know if the name was changed
    if new_name != contrast_name:
        warnings.warn(
            f'Contrast name "{contrast_name}" changed to "{new_name}"'
        )
    return new_name
