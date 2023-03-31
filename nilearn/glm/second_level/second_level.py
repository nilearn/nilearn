"""
This module provides facilities to realize a second level analysis on lists of
first level contrasts or directly on fitted first level models

Author: Martin Perez-Guevara, 2016
"""

import sys
import time
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Memory
from nibabel import Nifti1Image
from sklearn.base import clone

from nilearn._utils import fill_doc
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils import stringify_path
from nilearn.maskers import NiftiMasker
from nilearn.glm.contrasts import (compute_contrast,
                                   expression_to_contrast_vector)
from nilearn.glm.first_level import FirstLevelModel, run_glm
from nilearn.glm.first_level.design_matrix import \
    make_second_level_design_matrix
from nilearn.glm.regression import RegressionResults, SimpleRegressionResults
from nilearn.image import mean_img
from nilearn.mass_univariate import permuted_ols
from nilearn.glm._base import BaseGLM


def _check_second_level_input(second_level_input,
                              design_matrix,
                              confounds=None):
    """Checking second_level_input type"""
    input_type = _check_input_type(second_level_input)
    _check_input_as_type(
        second_level_input,
        input_type,
        confounds is None,
        design_matrix is None
    )


def _check_input_type(second_level_input):
    """Determines the type of input provided."""
    if isinstance(second_level_input, pd.DataFrame):
        return "df_object"
    if isinstance(second_level_input, (str, Nifti1Image)):
        return "nii_object"
    if isinstance(second_level_input, list):
        return _check_input_type_when_list(second_level_input)
    raise TypeError("second_level_input must be "
                    "either a pandas DataFrame, "
                    "a Niimg-like object, "
                    "a list of Niimg-like object or "
                    "a list of FirstLevelModel objects. "
                    f"Got {_return_type(second_level_input)} instead")


def _return_type(second_level_input):
    if isinstance(second_level_input, list):
        return [type(x) for x in second_level_input]
    else:
        return type(second_level_input)


def _check_input_type_when_list(second_level_input):
    """Determines the type of input provided when it is a list."""
    if len(second_level_input) < 2:
        raise TypeError('A second level model requires a list with at'
                            ' least two first level models or niimgs')
    _check_all_elements_of_same_type(second_level_input)   
    if all(isinstance(x, (str, Nifti1Image)) for x in second_level_input):
        return "nii_object"
    if all(isinstance(x, FirstLevelModel) for x in second_level_input):
        return "flm_object"
    raise TypeError("second_level_input must be "
                    "either a pandas DataFrame, "
                    "a Niimg-like object, "
                    "a list of Niimg-like object or "
                    "a list of FirstLevelModel objects. "
                    f"Got {_return_type(second_level_input)} instead")


def _check_all_elements_of_same_type(data):
    for idx, input in enumerate(data):
        if not isinstance(input, type(data[0])):  
             raise TypeError(
                 f'Elements of second_level_input must be of the same type.'
                 f' Got object type {type(input)} at idx {idx}.')


def _check_input_as_type(second_level_input,
                         input_type,
                         none_confounds,
                         none_design_matrix):
    if input_type == "flm_object":
        _check_input_as_first_level_model(second_level_input, none_confounds)
    elif input_type == "nii_object":
         _check_input_as_nifti_images(second_level_input, none_design_matrix)
    else:
         _check_input_as_dataframe(second_level_input)


def _check_input_as_first_level_model(second_level_input,
                                      none_confounds):
    for model_idx, first_level in enumerate(second_level_input):
        if (first_level.labels_ is None or first_level.results_ is None):
             raise ValueError(
                 'Model %s at index %i has not been fit yet'
                 '' % (first_level.subject_label, model_idx))
        if not none_confounds and first_level.subject_label is None:
                raise ValueError(
                    'In case confounds are provided, first level '
                    'objects need to provide the attribute '
                    'subject_label to match rows appropriately.'
                    f'Model at idx {model_idx} does not provide it. '
                    'To set it, you can do '
                    'first_level.subject_label = "01"')


def _check_input_as_dataframe(second_level_input):
    for col in ('subject_label', 'map_name', 'effects_map_path'):
        if col not in second_level_input.columns:
            raise ValueError('second_level_input DataFrame must have'
                             ' columns subject_label, map_name and'
                             ' effects_map_path')
    if not all(isinstance(_, str) 
               for _ in second_level_input['subject_label'].tolist()):
        raise ValueError('subject_label column must contain only strings')


def _check_input_as_nifti_images(second_level_input, none_design_matrix):
    if isinstance(second_level_input, (str, Nifti1Image)):
        second_level_input = [second_level_input]
    for niimg in second_level_input:
        check_niimg(niimg=niimg, atleast_4d=True)
    if none_design_matrix:
        raise ValueError('List of niimgs as second_level_input'
                         ' require a design matrix to be provided')


def _check_confounds(confounds):
    """Checking confounds type"""
    if confounds is not None:
        if not isinstance(confounds, pd.DataFrame):
            raise ValueError('confounds must be a pandas DataFrame')
        if 'subject_label' not in confounds.columns:
            raise ValueError('confounds DataFrame must contain column'
                             ' "subject_label"')
        if len(confounds.columns) < 2:
            raise ValueError('confounds should contain at least 2 columns'
                             ' one called "subject_label" and the other'
                             ' with a given confound')
        # Make sure subject_label contain strings
        if not all([isinstance(_, str) for _ in
                    confounds['subject_label'].tolist()]):
            raise ValueError('subject_label column must contain only strings')


def _check_first_level_contrast(second_level_input, first_level_contrast):
    if isinstance(second_level_input[0], FirstLevelModel):
        if first_level_contrast is None:
            raise ValueError('If second_level_input was a list of '
                             'FirstLevelModel, then first_level_contrast '
                             'is mandatory. It corresponds to the '
                             'second_level_contrast argument of the '
                             'compute_contrast method of FirstLevelModel')


def _check_output_type(output_type, valid_types):
    if output_type not in valid_types:
        raise ValueError('output_type must be one of {}'.format(valid_types))


def _check_design_matrix(design_matrix):
    """Checking design_matrix type"""
    if design_matrix is not None:
        if not isinstance(design_matrix, pd.DataFrame):
            raise ValueError('design matrix must be a pandas DataFrame')


def _check_effect_maps(effect_maps, design_matrix):
    if len(effect_maps) != design_matrix.shape[0]:
        raise ValueError(
            'design_matrix does not match the number of maps considered. '
            '%i rows in design matrix do not match with %i maps' %
            (design_matrix.shape[0], len(effect_maps)))


def _get_con_val(second_level_contrast, design_matrix):
    """ Check the contrast and return con_val when testing one contrast or more
    """
    if second_level_contrast is None:
        if design_matrix.shape[1] == 1:
            second_level_contrast = np.ones([1])
        else:
            raise ValueError('No second-level contrast is specified.')
    if not isinstance(second_level_contrast, str):
        con_val = second_level_contrast
        if np.all(con_val == 0):
            raise ValueError('Contrast is null')
    else:
        design_columns = design_matrix.columns.tolist()
        con_val = expression_to_contrast_vector(
            second_level_contrast, design_columns)
    return con_val


def _get_contrast(second_level_contrast, design_matrix):
    """ Check and return contrast when testing one contrast at the time """
    if isinstance(second_level_contrast, str):
        if second_level_contrast in design_matrix.columns.tolist():
            contrast = second_level_contrast
        else:
            raise ValueError('"{}" is not a valid contrast name'.format(
                second_level_contrast)
            )
    else:
        # Check contrast definition
        if second_level_contrast is None:
            if design_matrix.shape[1] == 1:
                second_level_contrast = np.ones([1])
            else:
                raise ValueError('No second-level contrast is specified.')
        elif (np.nonzero(second_level_contrast)[0]).size != 1:
            raise ValueError('second_level_contrast must be '
                             'a list of 0s and 1s')
        con_val = np.asarray(second_level_contrast, dtype=bool)
        contrast = np.asarray(design_matrix.columns.tolist())[con_val][0]
    return contrast


def _infer_effect_maps(second_level_input, contrast_def):
    """Deals with the different possibilities of second_level_input"""
    # Build the design matrix X and list of imgs Y for GLM fit
    if isinstance(second_level_input, pd.DataFrame):
        # If a Dataframe was given, we expect contrast_def to be in map_name
        def _is_contrast_def(x):
            return x['map_name'] == contrast_def

        is_con = second_level_input.apply(_is_contrast_def, axis=1)
        effect_maps = second_level_input[is_con]['effects_map_path'].tolist()

    elif isinstance(second_level_input[0], FirstLevelModel):
        # Get the first level model maps
        effect_maps = []
        for model in second_level_input:
            effect_map = model.compute_contrast(contrast_def,
                                                output_type='effect_size')
            effect_maps.append(effect_map)

    else:
        effect_maps = second_level_input

    # check niimgs
    for niimg in effect_maps:
        check_niimg(niimg, ensure_ndim=3)

    return effect_maps


def _process_second_level_input(second_level_input):
    """Helper function to process second_level_input."""
    if isinstance(second_level_input, pd.DataFrame):
        return _process_second_level_input_as_dataframe(second_level_input)
    elif(hasattr(second_level_input, "__iter__")
         and isinstance(second_level_input[0], FirstLevelModel)):
        return _process_second_level_input_as_firstlevelmodels(
            second_level_input
        )
    else:
        return mean_img(second_level_input), None


def _process_second_level_input_as_dataframe(second_level_input):
    """Helper function to process second_level_input provided
    as a pandas DataFrame.
    """
    sample_map = second_level_input['effects_map_path'][0]
    labels = second_level_input['subject_label']
    subjects_label = labels.values.tolist()
    return sample_map, subjects_label


def _sort_input_dataframe(second_level_input):
    """This function sorts the pandas dataframe by subject_label to
    avoid inconsistencies with the design matrix row order when
    automatically extracting maps.
    """
    columns = second_level_input.columns.tolist()
    column_index = columns.index('subject_label')
    sorted_matrix = sorted(
        second_level_input.values, key=lambda x: x[column_index]
    )
    return pd.DataFrame(sorted_matrix, columns=columns)


def _process_second_level_input_as_firstlevelmodels(second_level_input):
    """Helper function to process second_level_input provided
    as a list of FirstLevelModel objects.
    """
    sample_model = second_level_input[0]
    sample_condition = sample_model.design_matrices_[0].columns[0]
    sample_map = sample_model.compute_contrast(
        sample_condition, output_type='effect_size'
    )
    labels = [model.subject_label for model in second_level_input]
    return sample_map, labels


@fill_doc
class SecondLevelModel(BaseGLM):
    """Implementation of the :term:`General Linear Model<GLM>` for multiple
    subject :term:`fMRI` data.

    Parameters
    ----------
    mask_img : Niimg-like, :class:`~nilearn.maskers.NiftiMasker` or\
    :class:`~nilearn.maskers.MultiNiftiMasker`, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a
        :class:`~nilearn.maskers.MultiNiftiMasker` with default
        parameters. Automatic mask computation assumes first level imgs have
        already been masked.
    %(target_affine)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(target_shape)s

        .. note::
            This parameter is passed to :func:`nilearn.image.resample_img`.

    %(smoothing_fwhm)s
    %(memory)s
    %(memory_level1)s
    %(verbose0)s
        If 0 prints nothing. If 1 prints final computation time.
        If 2 prints masker computation details.
    %(n_jobs)s
    minimize_memory : :obj:`bool`, optional
        Gets rid of some variables on the model fit results that are not
        necessary for contrast computation and would only be useful for
        further inspection of model details. This has an important impact
        on memory consumption. Default=True.

    """
    def __init__(self, mask_img=None, target_affine=None, target_shape=None,
                 smoothing_fwhm=None,
                 memory=Memory(None), memory_level=1, verbose=0,
                 n_jobs=1, minimize_memory=True):
        self.mask_img = mask_img
        self.target_affine = target_affine
        self.target_shape = target_shape
        self.smoothing_fwhm = smoothing_fwhm
        memory = stringify_path(memory)
        if isinstance(memory, str):
            self.memory = Memory(memory)
        else:
            self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.minimize_memory = minimize_memory
        self.second_level_input_ = None
        self.confounds_ = None
        self.labels_ = None
        self.results_ = None

    @fill_doc
    def fit(self, second_level_input, confounds=None, design_matrix=None):
        """Fit the second-level :term:`GLM`.

        1. create design matrix
        2. do a masker job: fMRI_data -> Y
        3. fit regression to (Y, X)

        Parameters
        ----------
        %(second_level_input)s
        confounds : :class:`pandas.DataFrame`, optional
            Must contain a ``subject_label`` column. All other columns are
            considered as confounds and included in the model. If
            ``design_matrix`` is provided then this argument is ignored.
            The resulting second level design matrix uses the same column
            names as in the given :class:`~pandas.DataFrame` for confounds.
            At least two columns are expected, ``subject_label`` and at
            least one confound.

        design_matrix : :class:`pandas.DataFrame`, optional
            Design matrix to fit the :term:`GLM`. The number of rows
            in the design matrix must agree with the number of maps derived
            from ``second_level_input``.
            Ensure that the order of maps given by a ``second_level_input``
            list of Niimgs matches the order of the rows in the design matrix.

        """
        # check second_level_input
        _check_second_level_input(second_level_input, design_matrix,
                                  confounds=confounds)

        # check confounds
        _check_confounds(confounds)

        # check design matrix
        _check_design_matrix(design_matrix)

        if isinstance(second_level_input, pd.DataFrame):
            second_level_input = _sort_input_dataframe(second_level_input)
        self.second_level_input_ = second_level_input
        self.confounds_ = confounds
        sample_map, subjects_label = _process_second_level_input(
            second_level_input
        )

        # Report progress
        t0 = time.time()
        if self.verbose > 0:
            sys.stderr.write("Fitting second level model. "
                             "Take a deep breath\r")

        # Create and set design matrix, if not given
        if design_matrix is None:
            design_matrix = make_second_level_design_matrix(subjects_label,
                                                            confounds)
        self.design_matrix_ = design_matrix

        # Learn the mask. Assume the first level imgs have been masked.
        if not isinstance(self.mask_img, NiftiMasker):
            self.masker_ = NiftiMasker(
                mask_img=self.mask_img,
                target_affine=self.target_affine,
                target_shape=self.target_shape,
                smoothing_fwhm=self.smoothing_fwhm,
                memory=self.memory,
                verbose=max(0, self.verbose - 1),
                memory_level=self.memory_level)
        else:
            self.masker_ = clone(self.mask_img)
            for param_name in ['smoothing_fwhm', 'memory', 'memory_level']:
                our_param = getattr(self, param_name)
                if our_param is None:
                    continue
                if getattr(self.masker_, param_name) is not None:
                    warn('Parameter %s of the masker overridden' % param_name)
                setattr(self.masker_, param_name, our_param)
        self.masker_.fit(sample_map)

        # Report progress
        if self.verbose > 0:
            sys.stderr.write("\nComputation of second level model done in "
                             "%i seconds\n" % (time.time() - t0))

        return self

    @fill_doc
    def compute_contrast(self, second_level_contrast=None,
                         first_level_contrast=None,
                         second_level_stat_type=None, output_type='z_score'):
        """Generate different outputs corresponding to
        the contrasts provided e.g. z_map, t_map, effects and variance.

        Parameters
        ----------
        %(second_level_contrast)s
        first_level_contrast : :obj:`str` or :class:`numpy.ndarray` of\
        shape (n_col) with respect to\
        :class:`~nilearn.glm.first_level.FirstLevelModel`, optional

            - In case a :obj:`list` of
              :class:`~nilearn.glm.first_level.FirstLevelModel` was provided
              as ``second_level_input``, we have to provide a contrast to
              apply to the first level models to get the corresponding list
              of images desired, that would be tested at the second level.
            - In case a :class:`~pandas.DataFrame` was provided as
              ``second_level_input`` this is the map name to extract from the
              :class:`~pandas.DataFrame` ``map_name`` column. It has to be
              a 't' contrast.

        second_level_stat_type : {'t', 'F'} or None, optional
            Type of the second level contrast. Default=None.

        output_type : {'z_score', 'stat', 'p_value', \
                :term:`'effect_size'<Parameter Estimate>`, 'effect_variance', \
                'all'}, optional
            Type of the output map. Default='z-score'.

        Returns
        -------
        output_image : :class:`~nibabel.nifti1.Nifti1Image`
            The desired output image(s). If ``output_type == 'all'``, then
            the output is a dictionary of images, keyed by the type of image.

        """
        if self.second_level_input_ is None:
            raise ValueError('The model has not been fit yet')

        # check first_level_contrast
        _check_first_level_contrast(self.second_level_input_,
                                    first_level_contrast)

        # check contrast and obtain con_val
        con_val = _get_con_val(second_level_contrast, self.design_matrix_)

        # check output type
        # 'all' is assumed to be the final entry;
        # if adding more, place before 'all'
        valid_types = ['z_score', 'stat', 'p_value', 'effect_size',
                       'effect_variance', 'all']
        _check_output_type(output_type, valid_types)

        # Get effect_maps appropriate for chosen contrast
        effect_maps = _infer_effect_maps(self.second_level_input_,
                                         first_level_contrast)
        # Check design matrix X and effect maps Y agree on number of rows
        _check_effect_maps(effect_maps, self.design_matrix_)

        # Fit an Ordinary Least Squares regression for parametric statistics
        Y = self.masker_.transform(effect_maps)
        if self.memory:
            mem_glm = self.memory.cache(run_glm, ignore=['n_jobs'])
        else:
            mem_glm = run_glm
        labels, results = mem_glm(Y, self.design_matrix_.values,
                                  n_jobs=self.n_jobs, noise_model='ols')

        # We save memory if inspecting model details is not necessary
        if self.minimize_memory:
            for key in results:
                results[key] = SimpleRegressionResults(results[key])
        self.labels_ = labels
        self.results_ = results

        # We compute contrast object
        if self.memory:
            mem_contrast = self.memory.cache(compute_contrast)
        else:
            mem_contrast = compute_contrast
        contrast = mem_contrast(self.labels_, self.results_, con_val,
                                second_level_stat_type)

        output_types = \
            valid_types[:-1] if output_type == 'all' else [output_type]

        outputs = {}
        for output_type_ in output_types:
            # We get desired output from contrast object
            estimate_ = getattr(contrast, output_type_)()
            # Prepare the returned images
            output = self.masker_.inverse_transform(estimate_)
            contrast_name = str(con_val)
            output.header['descrip'] = (
                '%s of contrast %s' % (output_type, contrast_name))
            outputs[output_type_] = output

        return outputs if output_type == 'all' else output

    def _get_voxelwise_model_attribute(self, attribute,
                                       result_as_time_series):
        """Transform RegressionResults instances within a dictionary
        (whose keys represent the autoregressive coefficient under the 'ar1'
        noise model or only 0.0 under 'ols' noise_model and values are the
        RegressionResults instances) into input nifti space.

        Parameters
        ----------
        attribute : str
            an attribute of a RegressionResults instance.
            possible values include: 'residuals', 'normalized_residuals',
            'predicted', SSE, r_square, MSE.

        result_as_time_series : bool
            whether the RegressionResult attribute has a value
            per timepoint of the input nifti image.

        Returns
        -------
        output : list
            A list of Nifti1Image(s).

        """
        # check if valid attribute is being accessed.
        all_attributes = dict(vars(RegressionResults)).keys()
        possible_attributes = [prop
                               for prop in all_attributes
                               if '__' not in prop
                               ]
        if attribute not in possible_attributes:
            msg = ("attribute must be one of: "
                   "{attr}".format(attr=possible_attributes)
                   )
            raise ValueError(msg)

        if self.minimize_memory:
            raise ValueError(
                'To access voxelwise attributes like '
                'R-squared, residuals, and predictions, '
                'the `SecondLevelModel`-object needs to store '
                'there attributes. '
                'To do so, set `minimize_memory` to `False` '
                'when initializing the `SecondLevelModel`-object.')

        if self.labels_ is None or self.results_ is None:
            raise ValueError("The model has no results. This could be "
                             "because the model has not been fitted yet "
                             "or because no contrast has been computed "
                             "already.")

        if result_as_time_series:
            voxelwise_attribute = np.zeros(
                (self.design_matrix_.shape[0], len(self.labels_))
            )
        else:
            voxelwise_attribute = np.zeros((1, len(self.labels_)))

        for label_ in self.results_:
            label_mask = self.labels_ == label_
            voxelwise_attribute[:, label_mask] = getattr(
                self.results_[label_], attribute
            )
        return self.masker_.inverse_transform(voxelwise_attribute)


@fill_doc
def non_parametric_inference(
    second_level_input,
    confounds=None,
    design_matrix=None,
    second_level_contrast=None,
    first_level_contrast=None,
    mask=None,
    smoothing_fwhm=None,
    model_intercept=True,
    n_perm=10000,
    two_sided_test=False,
    random_state=None,
    n_jobs=1,
    verbose=0,
    threshold=None,
    tfce=False
):
    """Generate p-values corresponding to the contrasts provided \
    based on permutation testing.

    This function is a light wrapper around
    :func:`~nilearn.mass_univariate.permuted_ols`, with additional steps to
    ensure compatibility with the :mod:`~nilearn.glm.second_level` module.

    Parameters
    ----------
    %(second_level_input)s

    confounds : :obj:`pandas.DataFrame` or None, optional
        Must contain a subject_label column. All other columns are
        considered as confounds and included in the model. If
        ``design_matrix`` is provided then this argument is ignored.
        The resulting second level design matrix uses the same column
        names as in the given :obj:`~pandas.DataFrame` for confounds.
        At least two columns are expected, ``subject_label`` and at
        least one confound.

    design_matrix : :obj:`pandas.DataFrame` or None, optional
        Design matrix to fit the :term:`GLM`. The number of rows
        in the design matrix must agree with the number of maps derived
        from ``second_level_input``.
        Ensure that the order of maps given by a ``second_level_input``
        list of Niimgs matches the order of the rows in the design matrix.

    %(second_level_contrast)s

    first_level_contrast : :obj:`str`, optional
        In case a pandas DataFrame was provided as second_level_input this
        is the map name to extract from the pandas dataframe map_name column.
        It has to be a 't' contrast.

        .. versionadded:: 0.9.0

    mask : Niimg-like, :obj:`~nilearn.maskers.NiftiMasker` or \
            :obj:`~nilearn.maskers.MultiNiftiMasker` object, optional
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given, it will be computed
        automatically by a :class:`~nilearn.maskers.MultiNiftiMasker` with
        default parameters. Automatic mask computation assumes first level
        imgs have already been masked.

    %(smoothing_fwhm)s

    model_intercept : :obj:`bool`, optional
        If ``True``, a constant column is added to the confounding variates
        unless the tested variate is already the intercept.
        Default=True.

    n_perm : :obj:`int`, optional
        Number of permutations to perform.
        Permutations are costly but the more are performed, the more precision
        one gets in the p-values estimation. Default=10000.

    two_sided_test : :obj:`bool`, optional

        - If ``True``, performs an unsigned t-test.
          Both positive and negative effects are considered; the null
          hypothesis is that the effect is zero.
        - If ``False``, only positive effects are considered as relevant.
          The null hypothesis is that the effect is zero or negative.

        Default=False.

    %(random_state)s
        Use this parameter to have the same permutations in each
        computing units.

    %(n_jobs)s

    %(verbose0)s

    threshold : None or :obj:`float`, optional
        Cluster-forming threshold in p-scale.
        This is only used for cluster-level inference.
        If None, no cluster-level inference will be performed.
        Default=None.

        .. warning::

            Performing cluster-level inference will increase the computation
            time of the permutation procedure.

        .. versionadded:: 0.9.2

    tfce : :obj:`bool`, optional
        Whether to calculate :term:`TFCE` as part of the permutation procedure
        or not.
        The TFCE calculation is implemented as described in
        :footcite:t:`Smith2009a`.
        Default=False.

        .. warning::

            Performing TFCE-based inference will increase the computation
            time of the permutation procedure considerably.
            The permutations may take multiple hours, depending on how many
            permutations are requested and how many jobs are performed in
            parallel.

        .. versionadded:: 0.9.2

    Returns
    -------
    neg_log10_vfwe_pvals_img : :class:`~nibabel.nifti1.Nifti1Image`
        The image which contains negative logarithm of the
        voxel-level FWER-corrected p-values.

        .. note::
            This is returned if ``threshold`` is None (the default).

    outputs : :obj:`dict`
        Output images, organized in a dictionary.
        Each image is 3D/4D, with the potential fourth dimension corresponding
        to the regressors.

        .. note::
            This is returned if ``tfce`` is False or ``threshold`` is not None.

        .. versionadded:: 0.9.2

        Here are the keys:

        =============== =======================================================
        key             description
        =============== =======================================================
        t               T-statistics associated with the significance test of
                        the n_regressors explanatory variates against the
                        n_descriptors target variates.
        logp_max_t      Negative log10 family-wise error rate-corrected
                        p-values corrected based on the distribution of maximum
                        t-statistics from permutations.
        size            Cluster size values associated with the significance
                        test of the n_regressors explanatory variates against
                        the n_descriptors target variates.

                        Returned only if ``threshold`` is not None.
        logp_max_size   Negative log10 family-wise error rate-corrected
                        p-values corrected based on the distribution of maximum
                        cluster sizes from permutations.
                        This map is generated through cluster-level methods, so
                        the values in the map describe the significance of
                        clusters, rather than individual voxels.

                        Returned only if ``threshold`` is not None.
        mass            Cluster mass values associated with the significance
                        test of the n_regressors explanatory variates against
                        the n_descriptors target variates.

                        Returned only if ``threshold`` is not None.
        logp_max_mass   Negative log10 family-wise error rate-corrected
                        p-values corrected based on the distribution of maximum
                        cluster masses from permutations.
                        This map is generated through cluster-level methods, so
                        the values in the map describe the significance of
                        clusters, rather than individual voxels.

                        Returned only if ``threshold`` is not None.
        tfce            TFCE values associated with the significance test of
                        the n_regressors explanatory variates against the
                        n_descriptors target variates.

                        Returned only if ``tfce`` is True.
        logp_max_tfce   Negative log10 family-wise error rate-corrected
                        p-values corrected based on the distribution of maximum
                        TFCE values from permutations.

                        Returned only if ``tfce`` is True.
        =============== =======================================================

    See also
    --------
    :func:`~nilearn.mass_univariate.permuted_ols` : For more information on \
        the permutation procedure.

    References
    ----------
    .. footbibliography::
    """
    _check_second_level_input(second_level_input, design_matrix)
    _check_confounds(confounds)
    _check_design_matrix(design_matrix)

    if isinstance(second_level_input, pd.DataFrame):
        second_level_input = _sort_input_dataframe(second_level_input)
    sample_map, _ = _process_second_level_input(second_level_input)

    # Report progress
    t0 = time.time()
    if verbose > 0:
        sys.stderr.write("Fitting second level model...")

    # Learn the mask. Assume the first level imgs have been masked.
    if not isinstance(mask, NiftiMasker):
        masker = NiftiMasker(
            mask_img=mask, smoothing_fwhm=smoothing_fwhm,
            memory=Memory(None), verbose=max(0, verbose - 1),
            memory_level=1)

    else:
        masker = clone(mask)
        if smoothing_fwhm is not None:
            if getattr(masker, 'smoothing_fwhm') is not None:
                warn('Parameter smoothing_fwhm of the masker overridden')
                setattr(masker, 'smoothing_fwhm', smoothing_fwhm)

    masker.fit(sample_map)

    # Report progress
    if verbose > 0:
        sys.stderr.write("\nComputation of second level model done in "
                         "%i seconds\n" % (time.time() - t0))

    # Check and obtain the contrast
    contrast = _get_contrast(second_level_contrast, design_matrix)
    # Get first-level effect_maps
    effect_maps = _infer_effect_maps(second_level_input, first_level_contrast)

    # Check design matrix and effect maps agree on number of rows
    _check_effect_maps(effect_maps, design_matrix)

    # Obtain design matrix vars
    var_names = design_matrix.columns.tolist()

    # Obtain tested_var
    tested_var = np.asarray(design_matrix[contrast])
    # Remove tested var from remaining var names
    var_names.remove(contrast)

    # Obtain confounding vars
    if len(var_names) == 0:
        # No other vars in design matrix
        confounding_vars = None
    else:
        # Use remaining vars as confounding vars
        confounding_vars = np.asarray(design_matrix[var_names])

    # Mask data
    target_vars = masker.transform(effect_maps)

    # Perform massively univariate analysis with permuted OLS
    outputs = permuted_ols(
        tested_var,
        target_vars,
        confounding_vars=confounding_vars,
        model_intercept=model_intercept,
        n_perm=n_perm,
        two_sided_test=two_sided_test,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=max(0, verbose - 1),
        masker=masker,
        threshold=threshold,
        tfce=tfce,
        output_type='dict',
    )
    neg_log10_vfwe_pvals_img = masker.inverse_transform(np.ravel(
        outputs['logp_max_t']
    ))

    if (not tfce) and (threshold is None):
        return neg_log10_vfwe_pvals_img

    t_img = masker.inverse_transform(np.ravel(outputs['t']))

    out = {
        't': t_img,
        'logp_max_t': neg_log10_vfwe_pvals_img,
    }

    if tfce:
        neg_log10_tfce_pvals_img = masker.inverse_transform(
            np.ravel(outputs['logp_max_tfce']),
        )
        out['tfce'] = masker.inverse_transform(np.ravel(outputs['tfce']))
        out['logp_max_tfce'] = neg_log10_tfce_pvals_img

    if threshold is not None:
        # Cluster size-based p-values
        neg_log10_csfwe_pvals_img = masker.inverse_transform(
            np.ravel(outputs['logp_max_size']),
        )

        # Cluster mass-based p-values
        neg_log10_cmfwe_pvals_img = masker.inverse_transform(
            np.ravel(outputs['logp_max_mass']),
        )

        out['size'] = masker.inverse_transform(np.ravel(outputs['size']))
        out['logp_max_size'] = neg_log10_csfwe_pvals_img
        out['mass'] = masker.inverse_transform(np.ravel(outputs['mass']))
        out['logp_max_mass'] = neg_log10_cmfwe_pvals_img

    return out
