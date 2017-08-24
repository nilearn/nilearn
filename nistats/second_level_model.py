"""
This module presents an interface to use the glm implemented in
nistats.regression.

It provides facilities to realize a second level analysis on lists of
first level contrasts or directly on fitted first level models

Author: Martin Perez-Guevara, 2016
"""

from warnings import warn
import sys
import time
import pandas as pd
import numpy as np
from nibabel import Nifti1Image

from sklearn.base import BaseEstimator, TransformerMixin, clone
from nilearn._utils.niimg_conversions import check_niimg
from nilearn._utils import CacheMixin
from nilearn.input_data import NiftiMasker
from patsy import DesignInfo

from .first_level_model import FirstLevelModel
from .first_level_model import run_glm
from .regression import SimpleRegressionResults
from .contrasts import compute_contrast
from .utils import _basestring
from .design_matrix import create_second_level_design


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


class SecondLevelModel(BaseEstimator, TransformerMixin, CacheMixin):
    """ Implementation of the General Linear Model for multiple subject
    fMRI data

    Parameters
    ----------

    mask: Niimg-like, NiftiMasker or MultiNiftiMasker object, optional,
        Mask to be used on data. If an instance of masker is passed,
        then its mask will be used. If no mask is given,
        it will be computed automatically by a MultiNiftiMasker with default
        parameters. Automatic mask computation assumes first level imgs have
        already been masked.

    smoothing_fwhm: float, optional
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.

    memory: string, optional
        Path to the directory used to cache the masking process and the glm
        fit. By default, no caching is done. Creates instance of joblib.Memory.

    memory_level: integer, optional
        Rough estimator of the amount of memory used by caching. Higher value
        means more memory for caching.

    verbose : integer, optional
        Indicate the level of verbosity. By default, nothing is printed.
        If 0 prints nothing. If 1 prints final computation time.
        If 2 prints masker computation details.

    n_jobs : integer, optional
        The number of CPUs to use to do the computation. -1 means
        'all CPUs', -2 'all CPUs but one', and so on.

    minimize_memory : boolean, optional
        Gets rid of some variables on the model fit results that are not
        necessary for contrast computation and would only be useful for
        further inspection of model details. This has an important impact
        on memory consumption. True by default.

    """
    def __init__(self, mask=None, smoothing_fwhm=None,
                 memory=None, memory_level=1, verbose=0,
                 n_jobs=1, minimize_memory=True):
        self.mask = mask
        self.smoothing_fwhm = smoothing_fwhm
        self.memory = memory
        self.memory_level = memory_level
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.minimize_memory = minimize_memory
        self.second_level_input_ = None
        self.confounds_ = None

    def fit(self, second_level_input, confounds=None, design_matrix=None):
        """ Fit the second-level GLM

        1. create design matrix
        2. do a masker job: fMRI_data -> Y
        3. fit regression to (Y, X)

        Parameters
        ----------
        second_level_input: list of `FirstLevelModel` objects or pandas
                            DataFrame or list of Niimg-like objects.
            Giving FirstLevelModel objects will allow to easily compute
            the second level contast of arbitrary first level contrasts thanks
            to the first_level_contrast argument of the compute_contrast
            method. Effect size images will be computed for each model to
            contrast at the second level.

            If a pandas DataFrame, then they have to contain subject_label,
            map_name and effects_map_path. It can contain multiple maps that
            would be selected during contrast estimation with the argument
            first_level_contrast of the compute_contrast function. The
            DataFrame will be sorted based on the subject_label column to avoid
            order inconsistencies when extracting the maps. So the rows of the
            automatically computed design matrix, if not provided, will
            correspond to the sorted subject_label column.

            If list of Niimg-like objects then this is taken literally as Y
            for the model fit and design_matrix must be provided.

        confounds: pandas DataFrame, optional
            Must contain a subject_label column. All other columns are
            considered as confounds and included in the model. If
            design_matrix is provided then this argument is ignored.
            The resulting second level design matrix uses the same column
            names as in the given DataFrame for confounds. At least two columns
            are expected, "subject_label" and at least one confound.

        design_matrix: pandas DataFrame, optional
            Design matrix to fit the GLM. The number of rows
            in the design matrix must agree with the number of maps derived
            from second_level_input.
            Ensure that the order of maps given by a second_level_input
            list of Niimgs matches the order of the rows in the design matrix.
            Must contain a column of 1s with column name 'intercept'.
        """
        # Check parameters
        # check first level input
        if isinstance(second_level_input, list):
            if len(second_level_input) < 2:
                raise ValueError('A second level model requires a list with at'
                                 'least two first level models or niimgs')
            # Check FirstLevelModel objects case
            if isinstance(second_level_input[0], FirstLevelModel):
                models_input = enumerate(second_level_input)
                for model_idx, first_level_model in models_input:
                    if (first_level_model.labels_ is None or
                            first_level_model.results_ is None):
                        raise ValueError(
                            'Model %s at index %i has not been fit yet'
                            '' % (first_level_model.subject_label, model_idx))
                    if not isinstance(first_level_model, FirstLevelModel):
                        raise ValueError(' object at idx %d is %s instead of'
                                         ' FirstLevelModel object' %
                                         (model_idx, type(first_level_model)))
                    if confounds is not None:
                        if first_level_model.subject_label is None:
                            raise ValueError(
                                'In case confounds are provided, first level '
                                'objects need to provide the attribute '
                                'subject_label to match rows appropriately.'
                                'Model at idx %d does not provide it. '
                                'To set it, you can do '
                                'first_level_model.subject_label = "01"'
                                '' % (model_idx))
            # Check niimgs case
            elif isinstance(second_level_input[0], (str, Nifti1Image)):
                if design_matrix is None:
                    raise ValueError('List of niimgs as second_level_input'
                                     ' require a design matrix to be provided')
                for model_idx, niimg in enumerate(second_level_input):
                    if not isinstance(niimg, (str, Nifti1Image)):
                        raise ValueError(' object at idx %d is %s instead of'
                                         ' Niimg-like object' %
                                         (model_idx, type(niimg)))
        # Check pandas dataframe case
        elif isinstance(second_level_input, pd.DataFrame):
            for col in ['subject_label', 'map_name', 'effects_map_path']:
                if col not in second_level_input.columns:
                    raise ValueError('second_level_input DataFrame must have'
                                     ' columns subject_label, map_name and'
                                     ' effects_map_path')
            # Make sure subject_label contain strings
            second_level_columns = second_level_input.columns.tolist()
            labels_index = second_level_columns.index('subject_label')
            labels_dtype = second_level_input.dtypes[labels_index]
            if not isinstance(labels_dtype, np.object):
                raise ValueError('subject_label column must be of dtype '
                                 'object instead of dtype %s' % labels_dtype)
        else:
            raise ValueError('second_level_input must be a list of'
                             ' `FirstLevelModel` objects, a pandas DataFrame'
                             ' or a list Niimg-like objects. Instead %s '
                             'was provided' % type(second_level_input))

        # check confounds
        if confounds is not None:
            if not isinstance(confounds, pd.DataFrame):
                raise ValueError('confounds must be a pandas DataFrame')
            if 'subject_label' not in confounds.columns:
                raise ValueError('confounds DataFrame must contain column'
                                 '"subject_label"')
            if len(confounds.columns) < 2:
                raise ValueError('confounds should contain at least 2 columns'
                                 'one called "subject_label" and the other'
                                 'with a given confound')
            # Make sure subject_label contain strings
            labels_index = confounds.columns.tolist().index('subject_label')
            labels_dtype = confounds.dtypes[labels_index]
            if not isinstance(labels_dtype, np.object):
                raise ValueError('subject_label column must be of dtype '
                                 'object instead of dtype %s' % labels_dtype)

        # check design matrix
        if design_matrix is not None:
            if not isinstance(design_matrix, pd.DataFrame):
                raise ValueError('design matrix must be a pandas DataFrame')
            if 'intercept' not in design_matrix.columns:
                raise ValueError('design matrix must contain "intercept"')

        # sort a pandas dataframe by subject_label to avoid inconsistencies
        # with the design matrix row order when automatically extracting maps
        if isinstance(second_level_input, pd.DataFrame):
            # Avoid pandas df.sort_value to keep compatibility with numpy 1.8
            # also pandas df.sort since it is completely deprecated.
            columns = second_level_input.columns.tolist()
            column_index = columns.index('subject_label')
            sorted_matrix = sorted(
                second_level_input.as_matrix(), key=lambda x: x[column_index])
            sorted_input = pd.DataFrame(sorted_matrix, columns=columns)
            second_level_input = sorted_input

        self.second_level_input_ = second_level_input
        self.confounds_ = confounds

        # Report progress
        t0 = time.time()
        if self.verbose > 0:
            sys.stderr.write("Fitting second level model. "
                             "Take a deep breath\r")

        # Select sample map for masker fit and get subjects_label for design
        if isinstance(second_level_input, pd.DataFrame):
            sample_map = second_level_input['effects_map_path'][0]
            labels = second_level_input['subject_label']
            subjects_label = labels.values.tolist()
        elif isinstance(second_level_input[0], FirstLevelModel):
            sample_model = second_level_input[0]
            sample_condition = sample_model.design_matrices_[0].columns[0]
            sample_map = sample_model.compute_contrast(
                sample_condition, output_type='effect_size')
            labels = [model.subject_label for model in second_level_input]
            subjects_label = labels
        else:
            # In this case design matrix had to be provided
            sample_map = second_level_input[0]

        # Create and set design matrix, if not given
        if design_matrix is None:
            design_matrix = create_second_level_design(subjects_label,
                                                       confounds)
        self.design_matrix_ = design_matrix

        # Learn the mask. Assume the first level imgs have been masked.
        if not isinstance(self.mask, NiftiMasker):
            self.masker_ = NiftiMasker(
                mask_img=self.mask, smoothing_fwhm=self.smoothing_fwhm,
                memory=self.memory, verbose=max(0, self.verbose - 1),
                memory_level=self.memory_level)
        else:
            self.masker_ = clone(self.mask)
            for param_name in ['smoothing_fwhm', 'memory', 'memory_level']:
                our_param = getattr(self, param_name)
                if our_param is None:
                    continue
                if getattr(self.masker_, param_name) is not None:
                    warn('Parameter %s of the masker overriden' % param_name)
                setattr(self.masker_, param_name, our_param)
        self.masker_.fit(sample_map)

        # Report progress
        if self.verbose > 0:
            sys.stderr.write("\nComputation of second level model done in "
                             "%i seconds\n" % (time.time() - t0))

        return self

    def compute_contrast(
            self, second_level_contrast='intercept', first_level_contrast=None,
            second_level_stat_type=None, output_type='z_score'):
        """Generate different outputs corresponding to
        the contrasts provided e.g. z_map, t_map, effects and variance.

        Parameters
        ----------
        second_level_contrast: str or array of shape (n_col), optional
            Where ``n_col`` is the number of columns of the design matrix,
            The string can be a formula compatible with the linear constraint
            of the Patsy library. Basically one can use the name of the
            conditions as they appear in the design matrix of
            the fitted model combined with operators /*+- and numbers.
            Please check the patsy documentation for formula examples:
            http://patsy.readthedocs.io/en/latest/API-reference.html#patsy.DesignInfo.linear_constraint

            VERY IMPORTANT: The 'intercept' corresponds to the second level
            effect after taking confounders in consideration. If there are
            no confounders then this will be equivalent to a simple t test.
            By default we compute the 'intercept' second level contrast.

        first_level_contrast: str or array of shape (n_col) with respect to
                              FirstLevelModel, optional
            In case a list of FirstLevelModel was provided as
            second_level_input, we have to provide a contrast to apply to
            the first level models to get the corresponding list of images
            desired, that would be tested at the second level. In case a
            pandas DataFrame was provided as second_level_input this is the
            map name to extract from the pandas dataframe map_name column.
            It has to be a 't' contrast.

        second_level_stat_type: {'t', 'F'}, optional
            Type of the second level contrast

        output_type: str, optional
            Type of the output map. Can be 'z_score', 'stat', 'p_value',
            'effect_size' or 'effect_variance'

        Returns
        -------
        output_image: Nifti1Image
            The desired output image

        """
        if self.second_level_input_ is None:
            raise ValueError('The model has not been fit yet')

        # first_level_contrast check
        if isinstance(self.second_level_input_[0], FirstLevelModel):
            if first_level_contrast is None:
                raise ValueError('If second_level_input was a list of '
                                 'FirstLevelModel, then first_level_contrast '
                                 'is mandatory. It corresponds to the '
                                 'second_level_contrast argument of the '
                                 'compute_contrast method of FirstLevelModel')

        # check contrast definition
        if isinstance(second_level_contrast, np.ndarray):
            con_val = second_level_contrast
            if np.all(con_val == 0):
                raise ValueError('Contrast is null')
        else:
            design_info = DesignInfo(self.design_matrix_.columns.tolist())
            constraint = design_info.linear_constraint(second_level_contrast)
            con_val = constraint.coefs
        # check output type
        if isinstance(output_type, _basestring):
            if output_type not in ['z_score', 'stat', 'p_value', 'effect_size',
                                   'effect_variance']:
                raise ValueError(
                    'output_type must be one of "z_score", "stat"'
                    ', "p_value", "effect_size" or "effect_variance"')
        else:
            raise ValueError('output_type must be one of "z_score", "stat",'
                             ' "p_value", "effect_size" or "effect_variance"')

        # Get effect_maps appropriate for chosen contrast
        effect_maps = _infer_effect_maps(self.second_level_input_,
                                         first_level_contrast)
        # check design matrix X and effect maps Y agree on number of rows
        if len(effect_maps) != self.design_matrix_.shape[0]:
            raise ValueError(
                'design_matrix does not match the number of maps considered. '
                '%i rows in design matrix do not match with %i maps' %
                (self.design_matrix_.shape[0], len(effect_maps)))

        # Fit an OLS regression for parametric statistics
        Y = self.masker_.transform(effect_maps)
        if self.memory is not None:
            arg_ignore = ['n_jobs']
            mem_glm = self.memory.cache(run_glm, ignore=arg_ignore)
        else:
            mem_glm = run_glm
        labels, results = mem_glm(Y, self.design_matrix_.as_matrix(),
                                  n_jobs=self.n_jobs, noise_model='ols')
        # We save memory if inspecting model details is not necessary
        if self.minimize_memory:
            for key in results:
                results[key] = SimpleRegressionResults(results[key])
        self.labels_ = labels
        self.results_ = results

        # We compute contrast object
        if self.memory is not None:
            mem_contrast = self.memory.cache(compute_contrast)
        else:
            mem_contrast = compute_contrast
        contrast = mem_contrast(self.labels_, self.results_, con_val,
                                second_level_stat_type)

        # We get desired output from contrast object
        estimate_ = getattr(contrast, output_type)()

        # Prepare the returned images
        output = self.masker_.inverse_transform(estimate_)
        contrast_name = str(con_val)
        output.get_header()['descrip'] = (
            '%s of contrast %s' % (output_type, contrast_name))
        return output
