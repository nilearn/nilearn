"""Functions related to the documentation.

docdict contains the standard documentation entries
used across Nilearn.

Entries are listed in alphabetical order.

source: Eric Larson and MNE-python team.
https://github.com/mne-tools/mne-python/blob/main/mne/utils/docs.py
"""

# sourcery skip: merge-dict-assign

import sys

##############################################################################
#
# Parameters definitions
#
# Standard documentation entries
#
# Entries are listed in alphabetical order.
#
docdict = {}

##############################################################################
#
# Parameters definitions
#

# alphas
docdict["alphas"] = """
alphas : :obj:`float` or :obj:`list` of :obj:`float` or None, default=None
    Choices for the constant that scales the overall regularization term.
    This parameter is mutually exclusive with the `n_alphas` parameter.
    If None or list of floats is provided, then the best value will be
    selected by cross-validation.
"""

# annotate
docdict["annotate"] = """
annotate : :obj:`bool`, default=True
    If `annotate` is `True`, positions and left/right annotation
    are added to the plot.
"""

# avg_method
docdict["avg_method"] = """
avg_method : {"mean", "median", "min", "max", custom function, None}, \
             default=None
    How to average vertex values to derive the face value:

    - ``"mean"``: results in smooth boundaries

    - ``"median"``: results in sharp boundaries

    - ``"min"`` or ``"max"``: for sparse matrices

    - `custom function`: You can also pass a custom function
      which will be executed though :func:`numpy.apply_along_axis`.
      Here is an example of a custom function:

        .. code-block:: python

            def custom_function(vertices):
                return vertices[0] * vertices[1] * vertices[2]

"""

# ax
docdict["ax"] = """
ax : :class:`~matplotlib.axes.Axes`
    The matplotlib axes in which the plots will be drawn.
"""

# axes
docdict["axes"] = """
axes : :class:`matplotlib.axes.Axes`, or 4 :obj:`tuple` \
of :obj:`float`: (xmin, ymin, width, height), default=None
    The axes, or the coordinates, in matplotlib figure space,
    of the axes used to display the plot.
    If `None`, the complete figure is used.
"""

# bg_map
docdict["bg_map"] = """
bg_map : :obj:`str` or :obj:`pathlib.Path` or \
         :class:`numpy.ndarray` \
         or :obj:`~nilearn.surface.SurfaceImage` or None,\
         default=None
    Background image to be plotted on the :term:`mesh`
    underneath the surf_data in grayscale,
    most likely a sulcal depth map for realistic shading.
    If the map contains values outside [0, 1],
    it will be rescaled such that all values are in [0, 1].
    Otherwise, it will not be modified.
    If a :obj:`str` or :obj:`pathlib.Path` is passed,
    it should be loadable to a :class:`numpy.ndarray`
    by :func:`~nilearn.surface.load_surf_data`.
    If a :class:`numpy.ndarray` is passed,
    if should have a shape `(n_vertices, )`,
    with ``n_vertices`` matching that of the underlying mesh
    used for plotting.
"""

# bg_img
docdict["bg_img"] = """
bg_img : Niimg-like object, optional
    See :ref:`extracting_data`.
    The background image to plot on top of.
"""

# bg_on_data
docdict["bg_on_data"] = r"""
bg_on_data : :obj:`bool`, default=False
    If `True` and a `bg_map` is specified,
    the `surf_data` data is multiplied by the background image,
    so that e.g. sulcal depth is jointly visible with `surf_data`.
    Otherwise, the background image will only be visible
    where there is no surface data
    (either because `surf_data` contains `nan`\s
    or because is was thresholded).

    .. note::

        This non-uniformly changes the surf_data values according
        to e.g the sulcal depth.

"""

# black_bg
docdict["black_bg"] = """
black_bg : :obj:`bool`, or "auto", optional
    If `True`, the background of the image is set to be black.
    If you wish to save figures with a black background,
    you will need to pass `facecolor="k", edgecolor="k"`
    to :func:`matplotlib.pyplot.savefig`.
"""

# border_size
docdict["border_size"] = """
border_size : :obj:`int`, optional
    The size, in :term:`voxel` of the border used on the side of
    the image to determine the value of the background.
"""

# cbar_tick_format
docdict["cbar_tick_format"] = """
cbar_tick_format : :obj:`str`, optional
    Controls how to format the tick labels of the colorbar.
    Ex: use "%%.2g" to display using scientific notation.
"""

# classifier_options
svc = "Linear support vector classifier"
logistic = "Logistic regression"
rc = "Ridge classifier"
dc = "Dummy classifier with stratified strategy"

docdict["classifier_options"] = f"""

    - ``"svc"``: :class:`{svc} <sklearn.svm.LinearSVC>` with L2 penalty.

    .. code-block:: python

        svc = LinearSVC(penalty="l2", max_iter=1e4)

    - ``"svc_l2"``: :class:`{svc} <sklearn.svm.LinearSVC>` with L2 penalty.

    .. note::

        Same as option `svc`.

    - ``"svc_l1"``: :class:`{svc} <sklearn.svm.LinearSVC>` with L1 penalty.

    .. code-block:: python

        svc_l1 = LinearSVC(penalty="l1", dual=False, max_iter=1e4)

    - ``"logistic"``: \
        :class:`{logistic} <sklearn.linear_model.LogisticRegressionCV>` \
        with L2 penalty.

    .. code-block:: python

        logistic = LogisticRegressionCV(penalty="l2", solver="liblinear")

    - ``"logistic_l1"``: \
        :class:`{logistic} <sklearn.linear_model.LogisticRegressionCV>` \
        with L1 penalty.

    .. code-block:: python

        logistic_l1 = LogisticRegressionCV(penalty="l1", solver="liblinear")

    - ``"logistic_l2"``: \
        :class:`{logistic} <sklearn.linear_model.LogisticRegressionCV>` \
        with L2 penalty

    .. note::

        Same as option `logistic`.

    - ``"ridge_classifier"``: \
        :class:`{rc} <sklearn.linear_model.RidgeClassifierCV>`.

    .. code-block:: python

        ridge_classifier = RidgeClassifierCV()

    - ``"dummy_classifier"``: :class:`{dc} <sklearn.dummy.DummyClassifier>`.

    .. code-block:: python

        dummy = DummyClassifier(strategy="stratified", random_state=0)

"""

# clean_args
docdict["clean_args"] = """
clean_args : :obj:`dict` or None, default=None
    Keyword arguments to be passed
    to :func:`~nilearn.signal.clean`
    called within the masker.
    Within :func:`~nilearn.signal.clean`,
    kwargs prefixed with ``'butterworth__'``
    will be passed to the Butterworth filter.
"""
docdict["clean_args_"] = docdict["clean_args"].replace(
    "clean_args : :obj:`dict` or None, default=None",
    "clean_args_ : :obj:`dict`",
)

# cmap
docdict["cmap"] = """
cmap : :class:`matplotlib.colors.Colormap`, or :obj:`str`, optional
    The colormap to use.
    Either a string which is a name of a matplotlib colormap,
    or a matplotlib colormap object.
"""

# cmap or lut
docdict["cmap_lut"] = """
cmap : :class:`matplotlib.colors.Colormap`, or :obj:`str`, \
       or :class:`pandas.DataFrame`, optional
    The colormap to use.
    Either a string which is a name of a matplotlib colormap,
    or a matplotlib colormap object,
    or a BIDS compliant
    `look-up table <https://bids-specification.readthedocs.io/en/latest/derivatives/imaging.html#common-image-derived-labels>`_
    passed as a pandas dataframe.
    If the look up table does not contain a ``color`` column,
    then the default colormap of this function will be used.
"""

# colorbar
docdict["colorbar"] = """
colorbar : :obj:`bool`, optional
    If `True`, display a colorbar on the right of the plots.
"""

# connected
docdict["connected"] = """
connected : :obj:`bool`, optional
    If connected is `True`, only the largest connect component is kept.
"""

# confounds
docdict["confounds"] = """
confounds : :class:`numpy.ndarray`, :obj:`str`, :class:`pathlib.Path`, \
            :class:`pandas.DataFrame` \
            or :obj:`list` of confounds timeseries, default=None
    This parameter is passed to :func:`nilearn.signal.clean`.
    Please see the related documentation for details.
    shape: (number of scans, number of confounds)
"""
docdict["confounds_multi"] = """
confounds : :obj:`list` of confounds, default=None
    List of confounds (arrays, dataframes,
    str or path of files loadable into an array).
    As confounds are passed to :func:`nilearn.signal.clean`,
    please see the related documentation for details about accepted types.
    Must be of same length than imgs.
"""

# cut_coords
docdict["cut_coords"] = """
cut_coords : None, a :obj:`tuple` of :obj:`float`, or :obj:`int`, optional
    The MNI coordinates of the point where the cut is performed.

    - If `display_mode` is `'ortho'` or `'tiled'`, this should
      be a 3-tuple: `(x, y, z)`

    - For `display_mode == "x"`, "y", or "z", then these are
      the coordinates of each cut in the corresponding direction.

    - If `None` is given, the cuts are calculated automatically.

    - If `display_mode` is 'mosaic', and the number of cuts is the same
      for all directions, `cut_coords` can be specified as an integer.
      It can also be a length 3 :obj:`tuple`
      specifying the number of cuts for
      every direction if these are different.

    .. note::

        If `display_mode` is "x", "y" or "z",
        `cut_coords` can be an integer,
        in which case it specifies the number of cuts to perform.

"""

# darkness
docdict["darkness"] = """
darkness : :obj:`float` between 0 and 1, optional
    Specifying the darkness of the background image:

    - `1` indicates that the original values of the background are used

    - `0.5` indicates that the background values
        are reduced by half before being applied.

"""

# data_dir
docdict["debias"] = """
debias : :obj:`bool`, default=False
    If set, then the estimated weights maps will be debiased.
"""

# data_dir
docdict["data_dir"] = """
data_dir : :obj:`pathlib.Path` or :obj:`str` or None, optional
    Path where data should be downloaded.
    By default, files are downloaded in a ``nilearn_data`` folder
    in the home directory of the user.
    See also ``nilearn.datasets.utils.get_data_dirs``.
"""

# detrend
docdict["detrend"] = """
detrend : :obj:`bool`, optional
    Whether to detrend signals or not.
"""

# dimming factor
docdict["dim"] = """
dim : :obj:`float`, or "auto", optional
    Dimming factor applied to background image.
    By default, automatic heuristics are applied
    based upon the background image intensity.
    Accepted float values, where a typical span is between -2 and 2
    (-2 = increase contrast; 2 = decrease contrast),
    but larger values can be used for a more pronounced effect.
    `0` means no dimming.
"""

# display_mode
docdict["display_mode"] = """
display_mode : {"ortho", "tiled", "mosaic", "x", \
"y", "z", "yx", "xz", "yz"}, default="ortho"
    Choose the direction of the cuts:

    - ``"x"``: sagittal
    - ``"y"``: coronal
    - ``"z"``: axial
    - ``"ortho"``: three cuts are performed in orthogonal directions
    - ``"tiled"``: three cuts are performed and arranged in a 2x2 grid
    - ``"mosaic"``: three cuts are performed along
      multiple rows and columns

"""

# draw_cross
docdict["draw_cross"] = """
draw_cross : :obj:`bool`, default=True
    If `draw_cross` is `True`, a cross is drawn on the plot
    to indicate the cut position.
"""

# dtype
docdict["dtype"] = """
dtype : dtype like, "auto" or None, default=None
    Data type toward which the data should be converted.
    If "auto", the data will be converted to int32
    if dtype is discrete and float32 if it is continuous.
"""

# extractor / extract_type
docdict["extractor"] = """
extractor : {"local_regions", "connected_components"}, default="local_regions"
    This option can take two values:

    - ``"connected_components"``: each component/region in the image
      is extracted automatically by labeling each region based
      upon the presence of unique features in their respective regions.

    - ``"local_regions"``: each component/region is extracted
      based on their maximum peak value to define a seed marker
      and then using random walker segmentation algorithm
      on these markers for region separation.

"""
docdict["extract_type"] = docdict["extractor"].replace(
    "extractor", "extract_type"
)

# figure
docdict["figure"] = """
figure : :obj:`int`, or :class:`matplotlib.figure.Figure`, or None,  optional
    Matplotlib figure used or its number.
    If `None` is given, a new figure is created.
"""

# figure
docdict["first_level_contrast"] = """
first_level_contrast : :obj:`str` or :class:`numpy.ndarray` of \
                        shape (n_col) with respect to \
                        :class:`~nilearn.glm.first_level.FirstLevelModel` \
                        or None, default=None

    When the model is a :class:`~nilearn.glm.second_level.SecondLevelModel`:

    - in case a :obj:`list` of
      :class:`~nilearn.glm.first_level.FirstLevelModel` was provided
      as ``second_level_input``,
      we have to provide a :term:`contrast`
      to apply to the first level models
      to get the corresponding list of images desired,
      that would be tested at the second level,
    - in case a :class:`~pandas.DataFrame` was provided
      as ``second_level_input`` this is the map name to extract
      from the :class:`~pandas.DataFrame` ``map_name`` column.
      (it has to be a 't' contrast).

    This parameter is ignored for all other cases.
"""

# fwhm
docdict["fwhm"] = """
fwhm : scalar, :class:`numpy.ndarray`, or :obj:`tuple`, or :obj:`list`,\
or 'fast' or None, optional
    Smoothing strength, as a :term:`full-width at half maximum<FWHM>`,
    in millimeters:

    - If a nonzero scalar is given, width is identical in all 3 directions.

    - If a :class:`numpy.ndarray`, :obj:`tuple`, or :obj:`list` is given,
      it must have 3 elements, giving the :term:`FWHM` along each axis.
      If any of the elements is `0` or `None`,

      smoothing is not performed along that axis.
    - If `fwhm="fast"`, a fast smoothing will be performed with a filter
      [0.2, 1, 0.2] in each direction and a normalization to preserve the
      local average value.

    - If `fwhm` is `None`, no filtering is performed
      (useful when just removal of non-finite values is needed).

    .. note::

        In corner case situations, `fwhm` is simply kept to `None`
        when `fwhm` is specified as `fwhm=0`.

"""

# groups
docdict["groups"] = """
groups : None, default=None
    Group labels for the samples used
    while splitting the dataset into train/test set.

    Note that this parameter must be specified in some scikit-learn
    cross-validation generators to calculate the number of splits,
    for example sklearn.model_selection.LeaveOneGroupOut or
    sklearn.model_selection.LeavePGroupsOut.

    For more details see
    https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data
"""

# hemi
docdict["hemi"] = """
hemi : {"left", "right", "both"}, default="left"
    Hemisphere to display.
"""

# high_pass
docdict["high_pass"] = """
high_pass : :obj:`float` or :obj:`int` or None, default=None
    High cutoff frequency in Hertz.
    If specified, signals below this frequency will be filtered out.
"""

# hrf_model
docdict["hrf_model"] = """
hrf_model : :obj:`str`, function, :obj:`list` of functions, or None
    This parameter defines the :term:`HRF` model to be used.
    It can be a string if you are passing the name of a model
    implemented in Nilearn.
    Valid names are:

    - ``"spm"``:
      This is the :term:`HRF` model used in :term:`SPM`.
      See :func:`~nilearn.glm.first_level.spm_hrf`.

    - ``"spm + derivative"``:
      SPM model plus its time derivative.
      This gives 2 regressors.
      See :func:`~nilearn.glm.first_level.spm_hrf`, and
      :func:`~nilearn.glm.first_level.spm_time_derivative`.

    - ``"spm + derivative + dispersion"``:
      Same as above plus dispersion derivative.
      This gives 3 regressors.
      See :func:`~nilearn.glm.first_level.spm_hrf`,
      :func:`~nilearn.glm.first_level.spm_time_derivative`,
      and :func:`~nilearn.glm.first_level.spm_dispersion_derivative`.

    - ``"glover"``:
      This corresponds to the Glover :term:`HRF`.
      See :func:`~nilearn.glm.first_level.glover_hrf`.

    - ``"glover + derivative"``:
      The Glover :term:`HRF` + time derivative.
      This gives 2 regressors.
      See :func:`~nilearn.glm.first_level.glover_hrf`, and
      :func:`~nilearn.glm.first_level.glover_time_derivative`.

    - ``"glover"+ derivative + dispersion"``:
      Same as above plus dispersion derivative.
      This gives 3 regressors.
      See :func:`~nilearn.glm.first_level.glover_hrf`,
      :func:`~nilearn.glm.first_level.glover_time_derivative`, and
      :func:`~nilearn.glm.first_level.glover_dispersion_derivative`.

    - ``"fir"``:
      Finite impulse response basis.
      This is a set of delayed dirac models.

    It can also be a custom model.
    In this case, a function should be provided for each regressor.
    Each function should behave as the other models implemented within Nilearn.
    That is, it should take both ``t_r`` and ``oversampling`` as inputs
    and return a sample numpy array of appropriate shape.

    .. note::

        It is expected that ``"spm"`` standard and ``"glover"`` models
        would not yield large differences in most cases.

    .. note::

        In case of ``"glover"`` and ``"spm"`` models,
        the derived regressors are orthogonalized
        with respect to the main one.

"""

# img
docdict["img"] = """
img : Niimg-like object
    See :ref:`extracting_data`.
"""

# imgs
docdict["imgs"] = """
imgs : :obj:`list` of Niimg-like objects
    See :ref:`extracting_data`.
"""

# keep_masked_labels
docdict["keep_masked_labels"] = """
keep_masked_labels : :obj:`bool`, default=True
    When a mask is supplied through the "mask_img" parameter, some
    atlas regions may lie entirely outside of the brain mask, resulting
    in empty time series for those regions.
    If True, the masked atlas with these empty labels will be retained
    in the output, resulting in corresponding time series containing
    zeros only. If False, the empty labels will be removed from the
    output, ensuring no empty time series are present.

    .. deprecated:: 0.10.2

        The 'True' option for ``keep_masked_labels`` is deprecated.
        The default value will change to 'False' in 0.13,
        and the ``keep_masked_labels`` parameter will be removed in 0.15.

"""

# keep_masked_maps
docdict["keep_masked_maps"] = """
keep_masked_maps : :obj:`bool`, optional
    If True, masked atlas with invalid maps (maps that contain only
    zeros after applying the mask) will be retained in the output, resulting
    in corresponding time series containing zeros only. If False, the
    invalid maps will be removed from the trimmed atlas, resulting in
    no empty time series in the output.

    .. deprecated:: 0.10.2

        The 'True' option for ``keep_masked_maps`` is deprecated.
        The default value will change to 'False' in 0.13,
        and the ``keep_masked_maps`` parameter will be removed in 0.15.

"""

# linewidth
docdict["linewidths"] = """
linewidths : :obj:`float`, optional
    Set the boundary thickness of the contours.
    Only reflects when `view_type=contours`.
"""

# low_pass
docdict["low_pass"] = """
low_pass : :obj:`float` or :obj:`int` or None, default=None
    Low cutoff frequency in Hertz.
    If specified, signals above this frequency will be filtered out.
    If `None`, no low-pass filtering will be performed.
"""

# lower_cutoff
docdict["lower_cutoff"] = """
lower_cutoff : :obj:`float`, optional
    Lower fraction of the histogram to be discarded.
"""

# masker_lut
docdict["masker_lut"] = """lut : :obj:`pandas.DataFrame` or :obj:`str` \
            or :obj:`pathlib.Path` to a TSV file or None, default=None
        Mutually exclusive with ``labels``.
        Act as a look up table (lut)
        with at least columns 'index' and 'name'.
        Formatted according to 'dseg.tsv' format from
        `BIDS <https://bids-specification.readthedocs.io/en/latest/derivatives/imaging.html#common-image-derived-labels>`_.

        .. warning::

            If a region exist in the atlas image
            but is missing from its associated LUT,
            a new entry will be added to the LUT during fit
            with the name "unknown".
            Conversely, if regions listed in the LUT do not exist
            in the associated atlas image,
            they will be dropped from the LUT during fit.
        """


# mask_strategy
docdict["mask_strategy"] = """
mask_strategy : {"background", "epi", "whole-brain-template",\
"gm-template", "wm-template"}, optional
    The strategy used to compute the mask:

    - ``"background"``: Use this option if your images present
      a clear homogeneous background. Uses
      :func:`nilearn.masking.compute_background_mask` under the hood.

    - ``"epi"``: Use this option if your images are raw EPI images. Uses
      :func:`nilearn.masking.compute_epi_mask`.

    - ``"whole-brain-template"``: This will extract the whole-brain
      part of your data by resampling the MNI152 brain mask for
      your data's field of view. Uses
      :func:`nilearn.masking.compute_brain_mask` with
      ``mask_type="whole-brain"``.

      .. note::

          This option is equivalent to the previous 'template' option
          which is now deprecated.

    - ``"gm-template"``: This will extract the gray matter part of your
      data by resampling the corresponding MNI152 template for your
      data's field of view. Uses
      :func:`nilearn.masking.compute_brain_mask` with ``mask_type="gm"``.

      .. versionadded:: 0.8.1

    - ``"wm-template"``: This will extract the white matter part of your
      data by resampling the corresponding MNI152 template for your
      data's field of view. Uses
      :func:`nilearn.masking.compute_brain_mask` with ``mask_type="wm"``.

      .. versionadded:: 0.8.1
"""

# mask_type
docdict["mask_type"] = """
mask_type : {"whole-brain", "gm", "wm"}, default="whole-brain"
    Type of mask to be computed:

    - ``"whole-brain"``: Computes the whole-brain mask.
    - ``"gm"``: Computes the grey-matter mask.
    - ``"wm"``: Computes the white-matter mask.

"""

# kwargs for Maskers
docdict["masker_kwargs"] = """
kwargs : dict
    Keyword arguments to be passed to functions called within the masker.
    Kwargs prefixed with `'clean__'` will be passed to
    :func:`~nilearn.signal.clean`.
    Within :func:`~nilearn.signal.clean`, kwargs prefixed with
    `'butterworth__'` will be passed to the Butterworth filter
    (i.e., `clean__butterworth__`).

    .. deprecated:: 0.12.0

    .. admonition:: Use ``clean_args`` instead!
       :class: important

       It is recommended to pass parameters to use for data cleaning
       via :obj:`dict` to the ``clean_args`` parameter.

       Passing parameters via "kwargs" is mutually exclusive
       with passing cleaning parameters via ``clean_args``.
"""

docdict["masker_kwargs_"] = docdict["masker_kwargs"].replace(
    "kwargs : dict",
    "clean_kwargs_ : dict",
)

verbose = """
max_iter : :obj:`int`, default={}
    Maximum number of iterations for the solver.
"""
docdict["max_iter"] = verbose.format(200)
docdict["max_iter10"] = verbose.format(10)
docdict["max_iter50"] = verbose.format(50)
docdict["max_iter100"] = verbose.format(100)
docdict["max_iter1000"] = verbose.format(1000)
docdict["max_iter1000"] = verbose.format(5000)

# memory
docdict["memory"] = """
memory : None, instance of :class:`joblib.Memory`, :obj:`str`, or \
:class:`pathlib.Path`
    Used to cache the masking process.
    By default, no caching is done.
    If a :obj:`str` is given, it is the path to the caching directory.
"""

# memory_level
memory_level = """
memory_level : :obj:`int`, default={}
    Rough estimator of the amount of memory used by caching.
    Higher value means more memory for caching.
    Zero means no caching.
"""
docdict["memory_level"] = memory_level.format(0)
docdict["memory_level1"] = memory_level.format(1)

# n_jobs
n_jobs = """
n_jobs : :obj:`int`, default={}
    The number of CPUs to use to do the computation.
    `-1` means 'all CPUs'.
"""
docdict["n_jobs"] = n_jobs.format("1")
docdict["n_jobs_all"] = n_jobs.format("-1")

# n_jobs
docdict["n_perm"] = """
n_perm : :obj:`int`, default=10000
    Number of permutations to perform.
    Permutations are costly but the more are performed, the more precision
    one gets in the p-values estimation.
"""

# opening
docdict["opening"] = """
opening : :obj:`bool` or :obj:`int`, optional
    This parameter determines whether a morphological
    :term:`opening<Opening>` is performed, to keep only large structures.
    This step is useful to remove parts of the skull that might have been
    included. `opening` can be:

    - A :obj:`bool` : If `False`, no :term:`opening<Opening>` is performed.
      If `True`, it is equivalent to `opening=1`.

    - An :obj:`int` `n`: The :term:`opening<Opening>` is performed via `n`
      :term:`erosions<Erosion>` (see :func:`scipy.ndimage.binary_erosion`).
      The largest connected component is then estimated
      if `connected` is set to `True`,
      and 2`n` :term:`dilation<Dilation>` operations are performed
      (see :func:`scipy.ndimage.binary_dilation`)
      followed by `n` :term:`erosions<Erosion>`.
      This corresponds to 1 :term:`opening<Opening>` operation
      of order `n` followed by a :term:`closing<Closing>` operator
      of order `n`.

    .. note::

        Turning off :term:`opening<Opening>` (`opening=False`) will also
        prevent any smoothing applied to the image during the mask computation.

"""

# output_file
docdict["output_file"] = """
output_file : :obj:`str` or :obj:`pathlib.Path` or None, optional
    The name of an image file to export the plot to.
    Valid extensions are .png, .pdf, .svg.
    If `output_file` is not `None`, the plot is saved to a file,
    and the display is closed.
"""

# radiological
docdict["radiological"] = """
radiological : :obj:`bool`, default=False
    Invert x axis and R L labels to plot sections as a radiological view.
    If False (default), the left hemisphere is on the left of a coronal image.
    If True, left hemisphere is on the right.
"""

# random_state
docdict["random_state"] = """
random_state : :obj:`int` or np.random.RandomState, optional
    Pseudo-random number generator state used for random sampling.
"""

# regressor_options
docdict["regressor_options"] = """

    - ``ridge``: \
        :class:`{Ridge regression} <sklearn.linear_model.RidgeCV>`.

    .. code-block:: python

        ridge = RidgeCV()

    - ``ridge_regressor``: \
        :class:`{Ridge regression} <sklearn.linear_model.RidgeCV>`.

    .. note::

        Same option as `ridge`.

    - ``svr``: :class:`{Support vector regression} <sklearn.svm.SVR>`.

    .. code-block:: python

        svr = SVR(kernel="linear", max_iter=1e4)

    - ``lasso``: \
        :class:`{Lasso regression} <sklearn.linear_model.LassoCV>`.

    .. code-block:: python

        lasso = LassoCV()

    - ``lasso_regressor``: \
        :class:`{Lasso regression} <sklearn.linear_model.LassoCV>`.

    .. note::

        Same option as `lasso`.

    - ``dummy_regressor``: \
        :class:`{Dummy regressor} <sklearn.dummy.DummyRegressor>`.

    .. code-block:: python

        dummy = DummyRegressor(strategy="mean")

"""

# resampling_interpolation
docdict["resampling_interpolation"] = """
resampling_interpolation : :obj:`str`, optional
    Interpolation to use when resampling the image to
    the destination space. Can be:

    - ``"continuous"``: use 3rd-order spline interpolation
    - ``"nearest"``: use nearest-neighbor mapping.

    .. note::

        ``"nearest"`` is faster but can be noisier in some cases.

"""

# resolution template
docdict["resolution"] = """
resolution : :obj:`int` or None, default=None
        Resolution in millimeters.
        If resolution is different from 1,
        the template is re-sampled with the specified resolution.
        Default to ``1`` if None is passed.
"""

# resume
docdict["resume"] = """
resume : :obj:`bool`, default=True
    Whether to resume download of a partly-downloaded file.
"""

# sample_mask
docdict["sample_mask"] = """
sample_mask : Any type compatible with numpy-array indexing, default=None
    ``shape = (total number of scans - number of scans removed)``
    for explicit index (for example, ``sample_mask=np.asarray([1, 2, 4])``),
    or ``shape = (number of scans)`` for binary mask
    (for example,
    ``sample_mask=np.asarray([False, True, True, False, True])``).
    Masks the images along the last dimension to perform scrubbing:
    for example to remove volumes with high motion
    and/or non-steady-state volumes.
    This parameter is passed to :func:`nilearn.signal.clean`.
"""
docdict["sample_mask_multi"] = """
sample_mask : :obj:`list` of sample_mask, default=None
    List of sample_mask (any type compatible with numpy-array indexing)
    to use for scrubbing outliers.
    Must be of same length as ``imgs``.
    ``shape = (total number of scans - number of scans removed)``
    for explicit index (for example, ``sample_mask=np.asarray([1, 2, 4])``),
    or ``shape = (number of scans)`` for binary mask
    (for example,
    ``sample_mask=np.asarray([False, True, True, False, True])``).
    Masks the images along the last dimension to perform scrubbing:
    for example to remove volumes with high motion
    and/or non-steady-state volumes.
    This parameter is passed to :func:`nilearn.signal.clean`.
"""

docdict["screening_percentile"] = """
screening_percentile : int, float, \
                       in the closed interval [0, 100], or None, \
                       default=20
        Percentile value for ANOVA univariate feature selection.
        If ``None`` is passed, it will be set to ``100``.
        A value of ``100`` means "keep all features".
        This percentile is expressed
        with respect to the volume of either a standard (MNI152) brain
        (if ``mask_img_`` is a 3D volume)
        or a the number of vertices in the mask mesh
        (if ``mask_img_`` is a SurfaceImage).
        This means that the
        ``screening_percentile`` is corrected at runtime by premultiplying it
        with the ratio of volume of the
        standard brain to the volume of the mask of the data.

        .. note::

            If the mask used is too small
            compared to the total brain volume / surface,
            then all its elements (voxels / vertices)
            may be included even for very small ``screening_percentile``.

"""

# second_level_contrast
docdict["second_level_contrast"] = """
second_level_contrast : :obj:`str` or :class:`numpy.ndarray` of shape\
(n_col), optional
    Where `n_col` is the number of columns of the design matrix.
    The string can be a formula compatible with :meth:`pandas.DataFrame.eval`.
    Basically one can use the name of the conditions as they appear
    in the design matrix of the fitted model combined with operators +-
    and combined with numbers with operators +-`*`/.
    The default `None` is accepted if the design matrix has a single column,
    in which case the only possible contrast array((1)) is applied;
    when the design matrix has multiple columns, an error is raised.
"""

# second_level_confounds
docdict["second_level_confounds"] = """
confounds : :obj:`pandas.DataFrame` or None, default=None
    Must contain a ``subject_label`` column.
    All other columns are considered as confounds and included in the model.
    If ``design_matrix`` is provided then this argument is ignored.
    The resulting second level design matrix uses the same column names
    as in the given :class:`~pandas.DataFrame` for confounds.
    At least two columns are expected, ``subject_label``
    and at least one confound.
"""

# second_level_design_matrix
docdict["second_level_design_matrix"] = """
design_matrix : :obj:`pandas.DataFrame`, :obj:`str` or \
                or :obj:`pathlib.Path` to a CSV or TSV file, \
                or None, default=None
    Design matrix to fit the :term:`GLM`.
    The number of rows in the design matrix
    must agree with the number of maps
    derived from ``second_level_input``.
    Ensure that the order of maps given by a ``second_level_input``
    list of Niimgs matches the order of the rows in the design matrix.
"""

# second_level_input
docdict["second_level_input"] = """
second_level_input : :obj:`list` of \
    :class:`~nilearn.glm.first_level.FirstLevelModel` objects or \
    :class:`pandas.DataFrame` or \
    :obj:`list` of 3D Niimg-like objects or \
    4D Niimg-like objects or \
    :obj:`list` of :class:`~nilearn.surface.SurfaceImage` objects or \
    :obj:`pandas.Series` of Niimg-like objects.

    - Giving :class:`~nilearn.glm.first_level.FirstLevelModel` objects
      will allow to easily compute the second level contrast of arbitrary first
      level contrasts thanks to the `first_level_contrast` argument of
      :meth:`~nilearn.glm.first_level.FirstLevelModel.compute_contrast`.
      Effect size images will be computed for each model
      to contrast at the second level.

    - If a :class:`~pandas.DataFrame`, then it has to contain
      `subject_label`, `map_name` and `effects_map_path`.
      It can contain multiple maps that would be selected
      during contrast estimation with the argument `first_level_contrast`
      of :meth:`~nilearn.glm.first_level.FirstLevelModel.compute_contrast`.
      The :class:`~pandas.DataFrame` will be sorted
      based on the `subject_label` column to avoid order inconsistencies
      when extracting the maps.
      So the rows of the automatically computed design matrix,
      if not provided, will correspond to the sorted `subject_label` column.

    - If a :obj:`list` of Niimg-like objects
      or :class:`~nilearn.surface.SurfaceImage` objects
      then this is taken literally as Y for the model fit
      and `design_matrix` must be provided.

"""

# second_level_mask_img
docdict["second_level_mask_img"] = """
mask_img : Niimg-like, :obj:`~nilearn.maskers.NiftiMasker` or\
            :obj:`~nilearn.maskers.MultiNiftiMasker` or\
            :obj:`~nilearn.maskers.SurfaceMasker` object or None,\
            default=None
    Mask to be used on data.
    If an instance of masker is passed,
    then its mask will be used.
    If no mask is given,
    it will be computed automatically
    by a :class:`~nilearn.maskers.NiftiMasker`,
    or a :obj:`~nilearn.maskers.SurfaceMasker`
    (depending on the type passed at fit time)
    with default parameters.
    Automatic mask computation assumes first level imgs
    have already been masked.
"""
docdict["second_level_mask"] = docdict["second_level_mask_img"].replace(
    "mask_img :", "mask :"
)

# signals for inverse transform
docdict["signals_inv_transform"] = """
signals : 1D/2D :obj:`numpy.ndarray`
    Extracted signal.
    If a 1D array is provided,
    then the shape should be (number of elements,).
    If a 2D array is provided,
    then the shape should be (number of scans, number of elements).
"""
docdict["region_signals_inv_transform"] = docdict["signals_inv_transform"]
docdict["x_inv_transform"] = docdict["signals_inv_transform"]


# smoothing_fwhm
docdict["smoothing_fwhm"] = """
smoothing_fwhm : :obj:`float` or :obj:`int` or None, optional.
    If `smoothing_fwhm` is not `None`,
    it gives the :term:`full-width at half maximum<FWHM>` in millimeters
    of the spatial smoothing to apply to the signal.
"""

# standardize
standardize = """
standardize : :obj:`bool`, default={}
    If `standardize` is `True`, the data are centered and normed:
    their mean is put to 0 and their variance is put to 1
    in the time dimension.
"""
docdict["standardize"] = standardize.format("True")
docdict["standardize_false"] = standardize.format("False")

# standardize as used within maskers module
docdict["standardize_maskers"] = """
standardize : {'zscore_sample', 'zscore', 'psc', True, False}, default=False
    Strategy to standardize the signal:

    - ``'zscore_sample'``: The signal is z-scored. Timeseries are shifted
      to zero mean and scaled to unit variance. Uses sample std.

    - ``'zscore'``: The signal is z-scored. Timeseries are shifted
      to zero mean and scaled to unit variance. Uses population std
      by calling default :obj:`numpy.std` with N - ``ddof=0``.

    - ``'psc'``:  Timeseries are shifted to zero mean value and scaled
      to percent signal change (as compared to original mean signal).

    - ``True``: The signal is z-scored (same as option `zscore`).
      Timeseries are shifted to zero mean and scaled to unit variance.

    - ``False``: Do not standardize the data.

"""

# standardize_confounds
docdict["standardize_confounds"] = """
standardize_confounds : :obj:`bool`, default=True
    If set to `True`, the confounds are z-scored:
    their mean is put to 0 and their variance to 1 in the time dimension.
"""

# standardize_confounds
docdict["strategy"] = """
strategy : :obj:`str`, default="mean"
    The name of a valid function to reduce the region with.
    Must be one of: sum, mean, median, minimum, maximum, variance,
    standard_deviation.
"""

# surf_mesh
docdict["surf_mesh"] = """
surf_mesh : :obj:`str` or :obj:`list` of two :class:`numpy.ndarray` \
            or a :obj:`~nilearn.surface.InMemoryMesh`, or a \
            :obj:`~nilearn.surface.PolyMesh`, or None, default=None
    Surface :term:`mesh` geometry, can be a file (valid formats are .gii or
    Freesurfer specific files such as .orig, .pial, .sphere, .white,
    .inflated) or a list of two Numpy arrays, the first containing the
    x-y-z coordinates of the :term:`mesh` :term:`vertices<vertex>`, the
    second containing the indices (into coords) of the :term:`mesh`
    :term:`faces`, or a :obj:`~nilearn.surface.InMemoryMesh` object with
    "coordinates" and "faces" attributes, or a
    :obj:`~nilearn.surface.PolyMesh` object, or None.
"""

# symmetric_cbar
docdict["symmetric_cbar"] = """
symmetric_cbar : :obj:`bool`, or "auto", default="auto"
    Specifies whether the colorbar and colormap should range from `-vmax` to
    `vmax` (or from `vmin` to `-vmin` if `-vmin` is greater than `vmax`) or
    from `vmin` to `vmax`.
    Setting to `"auto"` (the default) will select the former if either
    `vmin` or `vmax` is `None` and the image has both positive and negative
    values.
"""

# t_r
docdict["t_r"] = """
t_r : :obj:`float` or :obj:`int` or None, default=None
    :term:`Repetition time<TR>`, in seconds (sampling period).
    Set to `None` if not provided.
"""

# target_affine
docdict["target_affine"] = """
target_affine : :class:`numpy.ndarray` or None, default=None
    If specified, the image is resampled corresponding to this new affine.
    `target_affine` can be a 3x3 or a 4x4 matrix.
"""

# target_shape
docdict["target_shape"] = """
target_shape : :obj:`tuple` or :obj:`list` or None, default=None
    If specified, the image will be resized to match this new shape.
    `len(target_shape)` must be equal to 3.

    .. note::

        If `target_shape` is specified, a `target_affine` of shape
        `(4, 4)` must also be given.

"""

# threshold
docdict["tfce"] = """
tfce : :obj:`bool`, default=False
    Whether to calculate :term:`TFCE`
    as part of the permutation procedure or not.
    The TFCE calculation is implemented
    as described in :footcite:t:`Smith2009a`.

    .. note::

       The number of thresholds used in the TFCE procedure
       will set between 10 and 1000.

       .. versionadded:: 0.12.0

    .. warning::

        Performing TFCE-based inference
        will increase the computation time
        of the permutation procedure considerably.
        The permutations may take multiple hours,
        depending on how many permutations
        are requested and how many jobs are performed in parallel.
"""

# threshold
docdict["threshold"] = """
threshold : :obj:`int` or :obj:`float`, None, or 'auto', optional
    If `None` is given, the image is not thresholded.
    If number is given, it must be non-negative. The specified value is used to
    threshold the image: values below the threshold (in absolute value) are
    plotted as transparent.
    If "auto" is given, the threshold is determined based on the score obtained
    using percentile value "80%" on the absolute value of the image data.
"""

# title
docdict["title"] = """
title : :obj:`str`, or None, default=None
    The title displayed on the figure.
"""

# transparency
docdict["transparency"] = """
transparency : :obj:`float` between 0 and 1, \
                or a Niimg-Like object, \
                or None, \
                default = None
    Value to be passed as alpha value to :func:`~matplotlib.pyplot.imshow`.
    if ``None`` is passed, it will be set to 1.
    If an image is passed, voxel-wise alpha blending will be applied,
    by relying on the absolute value of ``transparency`` at each voxel.

    .. versionadded:: 0.12.0
"""

# transparency
docdict["transparency_range"] = """
transparency_range : :obj:`tuple` or :obj:`list` of 2 non-negative numbers, \
                or None, \
                default = None
    When an image is passed to ``transparency``,
    this determines the range of values in the image
    to use for transparency (alpha blending).
    For example with ``transparency_range = [1.96, 3]``,
    any voxel / vertex (:math:`v_i`):

    - with a value between between -1.96 and 1.96,
      would be fully transparent (alpha = 0),
    - with a value less than -3 or greater than 3,
      would be fully opaque (alpha = 1),
    - with a value in the intervals ``[-3.0, -1.96]`` or ``[1.96, 3.0]``,
      would have an alpha_i value
      scaled linearly between 0 and 1 :
      :math:`alpha_i = (\\lvert v_i \\lvert - 1.96) / (3.0 - 1.96)`.

    This parameter will be ignored
    unless an image is passed as ``transparency``.
    The first number must be greater than 0 and less than the second one.
    if ``None`` is passed,
    this will be set to ``[0, max(abs(transparency))]``.

    .. versionadded:: 0.12.0
"""

# upper_cutoff
docdict["upper_cutoff"] = """
upper_cutoff : :obj:`float`, optional
    Upper fraction of the histogram to be discarded.
"""

# two_sided_test
docdict["two_sided_test"] = """
two_sided_test : :obj:`bool`, default=False

    - If ``True``, performs an unsigned t-test.
        Both positive and negative effects are considered; the null
        hypothesis is that the effect is zero.
    - If ``False``, only positive effects are considered as relevant.
        The null hypothesis is that the effect is zero or negative.
"""

# url
docdict["url"] = """
url : :obj:`str` or None, default=None
    URL of file to download.
    Override download URL.
    Used for test only (or if you setup a mirror of the data).
"""

# verbose
verbose = """
verbose : :obj:`int`, default={}
    Verbosity level (`0` means no message).
"""
docdict["verbose"] = verbose.format(1)
docdict["verbose0"] = verbose.format(0)
docdict["verbose2"] = verbose.format(2)
docdict["verbose3"] = verbose.format(3)

# view
docdict["view"] = """
view : :obj:`str`, or a pair of :obj:`float` or :obj:`int`, default="lateral"\
    if `hemi` is "left" or "right", if `hemi` is "both" "dorsal"
    If a string, and `hemi` is "left" or "right" must be in
    {"lateral", "medial", "dorsal", "ventral", "anterior", "posterior"}.
    If `hemi` is "both", must be in {"left", "right", "dorsal", "ventral",
    "anterior", "posterior"}.
    If a sequence, must be a pair (elev, azim) of :obj:`float` or :obj:`int`
    angles in degrees that will manually set a custom view.
    E.g., view=[270.0, 90] or view=(0, -180.0).
    View of the surface that is rendered.
"""

# vmax
docdict["vmax"] = """
vmax : :obj:`float` or obj:`int` or None, optional
    Upper bound of the colormap. The values above vmax are masked.
    If `None`, the max of the image is used.
    Passed to :func:`matplotlib.pyplot.imshow`.
"""

# vmin
docdict["vmin"] = """
vmin : :obj:`float`  or obj:`int` or None, optional
    Lower bound of the colormap. The values below vmin are masked.
    If `None`, the min of the image is used.
    Passed to :func:`matplotlib.pyplot.imshow`.
"""

# y
docdict["y_dummy"] = """
y : None
    This parameter is unused.
    It is solely included for scikit-learn compatibility.
"""


##############################################################################
#
# Other values definitions: return values, attributes...
#

# atlas_type
docdict["atlas_type"] = """'atlas_type' : :obj:`str`
        Type of atlas.
        See :term:`Probabilistic atlas` and :term:`Deterministic atlas`."""

docdict["base_decomposition_fit_attributes"] = """
Attributes
----------
maps_masker_ : instance of NiftiMapsMasker or SurfaceMapsMasker
    This masker was initialized with
    ``components_img_``, ``masker_.mask_img_``
    and is the masker used
    when calliing transform and inverse_transform.

mask_img_ : Niimg-like object or :obj:`~nilearn.surface.SurfaceImage`
    See :ref:`extracting_data`.
    The mask of the data.
    If no mask was given at masker creation :

    - for Nifti images, this contains automatically computed mask
        via the selected ``mask_strategy``.

    - for SurfaceImage objects, this mask encompasses all vertices of
        the input images.

"""

docdict["multi_pca_fit_attributes"] = """
components_ : 2D numpy array (n_components x n-voxels or n-vertices)
    Array of masked extracted components.

    .. note::

        Use attribute ``components_img_``
        rather than manually unmasking
        ``components_`` with ``masker_`` attribute.

components_img_ : 4D Nifti image \
                    or 2D :obj:`~nilearn.surface.SurfaceImage`
    The image giving the extracted components.
    Each 3D Nifti image or 1D SurfaceImage is a component.

    .. versionadded:: 0.4.1

masker_ :  :obj:`~nilearn.maskers.MultiNiftiMasker` or \
        :obj:`~nilearn.maskers.SurfaceMasker`
    Masker used to filter and mask data as first step.
    If :obj:`~nilearn.maskers.MultiNiftiMasker`
    or :obj:`~nilearn.maskers.SurfaceMasker` is given in
    ``mask`` parameter, this is a copy of it.
    Otherwise, a masker is created using the value of ``mask`` and
    other NiftiMasker/SurfaceMasker
    related parameters as initialization.

memory_ : joblib memory cache

n_elements_ : :obj:`int`
    The number of components.

"""

docdict["base_decoder_fit_attributes"] = """
Attributes
----------
coef_ : numpy.ndarray, shape=(n_classes, n_features)
    Contains the mean of the models weight vector across
    fold for each class. Returns None for Dummy estimators.

coef_img_ : :obj:`dict` of Nifti1Image
    Dictionary containing ``coef_`` with class names as keys,
    and ``coef_`` transformed in Nifti1Images as values.
    In the case of a regression,
    it contains a single Nifti1Image at the key 'beta'.
    Ignored if Dummy estimators are provided.

cv_ : :obj:`list` of pairs of lists
    List of the (n_folds,) folds.
    For the corresponding fold,
    each pair is composed of two lists of indices,
    one for the train samples and one for the test samples.

cv_params_ : :obj:`dict` of :obj:`list`
    Best point in the parameter grid for each tested fold
    in the inner cross validation loop.
    The grid is empty
    when Dummy estimators are provided.

    .. note::

        If the estimator used its built-in cross-validation,
        this will include an additional key
        for the single best value estimated
        by the built-in cross-validation
        ('best_C' for LogisticRegressionCV
        and 'best_alpha' for RidgeCV/RidgeClassifierCV/LassoCV),
        in addition to the input list of values.

cv_scores_ : :obj:`dict`, (classes, n_folds)
    Scores (misclassification) for each parameter, and on each fold

dummy_output_ : ndarray, shape=(n_classes, 2) \
                or shape=(1, 1) for regression
    Contains dummy estimator attributes after class predictions
    using strategies of :class:`sklearn.dummy.DummyClassifier`
    (class_prior)
    and  :class:`sklearn.dummy.DummyRegressor` (constant)
    from scikit-learn.
    This attribute is necessary for estimating class predictions
    after fit.
    Returns None if non-dummy estimators are provided.

estimator_ : Estimator object used during decoding.

intercept_ : ndarray, shape (nclasses,)
    Intercept (also known as bias) added to the decision function.
    Ignored if Dummy estimators are provided.

mask_img_ : Nifti1Image or :obj:`~nilearn.surface.SurfaceImage`
    Mask computed by the masker object.

masker_ : instance of NiftiMasker, MultiNiftiMasker, or SurfaceMasker
    The masker used to mask the data.

memory_ : joblib memory cache

n_elements_ : :obj:`int`
    The number of voxels or vertices in the mask.

    .. versionadded:: 0.12.1

n_outputs_ : :obj:`int`
    Number of outputs (column-wise)

scorer_ : function
    Scorer function used on the held out data to choose the best
    parameters for the model.

screening_percentile_ : :obj:`float`
    Percentile value for ANOVA univariate feature selection.
    A value of 100 means 'keep all features'.
    This percentile is expressed
    with respect to the volume of either a standard (MNI152) brain
    (if mask_img is a 3D volume)
    or a the number of vertices in the mask mesh
    (if mask_img is a SurfaceImage).

std_coef_ : numpy.ndarray, shape=(n_classes, n_features)
    Contains the standard deviation of the models weight vector across
    fold for each class.
    Note that folds are not independent,
    see
    https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data
    Ignored if Dummy estimators are provided.

std_coef_img_ : :obj:`dict` of Nifti1Image
    Dictionary containing `std_coef_` with class names as keys,
    and `coef_` transformed in Nifti1Image as values.
    In the case of a regression,
    it contains a single Nifti1Image at the key 'beta'.
    Ignored if Dummy estimators are provided.
"""

docdict["spacenet_fit_attributes"] = """
Attributes
----------
all_coef_ : ndarray, shape (n_l1_ratios, n_folds, n_features)
    Coefficients for all folds and features.

alpha_grids_ : ndarray, shape (n_folds, n_alphas)
    Alpha values considered for selection of the best ones
    (saved in `best_model_params_`)

best_model_params_ : ndarray, shape (n_folds, n_parameter)
    Best model parameters (alpha, l1_ratio) saved for the different
    cross-validation folds.

coef_ : ndarray, shape\
    (1, n_features) for 2 class classification problems\
    (i.e n_classes = 2)\
    (n_classes, n_features) for n_classes > 2
    Coefficient of the features in the decision function.

coef_img_ : nifti image
    Masked model coefficients

cv_ : list of pairs of lists
    Each pair is the list of indices for the train and test samples
    for the corresponding fold.

cv_scores_ : ndarray, shape (n_folds, n_alphas)\
    or (n_l1_ratios, n_folds, n_alphas)
    Scores (misclassification) for each alpha, and on each fold

intercept_ : narray, shape
    (1,) for 2 class classification problems (i.e n_classes = 2)
    (n_classes,) for n_classes > 2
    Intercept (a.k.a. bias) added to the decision function.
    It is available only when parameter intercept is set to True.

mask_ : ndarray 3D
    An array contains values of the mask image.

masker_ : instance of NiftiMasker
    The nifti masker used to mask the data.

mask_img_ : Nifti like image
    The mask of the data. If no mask was supplied by the user,
    this attribute is the mask image computed automatically from the
    data `X`.

memory_ : joblib memory cache

n_elements_ : :obj:`int`
    The number of features in the mask.

    .. versionadded:: 0.12.1

screening_percentile_ : float
    Screening percentile corrected according to volume of mask,
    relative to the volume of standard brain.

w_ : ndarray, shape
    (1, n_features + 1) for 2 class classification problems
    (i.e n_classes = 2)
    (n_classes, n_features + 1) for n_classes > 2, and (n_features,)
    for regression
    Model weights

Xmean_ : array, shape (n_features,)
    Mean of X across samples

Xstd_ : array, shape (n_features,)
    Standard deviation of X across samples

ymean_ : array, shape (n_samples,)
    Mean of prediction targets

"""

# dataset description
docdict["description"] = """'description' : :obj:`str`
        Description of the dataset."""

# fsaverage options
docdict["fsaverage_options"] = """

    - ``"fsaverage3"``: the low-resolution fsaverage3 mesh (642 nodes)
    - ``"fsaverage4"``: the low-resolution fsaverage4 mesh (2562 nodes)
    - ``"fsaverage5"``: the low-resolution fsaverage5 mesh (10242 nodes)
    - ``"fsaverage6"``: the medium-resolution fsaverage6 mesh (40962 nodes)
    - ``"fsaverage7"``: same as `"fsaverage"`
    - ``"fsaverage"``: the high-resolution fsaverage mesh (163842 nodes)

    .. note::

        The high-resolution fsaverage will result in more computation
        time and memory usage

"""

# image returned Nifti maskers by inverse_transform
docdict["img_inv_transform_nifti"] = """img : :obj:`nibabel.nifti1.Nifti1Image`
        Transformed image in brain space.
        Output shape for :

        - 1D array : 3D :obj:`nibabel.nifti1.Nifti1Image` will be returned.
        - 2D array : 4D :obj:`nibabel.nifti1.Nifti1Image` will be returned.

        See :ref:`extracting_data`.
        """
# image returned surface maskers by inverse_transform
docdict[
    "img_inv_transform_surface"
] = """img : :obj:`~nilearn.surface.SurfaceImage`
        Signal for each vertex projected on the mesh.
        Output shape for :

        - 1D array : 1D :obj:`~nilearn.surface.SurfaceImage` will be returned.
        - 2D array : 2D :obj:`~nilearn.surface.SurfaceImage` will be returned.

        See :ref:`extracting_data`.
        """

# atlas labels
docdict["labels"] = """'labels' : :obj:`list` of :obj:`str`
        List of the names of the regions."""

# mask_img_ for most nifti maskers
docdict[
    "nifti_mask_img_"
] = """mask_img_ : A 3D binary :obj:`nibabel.nifti1.Nifti1Image` or None.
        The mask of the data.
        If no ``mask_img`` was passed at masker construction,
        then ``mask_img_`` is ``None``, otherwise
        is the resulting binarized version of ``mask_img``
        where each voxel is ``True`` if all values across samples
        (for example across timepoints) is finite value different from 0."""

# look up table
docdict["lut"] = """lut : :obj:`pandas.DataFrame`
        Act as a look up table (lut)
        with at least columns 'index' and 'name'.
        Formatted according to 'dseg.tsv' format from
        `BIDS <https://bids-specification.readthedocs.io/en/latest/derivatives/imaging.html#common-image-derived-labels>`_."""

# signals returned Nifti maskers by transform, fit_transform...
docdict["signals_transform_nifti"] = """signals : :obj:`numpy.ndarray`
        Signal for each :term:`voxel`.
        Output shape for :

        - 3D images: (number of elements,) array
        - 4D images: (number of scans, number of elements) array
        """
# signals returned Mulit Nifti maskers by transform, fit_transform...
docdict[
    "signals_transform_multi_nifti"
] = """signals : :obj:`list` of :obj:`numpy.ndarray` or :obj:`numpy.ndarray`
        Signal for each :term:`voxel`.
        Output shape for :

        - 3D images: (number of elements,) array
        - 4D images: (number of scans, number of elements) array
        - list of 3D images: list of (number of elements,) array
        - list of 4D images: list of (number of scans, number of elements)
          array
        """
# signals returned Mulit Nifti maskers by transform, fit_transform...
docdict[
    "signals_transform_imgs_multi_nifti"
] = """signals : :obj:`list` of :obj:`numpy.ndarray`
        Signal for each :term:`voxel`.
        Output shape for :

        - list of 3D images: list of (number of elements,) array
        - list of 4D images: list of (number of scans, number of elements)
          array
        """
# signals returned surface maskers by transform, fit_transform...
docdict["signals_transform_surface"] = """signals : :obj:`numpy.ndarray`
        Signal for each element.
        Output shape for :

        - 1D images: (number of elements,) array
        - 2D images: (number of scans, number of elements) array
        """

# template
docdict["template"] = """'template' : :obj:`str`
        The standardized space of analysis
        in which the atlas results are provided.
        When known it should be a valid template name
        taken from the spaces described in
        `the BIDS specification <https://bids-specification.readthedocs.io/en/latest/appendices/coordinate-systems.html#image-based-coordinate-systems>`_."""


# templateflow
docdict["templateflow"] = """

.. admonition:: Nilearn MNI template
   :class: important

   The Nilearn template is asymmetrical ICBM152 2009, release a.

   The default template of :term:`fMRIPrep` is the asymmetrical ICBM152 2009,
   release c (MNI152NLin2009cSAsym).

   If you wish to use the exact same release as :term:`fMRIPrep`,
   please refer to `TemplateFlow <https://www.templateflow.org>`_.

"""

##############################################################################

docdict_indented: dict[int, dict[str, str]] = {}


def _indentcount_lines(lines):
    """Minimum indent for all lines in line list.

    >>> lines = [" one", "  two", "   three"]
    >>> _indentcount_lines(lines)
    1
    >>> lines = []
    >>> _indentcount_lines(lines)
    0
    >>> lines = [" one"]
    >>> _indentcount_lines(lines)
    1
    >>> _indentcount_lines(["    "])
    0

    """
    indentno = sys.maxsize
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indentno = min(indentno, len(line) - len(stripped))
    if indentno == sys.maxsize:
        return 0
    return indentno


def fill_doc(f):
    """Fill a docstring with docdict entries.

    Parameters
    ----------
    f : callable
        The function to fill the docstring of. Will be modified in place.

    Returns
    -------
    f : callable
        The function, potentially with an updated `__doc__`.

    """
    docstring = f.__doc__
    if not docstring:
        return f
    lines = docstring.splitlines()
    # Find the minimum indent of the main docstring, after first line
    icount = 0 if len(lines) < 2 else _indentcount_lines(lines[1:])
    # Insert this indent to dictionary docstrings
    try:
        indented = docdict_indented[icount]
    except KeyError:
        indent = " " * icount
        docdict_indented[icount] = indented = {}
        for name, dstr in docdict.items():
            lines = dstr.splitlines()
            try:
                newlines = [lines[0]] + [indent + line for line in lines[1:]]
                indented[name] = "\n".join(newlines)
            except IndexError:
                indented[name] = dstr
    try:
        f.__doc__ = docstring % indented
    except (TypeError, ValueError, KeyError) as exp:
        funcname = f.__name__
        funcname = docstring.split("\n")[0] if funcname is None else funcname
        raise RuntimeError(f"Error documenting {funcname}:\n{exp!s}")
    return f
