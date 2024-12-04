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

# Standard documentation entries
#
# Entries are listed in alphabetical order.
#
docdict = {}

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

# cmap
docdict["cmap"] = """
cmap : :class:`matplotlib.colors.Colormap`, or :obj:`str`, optional
    The colormap to use.
    Either a string which is a name of a matplotlib colormap,
    or a matplotlib colormap object.
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
confounds : CSV file or array-like, optional
    This parameter is passed to :func:`nilearn.signal.clean`.
    Please see the related documentation for details.
    shape: list of (number of scans, number of confounds)
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
docdict["data_dir"] = """
data_dir : :obj:`pathlib.Path` or :obj:`str`, optional
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

# hemi
docdict["hemi"] = """
hemi : {"left", "right"}, default="left"
    Hemisphere to display.
"""

# high_pass
docdict["high_pass"] = """
high_pass : :obj:`float`, default=None
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

# legacy_format
docdict["legacy_format"] = """
legacy_format : :obj:`bool`, default=True
    If set to `True`, the fetcher will return recarrays.
    Otherwise, it will return pandas dataframes.
"""

# linewidth
docdict["linewidths"] = """
linewidths : :obj:`float`, optional
    Set the boundary thickness of the contours.
    Only reflects when `view_type=contours`.
"""

# low_pass
docdict["low_pass"] = """
low_pass : :obj:`float` or None, default=None
    Low cutoff frequency in Hertz.
    If specified, signals above this frequency will be filtered out.
    If `None`, no low-pass filtering will be performed.
"""

# lower_cutoff
docdict["lower_cutoff"] = """
lower_cutoff : :obj:`float`, optional
    Lower fraction of the histogram to be discarded.
"""

# mask_strategy
docdict["mask_strategy"] = """
mask_strategy : {"background", "epi", "whole-brain-template",\
"gm-template", "wm-template"}, optional
    The strategy used to compute the mask:

    - ``"background"``: Use this option if your images present
      a clear homogeneous background.

    - ``"epi"``: Use this option if your images are raw EPI images

    - ``"whole-brain-template"``: This will extract the whole-brain
      part of your data by resampling the MNI152 brain mask for
      your data's field of view.

      .. note::

          This option is equivalent to the previous 'template' option
          which is now deprecated.

    - ``"gm-template"``: This will extract the gray matter part of your
      data by resampling the corresponding MNI152 template for your
      data's field of view.

      .. versionadded:: 0.8.1

    - ``"wm-template"``: This will extract the white matter part of your
      data by resampling the corresponding MNI152 template for your
      data's field of view.

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

# kwargs for Maskers
docdict["masker_kwargs"] = """
kwargs : dict
    Keyword arguments to be passed to functions called within the masker.
    Kwargs prefixed with `'clean__'` will be passed to
    :func:`~nilearn.signal.clean`.
    Within :func:`~nilearn.signal.clean`, kwargs prefixed with
    `'butterworth__'` will be passed to the Butterworth filter
    (i.e., `clean__butterworth__`).
"""

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
output_file : :obj:`str`, or None, optional
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
random_state : :obj:`int` or RandomState, optional
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

# resume
docdict["resume"] = """
resume : :obj:`bool`, default=True
    Whether to resume download of a partly-downloaded file.
"""

# sample_mask
docdict["sample_mask"] = """
sample_mask : Any type compatible with numpy-array indexing, optional
    shape: (number of scans - number of volumes removed, )
    Masks the niimgs along time/fourth dimension to perform scrubbing
    (remove volumes with high motion) and/or non-steady-state volumes.
    This parameter is passed to :func:`nilearn.signal.clean`.
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

# smoothing_fwhm
docdict["smoothing_fwhm"] = """
smoothing_fwhm : :obj:`float`, optional.
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
t_r : :obj:`float` or None, default=None
    :term:`Repetition time<TR>`, in seconds (sampling period).
    Set to `None` if not provided.
"""

# target_affine
docdict["target_affine"] = """
target_affine : :class:`numpy.ndarray`, default=None
    If specified, the image is resampled corresponding to this new affine.
    `target_affine` can be a 3x3 or a 4x4 matrix.
"""

# target_shape
docdict["target_shape"] = """
target_shape : :obj:`tuple` or :obj:`list`, default=None
    If specified, the image will be resized to match this new shape.
    `len(target_shape)` must be equal to 3.

    .. note::

        If `target_shape` is specified, a `target_affine` of shape
        `(4, 4)` must also be given.

"""

# templateflow
docdict["templateflow"] = """
    The default template of :term:`fMRIPrep` is the asymmetrical ICBM152 2009,
    release c (MNI152NLin2009cSAsym).
    The NiLearn template is asymmetrical ICBM152 2009, release a.
    If you wish to use the exact same release as :term:`fMRIPrep`,
    please refer to TemplateFlow (https://www.templateflow.org/).
"""

# threshold
docdict["threshold"] = """
threshold : a number, None, or 'auto', optional
    If `None` is given, the image is not thresholded.
    If a number is given, it is used to threshold the image:
    values below the threshold (in absolute value) are plotted as transparent.
    If "auto" is given, the threshold is determined magically
    by analysis of the image.
"""

# title
docdict["title"] = """
title : :obj:`str`, or None, default=None
    The title displayed on the figure.
"""

# upper_cutoff
docdict["upper_cutoff"] = """
upper_cutoff : :obj:`float`, optional
    Upper fraction of the histogram to be discarded.
"""

# url
docdict["url"] = """
url : :obj:`str`, default=None
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

# view
docdict["view"] = """
view : :obj:`str`, or a pair of :obj:`float` or :obj:`int`, default="lateral"
    If a string, must be in
    {"lateral", "medial", "dorsal", "ventral", "anterior", "posterior"}.
    If a sequence, must be a pair (elev, azim) of :obj:`float` or :obj:`int`
    angles in degrees that will manually set a custom view.
    E.g., view=[270.0, 90] or view=(0, -180.0).
    View of the surface that is rendered.
"""

# vmax
docdict["vmax"] = """
vmax : :obj:`float`, optional
    Upper bound of the colormap.
    If `None`, the max of the image is used.
    Passed to :func:`matplotlib.pyplot.imshow`.
"""

# vmin
docdict["vmin"] = """
vmin : :obj:`float`, optional
    Lower bound of the colormap.
    If `None`, the min of the image is used.
    Passed to :func:`matplotlib.pyplot.imshow`.
"""

##############################################################################

docdict_indented = {}


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
