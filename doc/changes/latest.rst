.. currentmodule:: nilearn

.. include:: names.rst

0.14.0.dev
==========

..
    Each changelog entry should begin with one of the following badges:
    - :bdg-primary:`Doc`
    - :bdg-secondary:`Maint`
    - :bdg-success:`API`
    - :bdg-info:`Plotting`
    - :bdg-warning:`Test`
    - :bdg-danger:`Deprecation`
    - :bdg-dark:`Code`

HIGHLIGHTS
----------

- Nilearn can leverage scikit-learn's Array API-supported estimators to speed up neuroimaging ML analyses using GPU acceleration. See :ref:`user guide page <gpu_usage>`.

.. warning::

 | **Minimum supported versions of the following packages have been bumped up:**
 | - numpy -- 1.26.0
 | - pandas -- 2.3.0
 | - scikit-learn -- 1.5.0
 | - scipy -- 1.11.1
 | - kaleido -- 1.1.0
 | - plotly -- 6.1.1
 | - request -- 2.33.0
 | - jinja2 -- 3.1.6
 |

.. admonition:: Google Chrome needed to save images with plotly / kaleido

    To be able to save images with plotly,
    make sure that Google Chrome is installed!
    You can install a compatible Chrome version using
    the ``kaleido_get_chrome`` command in command line or
    ``kaleido.get_chrome_sync()`` function
    in Python:

    .. code-block:: python

       import kaleido

       kaleido.get_chrome_sync()


NEW
---

Fixes
-----

- :bdg-success:`API` Fix bug when ``confounds_strategy=None`` in :func:`~glm.first_level.first_level_from_bids` (:gh:`6247` by `Pierre-Louis Barbarant`_).

- :bdg-dark:`Code` Ensure that maskers ``inverse_transform`` can accept pandas or polars dataframe as input (:gh:`6175` by `Rémi Gau`_).

- :bdg-secondary:`Maint` Allow local installation with ``uv sync`` (:gh:`6024` by `Mathieu Dugré`_)

- :bdg-secondary:`Maint` Add return type annotations and :obj:`~typing.overload` signatures to :func:`~connectome.vec_to_sym_matrix`, :func:`~connectome.group_sparse_covariance`, and :func:`~reporting.get_clusters_table` (:gh:`6368` by `Rémi Gau`_).

- :bdg-primary:`Doc` Rewrite :class:`~nilearn.decoding.Decoder` :ref:`example <sphx_glr_auto_examples_02_decoding_plot_haxby_grid_search.py>` with incorrect nested cross-validation implementation (:gh:`6059` by `Michelle Wang`_).

- :bdg-info:`Plotting` Fix ``nilearn.plotting.view_img`` resampling of non-isotropic images when no background image is used (:gh:`6031` by `Michelle Wang`_).

- :bdg-dark:`Code` Disallow boolean dtype for estimators. (:gh:`6271` by `Rémi Gau`_).

- :bdg-dark:`Code` Better checks of dtype for :class:`~surface.SurfaceImage` and :class:`~nilearn.surface.PolyData`. (:gh:`6271` by `Rémi Gau`_).

- :bdg-dark:`Code` Ensure that estimators that accept images can work will several datatypes as input and that their methods can output arrays or images of the requested datatype (:gh:`5511` by `Rémi Gau`_).

- :bdg-info:`Plotting` Fix bug introduced in ``0.13.0`` while plotting single valued data with :func:`~nilearn.plotting.plot_roi`.  (:gh:`6122` by `Hande Gözükan`_).

- :bdg-dark:`Code` Keep values equal to threshold while thresholding image with :func:`~nilearn.image.threshold_img`.  (:gh:`6130` by `Hande Gözükan`_).

- :bdg-dark:`Code` Do not copy mask image's header in ``masking.unmask`` (:gh:`6157` by `Taylor Salo`_).

- :bdg-dark:`Code` Make sure that area data of Freesurfer can be loaded as :class:`~surface.SurfaceImage` by :func:`~nilearn.datasets.load_fsaverage_data` (:gh:`6230` by `Rémi Gau`_).

- :bdg-dark:`Code` Fix order of vertices and data for ``fsaverage3`` and ``fsaverage4`` datasets and warn user (:gh:`6127` by `Rémi Gau`_ and `Hande Gözükan`_).

- :bdg-dark:`Code` Fix several issues in plotting, maskers and decomposition when handling map (or label) images with 0 or just a single map (or label) (:gh:`5908` by `Rémi Gau`_).

- :bdg-dark:`Code` Raise error when maps (or labels) images contain no map (or label) after resampling by :class:`~nilearn.maskers.NiftiMapsMasker` (or :class:`~nilearn.maskers.NiftiLabelsMasker`) (:gh:`6240` by `Rémi Gau`_).

- :bdg-dark:`Code` Fix :class:`~nilearn.maskers.SurfaceMapsMasker` to use ``allow_overlap`` parameter (:gh:`6211` by `Rémi Gau`_ and `Hande Gözükan`_).

- :bdg-dark:`Code` Fix resizing of reports when viewed in browser (:gh:`6234` by `Hande Gözükan`_).

- :bdg-dark:`Code` Fix missing subjects in ADHD dataset phenotypic data. (:gh:`6250` by `Hande Gözükan`_).

- :bdg-info:`Plotting` Fix colors while plotting ROI contours with ``nilearn.plotting.plot_roi`` (:gh:`6245` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Fix several typos and grammatical errors in the :ref:`introduction tutorial to fMRI decoding <sphx_glr_auto_examples_00_tutorials_plot_decoding_tutorial.py>` (:gh:`6269` by `Kosei Tanno`_).

Enhancements
------------

- :bdg-primary:`Doc`added jupyterlite option in the documentation (:gh:`6033` by `Hugo Delhaye`_).

- :bdg-success:`API` Allow the ``mask_img`` parameter to :func:`~glm.first_level.first_level_from_bids` to take the value ``"derivatives"`` to generate a mask from the masks in BIDS derivative to use in the model (:gh:`5981` by `Rémi Gau`_).

- :bdg-success:`API` The parameter ``estimator_args`` was added to all decoding estimators to allow to pass parameters directly to the underlying Scikit-Learn estimators (:gh:`5641` by `Rémi Gau`_).

- :bdg-success:`API` :class:`~nilearn.decoding.SearchLight` now has a ``random_state`` and ``estimator_args`` to pass to the underlying estimator (:gh:`6020` by `Rémi Gau`_).

- :bdg-success:`API` Extend the list of estimators supported by :class:`~nilearn.decoding.SearchLight` (:gh:`6215` by `Rémi Gau`_).

- :bdg-dark:`Code` Implement ``set_output`` for multi maskers to extract data to pandas or polars dataframe (:gh:`6175` by `Rémi Gau`_).

- :bdg-info:`Plotting` Allow string threshold in ``nilearn.plotting.plot_*`` functions (:gh:`5982` by `Saeed Babadi`_).

- :bdg-success:`Doc` Add an example to the plot_carpet function (:gh:`6065` by `Johanna Bayer`_).

- :bdg-dark:`Code` Add surface support to :func:`~nilearn.image.smooth_img` (:gh:`3267` by `Jason D. Yeatman`_ and `Noah C. Benson`_ ).

- :bdg-success:`API` Add a public method to access the fitted mask of GLM instances (:gh:`5981` by `Rémi Gau`_).

- :bdg-dark:`Code` Added ``screening_n_features`` parameter to :class:`~nilearn.decoding.Decoder`,  :class:`~nilearn.decoding.DecoderRegressor`, :class:`~nilearn.decoding.FREMClassifier`,  and :class:`~nilearn.decoding.FREMRegressor`.

- :bdg-success:`API` Support pathlike objects for ``cmap`` argument in :func:`~plotting.plot_surf_roi` (:gh:`5981` by `Joseph Paillard`_).

- :bdg-info:`Plotting` Add example showing how to use TemplateFlow to make surface plots and change volume template (:gh:`5968` by `Joseph Paillard`_ and `Rémi Gau`_).

- :bdg-primary:`Doc` Added a :ref:`user guide page <gpu_usage>` to demonstrate speedup using GPU (:gh:`5958` by `Himanshu Aggarwal`_ and `Elizabeth DuPre`_).

- :bdg-secondary:`Maint` Add list of badges to changelog template on :ref:`maintenance_process` page (:gh:`6084` by `Michelle Wang`_).

- :bdg-info:`Plotting`  Add opacity slider on :func:`~plotting.view_img` (:gh:`6107` by `Rémi Gau`_).

- :bdg-info:`Plotting`  Add support for `niivue <https://niivue.com/>`_ as a backend engine for :func:`~plotting.view_surf` (:gh:`3729` by  `Alexis Thual`_, `Himanshu Aggarwal`_, `Rémi Gau`_, `Pierre-Louis Barbarant`_, `Chris Rorden`_, `Taylor Hanayik`_, `Hande Gözükan`_).

- :bdg-info:`Plotting`  Enable using brainsprite as a plotting engine to create interactive visualization in the reports generated by the :class:`~maskers.NiftiMasker` and the :class:`~maskers.NiftiLabelsMasker` (:gh:`6037` by `Rémi Gau`_).

- :bdg-info:`Plotting` Add ``node_labels`` parameter to :func:`~plotting.view_connectome` (:gh:`6153` by `Rishika Kapil`_).

- :bdg-primary:`Doc` Unify estimator documentation to explicitly accept scikit-learn compatible objects and add cross-links to the developer guide (:gh:`6181` by `Xichun Xu`_).

- :bdg-primary:`Doc` Add API documentation for :class:`~nilearn.connectome.ConnectivityMeasure` to elaborate the requirement for customized ``cov_estimator`` and extra test to ensure the user input is a valid sklearn estimator (:gh:`6074` by `Hao-Ting Wang`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.image.crop_img` function (:gh:`6316` by `Thibault de Varax`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for nine utility functions in the public API: :func:`~nilearn.connectome.cov_to_corr`, :func:`~nilearn.connectome.prec_to_partial`, :func:`~nilearn.glm.expression_to_contrast_vector`, :func:`~nilearn.glm.fdr_threshold`, :func:`~nilearn.glm.first_level.glover_dispersion_derivative`, :func:`~nilearn.glm.first_level.glover_hrf`, :func:`~nilearn.glm.first_level.glover_time_derivative`, :func:`~nilearn.glm.first_level.mean_scaling` and :func:`~nilearn.interfaces.bids.parse_bids_filename` (:gh:`6259` and :gh:`6262` by `Laura Piñero Roig`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for one function in the public API: :func:`~nilearn.masking.compute_background_mask` (:gh:`6303` by `Marco Flores`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for one function in the public API: :func:`~nilearn.image.largest_connected_component_img` (:gh:`6283` by `Bastien Cagna`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for a utility function in the public API: :func:`~nilearn.image.threshold_img` (:gh:`6280` by `Fernanda Ponce`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for a utility function in the public API: :func:`~nilearn.image.clean_img` (:gh:`6312` by `Nirmitee Mulay`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for a utility function in the public API: :func:`~nilearn.masking.intersect_masks` (:gh:`6318` by `Marco Bedini`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for a utility function in the public API: :func:`~nilearn.interfaces.bids.get_bids_files` (:gh:`6317` by `Gabriele Amorosino`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.image.smooth_img` function (:gh:`6302` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for a utility function in the public API: :func:`~nilearn.image.copy_img` (:gh:`6299` by `Ruben Dörfel`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.image.smooth_img` function (:gh:`6302` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.glm.threshold_stats_img` function (:gh:`6297` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for :func:`~nilearn.image.check_niimg`, :func:`~nilearn.image.check_niimg_3d` and	:func:`~nilearn.image.check_niimg_4d` functions (:gh:`6286` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring section for  :func:`~nilearn.regions.signals_to_img_labels` function (:gh:`6306` by `Hande Gözükan`_).

- :bdg-primary:`Doc` Add ``Examples`` docstring sections for one function in the public API: :func:`~nilearn.glm.first_level.make_first_level_design_matrix` (:gh:`6320` by `Marco Flores`_).

Changes
-------

- :bdg-dark:`Code` Add return type annotations to public functions in :mod:`nilearn.signal` and :mod:`nilearn.surface` (:gh:`6353` by `Rémi Gau`_).

- :bdg-dark:`Code` Add return type annotations to public functions in :mod:`nilearn.masking` (:gh:`6345` by `Rémi Gau`_).

- :bdg-dark:`Code` Add return type annotations to public functions in :mod:`nilearn.image` (:gh:`6338` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The function ``nilearn.datasets.utils.load_sample_motor_activation_image`` and ``nilearn.datasets.fetch_neurovault_motor_task`` were removed. Use :func:`~datasets.load_sample_motor_activation_image` instead (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The private functions ``nilearn._utils.niimg_conversions.check_niimg*`` have been removed, please use their public equivalent :func:`~image.check_niimg`, :func:`~image.check_niimg_3d` and :func:`~image.check_niimg_4d` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` Accessing the maskers from ``nilearn.input_data`` is no longer possible, they now must be accessed via :mod:`nilearn.maskers` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The ``version`` parameters of in :func:`~datasets.fetch_atlas_pauli_2017` has now permanently been replaced by ``atlas_type`` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` ``plot_img_comparison`` is no longer accessible from ``nilearn.plotting.image.img_plotting``, access it from ``nilearn.plotting`` or from ``nilearn.plotting.img_comparison`` (:gh:`5995` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The attributes ``residuals``, ``predicted`` and ``r_square`` are deprecated and will be removed in version >=0.16.0. "Use ``residuals_``, ``predicted_`` and ``r_square_`` instead." (:gh:`6236` by `Rémi Gau`_).

- :bdg-danger:`Deprecation` The ``"z_score"`` value for the ``standardize`` parameter is no longer supported. Use ``standardize="z_score_sample"`` instead (:gh:`5995` by `Rémi Gau`_).

- :bdg-dark:`Code` Remove aggressive garbage collection in safe_get_data for performance, mainly in CI. (:gh:`6039` by `Basile Pinsard`_).

- :bdg-secondary:`Maint` Refactor and speed up tests for :func:`~plotting.view_img` (:gh:`6072` by `Michelle Wang`_)

- :bdg-secondary:`Maint` Several optional dependencies (``dev``, ``test``, ``doc``...) have been swapped in favor of `dependency groups <https://packaging.python.org/en/latest/specifications/dependency-groups>`_, so they will no longer work when installing Nilearn from pypi. When installing locally, instead of ``pip install '.[dev]'`` do ``pip install . --group dev`` (:gh:`6108` by `Rémi Gau`_).

- :bdg-secondary:`Maint` The optional dependency ``plotly`` has been removed in favor of ``plotting``, so instead of ``pip install 'nilearn[plotly]'`` do ``pip install 'nilearn[plotting]'`` (:gh:`6108` by `Rémi Gau`_).

- :bdg-secondary:`Maint` update fsaverage3 and fsaverage3 datasets on the open-science framework (OSF) with correctly ordered vertices (see :gh:`6127`), add Talairach and AAL atlas to OSF as fall back url in case the primary one fails (:gh:`6246` by `Rémi Gau`_).

- :bdg-dark:`Code` Ensure input images of :class:`~.glm.first_level.FirstLevelModel` and :class:`~.glm.second_level.SecondLevelModel` have same field of view (for nifti) or compatible meshes (for surfaces) (:gh:`6254` by `Rémi Gau`_).

- :bdg-dark:`Code` Ensure implicit mask of :class:`~.glm.first_level.FirstLevelModel` is computed on data from all runs (:gh:`6254` by `Rémi Gau`_).

- :bdg-dark:`Code` Ensure input of :class:`~.glm.second_level.SecondLevelModel` cannot be a 5D volume image or list of 2D surface images (:gh:`6254` by `Rémi Gau`_).

- :bdg-dark:`Code` For some input of :class:`~.glm.second_level.SecondLevelModel` check at fit time that number of images match number of row in design matrix (:gh:`6254` by `Rémi Gau`_).
