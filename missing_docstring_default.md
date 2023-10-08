# nilearn/masking.py
nilearn/masking.py::intersect_masks - **line 123**
<br>Potential arguments to fix
- [ ] `threshold` `Default=0.5`

nilearn/masking.py::compute_epi_mask - **line 210**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`
- [ ] `ensure_finite` `Default=True`
- [ ] `exclude_zeros` `Default=False`

nilearn/masking.py::compute_multi_epi_mask - **line 328**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`
- [ ] `exclude_zeros` `Default=False`
- [ ] `threshold` `Default=0.5`

nilearn/masking.py::compute_background_mask - **line 422**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`

nilearn/masking.py::compute_multi_background_mask - **line 504**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`
- [ ] `exclude_zeros` `Default=False`
- [ ] `threshold` `Default=0.5`
- [ ] `opening` `Default=2`
- [ ] `upper_cutoff` `Default=0.85`

nilearn/masking.py::compute_brain_mask - **line 586**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`
- [ ] `threshold` `Default=0.5`

nilearn/masking.py::compute_multi_brain_mask - **line 663**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`
- [ ] `threshold` `Default=0.5`

nilearn/masking.py::apply_mask - **line 751**
<br>Potential arguments to fix
- [ ] `ensure_finite` `Default=True`
- [ ] `dtype` `Default='f'`

nilearn/masking.py::unmask - **line 919**
<br>Potential arguments to fix
- [ ] `order` `Default='F'`

# nilearn/signal.py
nilearn/signal.py::butterworth - **line 313**
<br>Potential arguments to fix
- [ ] `copy` `Default=False`
- [ ] `padlen` `Default=None`
- [ ] `padtype` `Default='odd'`
- [ ] `order` `Default=5`

nilearn/signal.py::high_variance_confounds - **line 454**
<br>Potential arguments to fix
- [ ] `percentile` `Default=2.0`
- [ ] `n_confounds` `Default=5`

nilearn/signal.py::clean - **line 531**
<br>Potential arguments to fix
- [ ] `ensure_finite` `Default=False`
- [ ] `filter` `Default='butterworth'`
- [ ] `confounds` `Default=None`
- [ ] `sample_mask` `Default=None`
- [ ] `standardize` `Default='zscore'`
- [ ] `runs` `Default=None`

# nilearn/reporting/glm_reporter.py
nilearn/reporting/glm_reporter.py::make_glm_report - **line 49**
<br>Potential arguments to fix
- [ ] `report_dims` `Default=(1600, 800)`
- [ ] `display_mode` `Default=None`
- [ ] `plot_type` `Default='slice'`
- [ ] `min_distance` `Default=8.0`
- [ ] `two_sided` `Default=False`
- [ ] `height_control` `Default='fpr'`
- [ ] `cluster_threshold` `Default=0`
- [ ] `alpha` `Default=0.001`
- [ ] `threshold` `Default=3.09`
- [ ] `bg_img` `Default='MNI152TEMPLATE'`
- [ ] `title` `Default=None`

# nilearn/reporting/_get_clusters_table.py
nilearn/reporting/_get_clusters_table.py::get_clusters_table - **line 209**
<br>Potential arguments to fix
- [ ] `return_label_maps` `Default=False`
- [ ] `min_distance` `Default=8.0`
- [ ] `two_sided` `Default=False`
- [ ] `cluster_threshold` `Default=None`

# nilearn/reporting/_visual_testing/_glm_reporter_visual_inspection_suite_.py
nilearn/reporting/_visual_testing/_glm_reporter_visual_inspection_suite_.py::report_flm_adhd_dmn - **line: 34**
- [ ] No docstring detected.

nilearn/reporting/_visual_testing/_glm_reporter_visual_inspection_suite_.py::report_flm_bids_features - **line: 166**
- [ ] No docstring detected.

nilearn/reporting/_visual_testing/_glm_reporter_visual_inspection_suite_.py::report_flm_fiac - **line: 190**
- [ ] No docstring detected.

nilearn/reporting/_visual_testing/_glm_reporter_visual_inspection_suite_.py::report_slm_oasis - **line: 246**
- [ ] No docstring detected.

nilearn/reporting/_visual_testing/_glm_reporter_visual_inspection_suite_.py::prefer_parallel_execution - **line: 276**
- [ ] No docstring detected.

nilearn/reporting/_visual_testing/_glm_reporter_visual_inspection_suite_.py::run_function - **line: 300**
- [ ] No docstring detected.

nilearn/reporting/_visual_testing/_glm_reporter_visual_inspection_suite_.py::prefer_serial_execution - **line: 304**
- [ ] No docstring detected.

# nilearn/mass_univariate/permuted_least_squares.py
nilearn/mass_univariate/permuted_least_squares.py::permuted_ols - **line 298**
<br>Potential arguments to fix
- [ ] `output_type` `Default='legacy'`
- [ ] `threshold` `Default=None`
- [ ] `tfce` `Default=False`
- [ ] `masker` `Default=None`
- [ ] `verbose` `Default=0`
- [ ] `n_jobs` `Default=1`
- [ ] `random_state` `Default=None`
- [ ] `two_sided_test` `Default=True`
- [ ] `n_perm` `Default=10000`
- [ ] `model_intercept` `Default=True`
- [ ] `confounding_vars` `Default=None`

# nilearn/mass_univariate/tests/utils.py
nilearn/mass_univariate/tests/utils.py::get_tvalue_with_alternative_library - **line 5**
<br>Potential arguments to fix
- [ ] `covars` `Default=None`

# nilearn/datasets/atlas.py
nilearn/datasets/atlas.py::fetch_atlas_difumo - **line 30**
<br>Potential arguments to fix
- [ ] `resolution_mm` `Default=2`
- [ ] `dimension` `Default=64`

nilearn/datasets/atlas.py::fetch_atlas_craddock_2012 - **line 157**
<br>Potential arguments to fix
- [ ] `grp_mean` `Default=True`
- [ ] `homogeneity` `Default=None`

nilearn/datasets/atlas.py::fetch_atlas_destrieux_2009 - **line 278**
<br>Potential arguments to fix
- [ ] `lateralized` `Default=True`

nilearn/datasets/atlas.py::fetch_atlas_harvard_oxford - **line 360**
<br>Potential arguments to fix
- [ ] `symmetric_split` `Default=False`

nilearn/datasets/atlas.py::fetch_atlas_juelich - **line 517**
<br>Potential arguments to fix
- [ ] `symmetric_split` `Default=False`

nilearn/datasets/atlas.py::fetch_atlas_smith_2009 - **line 930**
<br>Potential arguments to fix
- [ ] `resting` `Default=True`
- [ ] `dimension` `Default=None`
- [ ] `mirror` `Default='origin'`

nilearn/datasets/atlas.py::fetch_atlas_aal - **line 1174**
<br>Potential arguments to fix
- [ ] `version` `Default='SPM12'`

nilearn/datasets/atlas.py::fetch_atlas_basc_multiscale_2015 - **line 1315**
<br>Potential arguments to fix
- [ ] `version` `Default='sym'`
- [ ] `resolution` `Default=None`

nilearn/datasets/atlas.py::fetch_coords_dosenbach_2010 - **line 1468**
<br>Potential arguments to fix
- [ ] `ordered_regions` `Default=True`

nilearn/datasets/atlas.py::fetch_coords_seitzman_2018 - **line 1533**
<br>Potential arguments to fix
- [ ] `ordered_regions` `Default=True`

nilearn/datasets/atlas.py::fetch_atlas_pauli_2017 - **line 1922**
<br>Potential arguments to fix
- [ ] `version` `Default='prob'`

nilearn/datasets/atlas.py::fetch_atlas_schaefer_2018 - **line 2012**
<br>Potential arguments to fix
- [ ] `base_url` `Default=None`
- [ ] `resolution_mm` `Default=1`
- [ ] `yeo_networks` `Default=7`
- [ ] `n_rois` `Default=400`

# nilearn/datasets/struct.py
nilearn/datasets/struct.py::load_mni152_template - **line 173**
<br>Potential arguments to fix
- [ ] `resolution` `Default=None`

nilearn/datasets/struct.py::load_mni152_gm_template - **line 233**
<br>Potential arguments to fix
- [ ] `resolution` `Default=None`

nilearn/datasets/struct.py::load_mni152_wm_template - **line 283**
<br>Potential arguments to fix
- [ ] `resolution` `Default=None`

nilearn/datasets/struct.py::load_mni152_brain_mask - **line 333**
<br>Potential arguments to fix
- [ ] `threshold` `Default=0.2`
- [ ] `resolution` `Default=None`

nilearn/datasets/struct.py::load_mni152_gm_mask - **line 377**
<br>Potential arguments to fix
- [ ] `n_iter` `Default=2`
- [ ] `threshold` `Default=0.2`
- [ ] `resolution` `Default=None`

nilearn/datasets/struct.py::load_mni152_wm_mask - **line 428**
<br>Potential arguments to fix
- [ ] `n_iter` `Default=2`
- [ ] `threshold` `Default=0.2`
- [ ] `resolution` `Default=None`

nilearn/datasets/struct.py::fetch_icbm152_brain_gm_mask - **line 481**
<br>Potential arguments to fix
- [ ] `n_iter` `Default=2`
- [ ] `threshold` `Default=0.2`

nilearn/datasets/struct.py::fetch_oasis_vbm - **line 595**
<br>Potential arguments to fix
- [ ] `dartel_version` `Default=True`
- [ ] `n_subjects` `Default=None`

nilearn/datasets/struct.py::fetch_surf_fsaverage - **line 863**
<br>Potential arguments to fix
- [ ] `mesh` `Default='fsaverage5'`

# nilearn/datasets/neurovault.py
nilearn/datasets/neurovault.py::neurosynth_words_vectorized - **line 1179**
<br>Potential arguments to fix
- [ ] `verbose` `Default=3`

nilearn/datasets/neurovault.py::fetch_neurovault - **line 2444**
<br>Potential arguments to fix
- [ ] `verbose` `Default=3`
- [ ] `vectorize_words` `Default=True`
- [ ] `resample` `Default=False`
- [ ] `fetch_neurosynth_words` `Default=False`
- [ ] `data_dir` `Default=None`
- [ ] `mode` `Default='download_new'`
- [ ] `image_filter` `Default=_empty_filter`
- [ ] `image_terms` `Default=basic_image_terms()`
- [ ] `collection_filter` `Default=_empty_filter`
- [ ] `collection_terms` `Default=basic_collection_terms()`
- [ ] `max_images` `Default=_DEFAULT_MAX_IMAGES`

nilearn/datasets/neurovault.py::fetch_neurovault_ids - **line 2675**
<br>Potential arguments to fix
- [ ] `verbose` `Default=3`
- [ ] `vectorize_words` `Default=True`
- [ ] `resample` `Default=False`
- [ ] `fetch_neurosynth_words` `Default=False`
- [ ] `data_dir` `Default=None`
- [ ] `mode` `Default='download_new'`
- [ ] `image_ids` `Default=()`
- [ ] `collection_ids` `Default=()`

nilearn/datasets/neurovault.py::fetch_neurovault_motor_task - **line 2788**
<br>Potential arguments to fix
- [ ] `verbose` `Default=1`
- [ ] `data_dir` `Default=None`

nilearn/datasets/neurovault.py::fetch_neurovault_auditory_computation_task - **line 2829**
<br>Potential arguments to fix
- [ ] `verbose` `Default=1`
- [ ] `data_dir` `Default=None`

# nilearn/datasets/func.py
nilearn/datasets/func.py::fetch_haxby - **line 48**
<br>Potential arguments to fix
- [ ] `fetch_stimuli` `Default=False`
- [ ] `subjects` `Default=(2,)`

nilearn/datasets/func.py::fetch_adhd - **line 267**
<br>Potential arguments to fix
- [ ] `n_subjects` `Default=30`

nilearn/datasets/func.py::fetch_localizer_contrasts - **line 534**
<br>Potential arguments to fix
- [ ] `get_anats` `Default=False`
- [ ] `get_masks` `Default=False`
- [ ] `get_tmaps` `Default=False`
- [ ] `n_subjects` `Default=None`

nilearn/datasets/func.py::fetch_localizer_calculation_task - **line 908**
<br>Potential arguments to fix
- [ ] `n_subjects` `Default=1`

nilearn/datasets/func.py::fetch_abide_pcp - **line 1009**
<br>Potential arguments to fix
- [ ] `quality_checked` `Default=True`
- [ ] `derivatives` `Default=['func_preproc']`
- [ ] `global_signal_regression` `Default=False`
- [ ] `band_pass_filtering` `Default=False`
- [ ] `pipeline` `Default='cpac'`
- [ ] `n_subjects` `Default=None`

nilearn/datasets/func.py::fetch_mixed_gambles - **line 1270**
<br>Potential arguments to fix
- [ ] `return_raw_data` `Default=False`
- [ ] `n_subjects` `Default=1`

nilearn/datasets/func.py::fetch_megatrawls_netmats - **line 1349**
<br>Potential arguments to fix
- [ ] `matrices` `Default='partial_correlation'`
- [ ] `dimensionality` `Default=100`

nilearn/datasets/func.py::fetch_surf_nki_enhanced - **line 1592**
<br>Potential arguments to fix
- [ ] `n_subjects` `Default=10`

nilearn/datasets/func.py::fetch_development_fmri - **line 1878**
<br>Potential arguments to fix
- [ ] `age_group` `Default='both'`
- [ ] `reduce_confounds` `Default=True`
- [ ] `n_subjects` `Default=None`

nilearn/datasets/func.py::fetch_openneuro_dataset_index - **line 2178**
<br>Potential arguments to fix
- [ ] `verbose` `Default=1`
- [ ] `dataset_version` `Default='ds000030_R1.0.4'`

nilearn/datasets/func.py::select_from_index - **line 2290**
<br>Potential arguments to fix
- [ ] `n_subjects` `Default=None`
- [ ] `exclusion_filters` `Default=None`
- [ ] `inclusion_filters` `Default=None`

nilearn/datasets/func.py::fetch_openneuro_dataset - **line 2395**
<br>Potential arguments to fix
- [ ] `dataset_version` `Default='ds000030_R1.0.4'`
- [ ] `urls` `Default=None`

nilearn/datasets/func.py::fetch_spm_auditory - **line 2695**
<br>Potential arguments to fix
- [ ] `subject_id` `Default='sub001'`
- [ ] `data_name` `Default='spm_auditory'`

nilearn/datasets/func.py::fetch_spm_multimodal_fmri - **line 2881**
<br>Potential arguments to fix
- [ ] `subject_id` `Default='sub001'`
- [ ] `data_name` `Default='spm_multimodal_fmri'`

# nilearn/datasets/_testing.py
nilearn/datasets/_testing.py::dict_to_archive - **line 375**
<br>Potential arguments to fix
- [ ] `archive_format` `Default='gztar'`

nilearn/datasets/_testing.py::list_to_archive - **line 425**
<br>Potential arguments to fix
- [ ] `content` `Default=''`
- [ ] `archive_format` `Default='gztar'`

# nilearn/image/resampling.py
nilearn/image/resampling.py::resample_img - **line 332**
<br>Potential arguments to fix
- [ ] `force_resample` `Default=False`
- [ ] `fill_value` `Default=0`
- [ ] `clip` `Default=True`
- [ ] `order` `Default='F'`
- [ ] `copy` `Default=True`
- [ ] `interpolation` `Default='continuous'`
- [ ] `target_shape` `Default=None`
- [ ] `target_affine` `Default=None`

nilearn/image/resampling.py::resample_to_img - **line 683**
<br>Potential arguments to fix
- [ ] `force_resample` `Default=False`
- [ ] `fill_value` `Default=0`
- [ ] `clip` `Default=False`
- [ ] `order` `Default='F'`
- [ ] `copy` `Default=True`
- [ ] `interpolation` `Default='continuous'`

nilearn/image/resampling.py::reorder_img - **line 765**
<br>Potential arguments to fix
- [ ] `resample` `Default=None`

# nilearn/image/image.py
nilearn/image/image.py::high_variance_confounds - **line 54**
<br>Potential arguments to fix
- [ ] `mask_img` `Default=None`
- [ ] `detrend` `Default=True`
- [ ] `percentile` `Default=2.0`
- [ ] `n_confounds` `Default=5`

nilearn/image/image.py::crop_img - **line 333**
<br>Potential arguments to fix
- [ ] `return_offset` `Default=False`
- [ ] `pad` `Default=True`
- [ ] `copy` `Default=True`
- [ ] `rtol` `Default=1e-08`

nilearn/image/image.py::mean_img - **line 491**
<br>Potential arguments to fix
- [ ] `n_jobs` `Default=1`
- [ ] `verbose` `Default=0`
- [ ] `target_shape` `Default=None`
- [ ] `target_affine` `Default=None`

nilearn/image/image.py::new_img_like - **line 721**
<br>Potential arguments to fix
- [ ] `copy_header` `Default=False`
- [ ] `affine` `Default=None`

nilearn/image/image.py::threshold_img - **line 856**
<br>Potential arguments to fix
- [ ] `copy` `Default=True`
- [ ] `mask_img` `Default=None`
- [ ] `two_sided` `Default=True`
- [ ] `cluster_threshold` `Default=0`

nilearn/image/image.py::binarize_img - **line 1066**
<br>Potential arguments to fix
- [ ] `mask_img` `Default=None`
- [ ] `threshold` `Default=0`

nilearn/image/image.py::clean_img - **line 1120**
<br>Potential arguments to fix
- [ ] `mask_img` `Default=None`
- [ ] `ensure_finite` `Default=False`
- [ ] `t_r` `Default=None`
- [ ] `high_pass` `Default=None`
- [ ] `low_pass` `Default=None`
- [ ] `confounds` `Default=None`
- [ ] `standardize` `Default=True`
- [ ] `detrend` `Default=True`
- [ ] `runs` `Default=None`

nilearn/image/image.py::load_img - **line 1291**
<br>Potential arguments to fix
- [ ] `dtype` `Default=None`
- [ ] `wildcards` `Default=True`

# nilearn/surface/surface.py
nilearn/surface/surface.py::vol_to_surf - **line 467**
<br>Potential arguments to fix
- [ ] `depth` `Default=None`
- [ ] `inner_mesh` `Default=None`
- [ ] `mask_img` `Default=None`
- [ ] `n_samples` `Default=None`
- [ ] `kind` `Default='auto'`
- [ ] `interpolation` `Default='linear'`
- [ ] `radius` `Default=3.0`

# nilearn/decoding/space_net.py
nilearn/decoding/space_net.py::path_scores - **line 295**
<br>Potential arguments to fix
- [ ] `screening_percentile` `Default=20.0`
- [ ] `Xmean` `Default=None`
- [ ] `debias` `Default=False`
- [ ] `key` `Default=None`
- [ ] `eps` `Default=0.001`
- [ ] `n_alphas` `Default=10`
- [ ] `is_classif` `Default=False`

# nilearn/decoding/space_net_solvers.py
nilearn/decoding/space_net_solvers.py::tvl1_solver - **line 475**
<br>Potential arguments to fix
- [ ] `verbose` `Default=1`
- [ ] `callback` `Default=None`
- [ ] `tol` `Default=0.0001`
- [ ] `prox_max_iter` `Default=5000`
- [ ] `init` `Default=None`
- [ ] `lipschitz_constant` `Default=None`
- [ ] `max_iter` `Default=100`
- [ ] `loss` `Default=None`

# nilearn/decoding/fista.py
nilearn/decoding/fista.py::mfista - **line 65**
<br>Potential arguments to fix
- [ ] `verbose` `Default=2`
- [ ] `callback` `Default=None`
- [ ] `dgap_factor` `Default=None`
- [ ] `check_lipschitz` `Default=False`
- [ ] `tol` `Default=0.0001`
- [ ] `max_iter` `Default=1000`
- [ ] `init` `Default=None`
- [ ] `dgap_tol` `Default=None`

# nilearn/decoding/searchlight.py
nilearn/decoding/searchlight.py::search_light - **line 31**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`
- [ ] `n_jobs` `Default=-1`
- [ ] `cv` `Default=None`
- [ ] `scoring` `Default=None`
- [ ] `groups` `Default=None`

# nilearn/decoding/tests/_utils.py
nilearn/decoding/tests/_utils.py::create_graph_net_simulation_data - **line 9**
<br>Potential arguments to fix
- [ ] `smooth_X` `Default=1`
- [ ] `task` `Default='regression'`
- [ ] `random_state` `Default=42`
- [ ] `n_points` `Default=10`
- [ ] `size` `Default=8`
- [ ] `n_samples` `Default=200`
- [ ] `snr` `Default=1.0`

# nilearn/externals/conftest.py
nilearn/externals/conftest.py::pytest_ignore_collect - **line: 7**
- [ ] No docstring detected.

# nilearn/externals/tempita/__init__.py
nilearn/externals/tempita/__init__.py::get_file_template - **line: 82**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::sub - **line: 392**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::paste_script_template_renderer - **line: 398**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::html_quote - **line: 456**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::url - **line: 474**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::attr - **line: 481**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::sub_html - **line: 516**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::lex - **line: 652**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::trim_lex - **line: 741**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::parse - **line: 822**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::parse_expr - **line: 947**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::parse_cond - **line: 999**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::parse_one_cond - **line: 1014**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::parse_for - **line: 1040**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::parse_default - **line: 1074**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::parse_inherit - **line: 1096**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::parse_def - **line: 1103**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::parse_signature - **line: 1133**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::isolate_expression - **line: 1219**
- [ ] No docstring detected.

nilearn/externals/tempita/__init__.py::fill_command - **line: 1242**
- [ ] No docstring detected.

# nilearn/externals/tempita/compat3.py
nilearn/externals/tempita/compat3.py::is_unicode - **line: 39**
- [ ] No docstring detected.

nilearn/externals/tempita/compat3.py::coerce_text - **line: 46**
- [ ] No docstring detected.

# nilearn/glm/thresholding.py
nilearn/glm/thresholding.py::cluster_level_inference - **line 110**
<br>Potential arguments to fix
- [ ] `verbose` `Default=False`
- [ ] `alpha` `Default=0.05`
- [ ] `threshold` `Default=3.0`
- [ ] `mask_img` `Default=None`

nilearn/glm/thresholding.py::threshold_stats_img - **line 183**
<br>Potential arguments to fix
- [ ] `two_sided` `Default=True`
- [ ] `cluster_threshold` `Default=0`
- [ ] `height_control` `Default='fpr'`
- [ ] `threshold` `Default=3.0`
- [ ] `alpha` `Default=0.001`
- [ ] `mask_img` `Default=None`
- [ ] `stat_img` `Default=None`

# nilearn/glm/contrasts.py
nilearn/glm/contrasts.py::compute_contrast - **line 49**
<br>Potential arguments to fix
- [ ] `contrast_type` `Default=None`

nilearn/glm/contrasts.py::compute_fixed_effects - **line 406**
<br>Potential arguments to fix
- [ ] `return_z_score` `Default=False`
- [ ] `dofs` `Default=None`
- [ ] `precision_weighted` `Default=False`
- [ ] `mask` `Default=None`

# nilearn/glm/_utils.py
nilearn/glm/_utils.py::z_score - **line 153**
<br>Potential arguments to fix
- [ ] `one_minus_pvalue` `Default=None`

nilearn/glm/_utils.py::full_rank - **line 293**
<br>Potential arguments to fix
- [ ] `cmax` `Default=1000000000000000.0`

# nilearn/glm/first_level/first_level.py
nilearn/glm/first_level/first_level.py::mean_scaling - **line 53**
<br>Potential arguments to fix
- [ ] `axis` `Default=0`

nilearn/glm/first_level/first_level.py::run_glm - **line 120**
<br>Potential arguments to fix
- [ ] `random_state` `Default=None`
- [ ] `verbose` `Default=0`
- [ ] `n_jobs` `Default=1`
- [ ] `bins` `Default=100`
- [ ] `noise_model` `Default='ar1'`

nilearn/glm/first_level/first_level.py::first_level_from_bids - **line 936**
<br>Potential arguments to fix
- [ ] `derivatives_folder` `Default='derivatives'`
- [ ] `minimize_memory` `Default=True`
- [ ] `n_jobs` `Default=1`
- [ ] `verbose` `Default=0`
- [ ] `noise_model` `Default='ar1'`
- [ ] `signal_scaling` `Default=0`
- [ ] `standardize` `Default=False`
- [ ] `memory_level` `Default=1`
- [ ] `memory` `Default=Memory(None)`
- [ ] `smoothing_fwhm` `Default=None`
- [ ] `target_shape` `Default=None`
- [ ] `target_affine` `Default=None`
- [ ] `mask_img` `Default=None`
- [ ] `min_onset` `Default=-24`
- [ ] `fir_delays` `Default=[0]`
- [ ] `drift_order` `Default=1`
- [ ] `high_pass` `Default=0.01`
- [ ] `drift_model` `Default='cosine'`
- [ ] `hrf_model` `Default='glover'`
- [ ] `slice_time_ref` `Default=0.0`
- [ ] `t_r` `Default=None`
- [ ] `img_filters` `Default=None`
- [ ] `sub_labels` `Default=None`
- [ ] `space_label` `Default=None`

# nilearn/glm/first_level/design_matrix.py
nilearn/glm/first_level/design_matrix.py::make_first_level_design_matrix - **line 278**
<br>Potential arguments to fix
- [ ] `oversampling` `Default=50`
- [ ] `min_onset` `Default=-24`
- [ ] `add_reg_names` `Default=None`
- [ ] `add_regs` `Default=None`
- [ ] `fir_delays` `Default=[0]`
- [ ] `drift_order` `Default=1`
- [ ] `high_pass` `Default=0.01`
- [ ] `drift_model` `Default='cosine'`
- [ ] `events` `Default=None`

nilearn/glm/first_level/design_matrix.py::make_second_level_design_matrix - **line 450**
<br>Potential arguments to fix
- [ ] `confounds` `Default=None`

# nilearn/glm/first_level/hemodynamic_models.py
nilearn/glm/first_level/hemodynamic_models.py::spm_hrf - **line 86**
<br>Potential arguments to fix
- [ ] `onset` `Default=0.0`
- [ ] `time_length` `Default=32.0`
- [ ] `oversampling` `Default=50`

nilearn/glm/first_level/hemodynamic_models.py::glover_hrf - **line 112**
<br>Potential arguments to fix
- [ ] `onset` `Default=0.0`
- [ ] `time_length` `Default=32.0`
- [ ] `oversampling` `Default=50`

nilearn/glm/first_level/hemodynamic_models.py::spm_time_derivative - **line 185**
<br>Potential arguments to fix
- [ ] `onset` `Default=0.0`
- [ ] `time_length` `Default=32.0`
- [ ] `oversampling` `Default=50`

nilearn/glm/first_level/hemodynamic_models.py::glover_time_derivative - **line 217**
<br>Potential arguments to fix
- [ ] `onset` `Default=0.0`
- [ ] `time_length` `Default=32.0`
- [ ] `oversampling` `Default=50`

nilearn/glm/first_level/hemodynamic_models.py::spm_dispersion_derivative - **line 291**
<br>Potential arguments to fix
- [ ] `onset` `Default=0.0`
- [ ] `time_length` `Default=32.0`
- [ ] `oversampling` `Default=50`

nilearn/glm/first_level/hemodynamic_models.py::glover_dispersion_derivative - **line 321**
<br>Potential arguments to fix
- [ ] `onset` `Default=0.0`
- [ ] `time_length` `Default=32.0`
- [ ] `oversampling` `Default=50`

nilearn/glm/first_level/hemodynamic_models.py::compute_regressor - **line 661**
<br>Potential arguments to fix
- [ ] `min_onset` `Default=-24`
- [ ] `fir_delays` `Default=None`
- [ ] `oversampling` `Default=50`
- [ ] `con_id` `Default='cond'`

# nilearn/glm/tests/_utils.py
nilearn/glm/tests/_utils.py::modulated_event_paradigm - **line: 19**
- [ ] No docstring detected.

nilearn/glm/tests/_utils.py::block_paradigm - **line: 31**
- [ ] No docstring detected.

nilearn/glm/tests/_utils.py::modulated_block_paradigm - **line: 42**
- [ ] No docstring detected.

nilearn/glm/tests/_utils.py::spm_paradigm - **line: 56**
- [ ] No docstring detected.

nilearn/glm/tests/_utils.py::design_with_null_durations - **line: 67**
- [ ] No docstring detected.

nilearn/glm/tests/_utils.py::design_with_nan_durations - **line: 82**
- [ ] No docstring detected.

nilearn/glm/tests/_utils.py::design_with_nan_onsets - **line: 97**
- [ ] No docstring detected.

nilearn/glm/tests/_utils.py::design_with_negative_onsets - **line: 112**
- [ ] No docstring detected.

nilearn/glm/tests/_utils.py::design_with_negative_durations - **line: 125**
- [ ] No docstring detected.

nilearn/glm/tests/_utils.py::duplicate_events_paradigm - **line: 138**
- [ ] No docstring detected.

# nilearn/glm/second_level/second_level.py
nilearn/glm/second_level/second_level.py::non_parametric_inference - **line 731**
<br>Potential arguments to fix
- [ ] `tfce` `Default=False`
- [ ] `threshold` `Default=None`
- [ ] `verbose` `Default=0`
- [ ] `two_sided_test` `Default=False`
- [ ] `n_perm` `Default=10000`
- [ ] `model_intercept` `Default=True`
- [ ] `mask` `Default=None`
- [ ] `first_level_contrast` `Default=None`
- [ ] `design_matrix` `Default=None`
- [ ] `confounds` `Default=None`

# nilearn/_utils/extmath.py
nilearn/_utils/extmath.py::fast_abs_percentile - **line 7**
<br>Potential arguments to fix
- [ ] `percentile` `Default=80`

nilearn/_utils/extmath.py::is_spd - **line 40**
<br>Potential arguments to fix
- [ ] `verbose` `Default=1`
- [ ] `decimal` `Default=15`

# nilearn/_utils/param_validation.py
nilearn/_utils/param_validation.py::check_threshold - **line 15**
<br>Potential arguments to fix
- [ ] `name` `Default='threshold'`

nilearn/_utils/param_validation.py::check_feature_screening - **line 170**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`

# nilearn/_utils/numpy_conversions.py
nilearn/_utils/numpy_conversions.py::as_ndarray - **line 39**
<br>Potential arguments to fix
- [ ] `order` `Default='K'`
- [ ] `dtype` `Default=None`
- [ ] `copy` `Default=False`

nilearn/_utils/numpy_conversions.py::csv_to_array - **line 144**
<br>Potential arguments to fix
- [ ] `delimiters` `Default=' \t,;'`

# nilearn/_utils/niimg_conversions.py
nilearn/_utils/niimg_conversions.py::check_niimg - **line 206**
<br>Potential arguments to fix
- [ ] `wildcards` `Default=True`
- [ ] `return_iterator` `Default=False`
- [ ] `dtype` `Default=None`
- [ ] `atleast_4d` `Default=False`
- [ ] `ensure_ndim` `Default=None`

nilearn/_utils/niimg_conversions.py::check_niimg_3d - **line 333**
<br>Potential arguments to fix
- [ ] `dtype` `Default=None`

nilearn/_utils/niimg_conversions.py::check_niimg_4d - **line 370**
<br>Potential arguments to fix
- [ ] `dtype` `Default=None`
- [ ] `return_iterator` `Default=False`

nilearn/_utils/niimg_conversions.py::concat_niimgs - **line 412**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`
- [ ] `auto_resample` `Default=False`
- [ ] `memory_level` `Default=0`
- [ ] `memory` `Default=Memory(location=None)`
- [ ] `ensure_ndim` `Default=None`
- [ ] `dtype` `Default=np.float32`

# nilearn/_utils/logger.py
nilearn/_utils/logger.py::log - **line 12**
<br>Potential arguments to fix
- [ ] `msg_level` `Default=1`
- [ ] `stack_level` `Default=1`
- [ ] `object_classes` `Default=(BaseEstimator,)`
- [ ] `verbose` `Default=1`

# nilearn/_utils/data_gen.py
nilearn/_utils/data_gen.py::generate_mni_space_img - **line 22**
<br>Potential arguments to fix
- [ ] `mask_dilation` `Default=2`
- [ ] `random_state` `Default=0`
- [ ] `res` `Default=30`
- [ ] `n_scans` `Default=1`

nilearn/_utils/data_gen.py::generate_timeseries - **line 64**
<br>Potential arguments to fix
- [ ] `random_state` `Default=0`

nilearn/_utils/data_gen.py::generate_regions_ts - **line 89**
<br>Potential arguments to fix
- [ ] `window` `Default='boxcar'`
- [ ] `random_state` `Default=0`
- [ ] `overlap` `Default=0`

nilearn/_utils/data_gen.py::generate_maps - **line 148**
<br>Potential arguments to fix
- [ ] `affine` `Default=np.eye(4)`
- [ ] `random_state` `Default=0`
- [ ] `window` `Default='boxcar'`
- [ ] `border` `Default=1`
- [ ] `overlap` `Default=0`

nilearn/_utils/data_gen.py::generate_labeled_regions - **line 198**
<br>Potential arguments to fix
- [ ] `dtype` `Default='int32'`
- [ ] `affine` `Default=np.eye(4)`
- [ ] `labels` `Default=None`
- [ ] `random_state` `Default=0`

nilearn/_utils/data_gen.py::generate_fake_fmri - **line 252**
<br>Potential arguments to fix
- [ ] `random_state` `Default=0`
- [ ] `block_type` `Default='classification'`
- [ ] `block_size` `Default=None`
- [ ] `n_blocks` `Default=None`
- [ ] `affine` `Default=np.eye(4)`
- [ ] `kind` `Default='noise'`
- [ ] `length` `Default=17`
- [ ] `shape` `Default=(10, 11, 12)`

nilearn/_utils/data_gen.py::generate_fake_fmri_data_and_design - **line 375**
<br>Potential arguments to fix
- [ ] `random_state` `Default=0`
- [ ] `affine` `Default=np.eye(4)`
- [ ] `rk` `Default=3`

nilearn/_utils/data_gen.py::write_fake_fmri_data_and_design - **line 426**
<br>Potential arguments to fix
- [ ] `file_path` `Default=None`
- [ ] `random_state` `Default=0`
- [ ] `affine` `Default=np.eye(4)`
- [ ] `rk` `Default=3`

nilearn/_utils/data_gen.py::write_fake_bold_img - **line 512**
<br>Potential arguments to fix
- [ ] `random_state` `Default=0`
- [ ] `affine` `Default=np.eye(4)`

nilearn/_utils/data_gen.py::generate_signals_from_precisions - **line 546**
<br>Potential arguments to fix
- [ ] `random_state` `Default=0`
- [ ] `max_n_samples` `Default=100`
- [ ] `min_n_samples` `Default=50`

nilearn/_utils/data_gen.py::generate_group_sparse_gaussian_graphs - **line 590**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`
- [ ] `random_state` `Default=0`
- [ ] `density` `Default=0.1`
- [ ] `max_n_samples` `Default=50`
- [ ] `min_n_samples` `Default=30`
- [ ] `n_features` `Default=30`
- [ ] `n_subjects` `Default=5`

nilearn/_utils/data_gen.py::basic_paradigm - **line 685**
<br>Potential arguments to fix
- [ ] `condition_names_have_spaces` `Default=False`

nilearn/_utils/data_gen.py::basic_confounds - **line 713**
<br>Potential arguments to fix
- [ ] `random_state` `Default=0`

nilearn/_utils/data_gen.py::generate_random_img - **line 791**
<br>Potential arguments to fix
- [ ] `random_state` `Default=np.random.RandomState(0)`
- [ ] `affine` `Default=np.eye(4)`

nilearn/_utils/data_gen.py::create_fake_bids_dataset - **line 830**
<br>Potential arguments to fix
- [ ] `entities` `Default=None`
- [ ] `random_state` `Default=0`
- [ ] `confounds_tag` `Default='desc-confounds_timeseries'`
- [ ] `with_confounds` `Default=True`
- [ ] `with_derivatives` `Default=True`
- [ ] `n_runs` `Default=[1, 3]`
- [ ] `tasks` `Default=['localizer', 'main']`
- [ ] `n_ses` `Default=2`
- [ ] `n_sub` `Default=10`
- [ ] `base_dir` `Default=Path()`

# nilearn/_utils/niimg.py
nilearn/_utils/niimg.py::load_niimg - **line 101**
<br>Potential arguments to fix
- [ ] `dtype` `Default=None`

# nilearn/_utils/class_inspect.py
nilearn/_utils/class_inspect.py::get_params - **line 10**
<br>Potential arguments to fix
- [ ] `ignore` `Default=None`

nilearn/_utils/class_inspect.py::enclosing_scope_name - **line 51**
<br>Potential arguments to fix
- [ ] `stack_level` `Default=2`
- [ ] `ensure_estimator` `Default=True`

# nilearn/_utils/cache_mixin.py
nilearn/_utils/cache_mixin.py::cache - **line 87**
<br>Potential arguments to fix
- [ ] `shelve` `Default=False`
- [ ] `memory_level` `Default=None`
- [ ] `func_memory_level` `Default=None`

# nilearn/_utils/helpers.py
nilearn/_utils/helpers.py::rename_parameters - **line 7**
<br>Potential arguments to fix
- [ ] `lib_name` `Default='Nilearn'`
- [ ] `end_version` `Default='future'`

nilearn/_utils/helpers.py::remove_parameters - **line 109**
<br>Potential arguments to fix
- [ ] `end_version` `Default='future'`

# nilearn/plotting/cm.py
nilearn/plotting/cm.py::alpha_cmap - **line 120**
<br>Potential arguments to fix
- [ ] `alpha_max` `Default=1.0`
- [ ] `alpha_min` `Default=0.5`
- [ ] `name` `Default=''`

nilearn/plotting/cm.py::dim_cmap - **line 310**
<br>Potential arguments to fix
- [ ] `to_white` `Default=True`
- [ ] `factor` `Default=0.3`

# nilearn/plotting/matrix_plotting.py
nilearn/plotting/matrix_plotting.py::plot_matrix - **line 206**
<br>Potential arguments to fix
- [ ] `reorder` `Default=False`
- [ ] `grid` `Default=False`
- [ ] `auto_fit` `Default=True`
- [ ] `tri` `Default='full'`
- [ ] `axes` `Default=None`
- [ ] `figure` `Default=None`
- [ ] `labels` `Default=None`

nilearn/plotting/matrix_plotting.py::plot_contrast_matrix - **line 334**
<br>Potential arguments to fix
- [ ] `ax` `Default=None`

nilearn/plotting/matrix_plotting.py::plot_design_matrix - **line 412**
<br>Potential arguments to fix
- [ ] `ax` `Default=None`
- [ ] `rescale` `Default=True`

# nilearn/plotting/html_connectome.py
nilearn/plotting/html_connectome.py::view_connectome - **line 249**
<br>Potential arguments to fix
- [ ] `title_fontsize` `Default=25`
- [ ] `title` `Default=None`
- [ ] `colorbar_fontsize` `Default=25`
- [ ] `colorbar_height` `Default=0.5`
- [ ] `colorbar` `Default=True`
- [ ] `node_size` `Default=3.0`
- [ ] `node_color` `Default='auto'`
- [ ] `linewidth` `Default=6.0`
- [ ] `symmetric_cmap` `Default=True`
- [ ] `edge_cmap` `Default=cm.bwr`
- [ ] `edge_threshold` `Default=None`

nilearn/plotting/html_connectome.py::view_markers - **line 355**
<br>Potential arguments to fix
- [ ] `title_fontsize` `Default=25`
- [ ] `title` `Default=None`
- [ ] `marker_labels` `Default=None`
- [ ] `marker_size` `Default=5.0`
- [ ] `marker_color` `Default='auto'`

# nilearn/plotting/surf_plotting.py
nilearn/plotting/surf_plotting.py::plot_surf - **line 631**
<br>Potential arguments to fix
- [ ] `axes` `Default=None`
- [ ] `title_font_size` `Default=18`
- [ ] `cbar_vmax` `Default=None`
- [ ] `cbar_vmin` `Default=None`
- [ ] `threshold` `Default=None`
- [ ] `symmetric_cmap` `Default=False`
- [ ] `engine` `Default='matplotlib'`
- [ ] `bg_map` `Default=None`
- [ ] `surf_map` `Default=None`

nilearn/plotting/surf_plotting.py::plot_surf_contours - **line 854**
<br>Potential arguments to fix
- [ ] `legend` `Default=False`
- [ ] `colors` `Default=None`
- [ ] `labels` `Default=None`
- [ ] `levels` `Default=None`
- [ ] `axes` `Default=None`

nilearn/plotting/surf_plotting.py::plot_surf_stat_map - **line 992**
<br>Potential arguments to fix
- [ ] `axes` `Default=None`
- [ ] `title_font_size` `Default=18`
- [ ] `alpha` `Default='auto'`
- [ ] `threshold` `Default=None`
- [ ] `engine` `Default='matplotlib'`
- [ ] `bg_map` `Default=None`

nilearn/plotting/surf_plotting.py::plot_img_on_surf - **line 1284**
<br>Potential arguments to fix
- [ ] `symmetric_cbar` `Default='auto'`
- [ ] `views` `Default=['lateral', 'medial']`
- [ ] `inflate` `Default=False`
- [ ] `hemispheres` `Default=['left', 'right']`
- [ ] `mask_img` `Default=None`
- [ ] `surf_mesh` `Default='fsaverage5'`

nilearn/plotting/surf_plotting.py::plot_surf_roi - **line 1471**
<br>Potential arguments to fix
- [ ] `axes` `Default=None`
- [ ] `title_font_size` `Default=18`
- [ ] `vmax` `Default=None`
- [ ] `vmin` `Default=None`
- [ ] `alpha` `Default='auto'`
- [ ] `threshold` `Default=1e-14`
- [ ] `engine` `Default='matplotlib'`
- [ ] `bg_map` `Default=None`

# nilearn/plotting/html_stat_map.py
nilearn/plotting/html_stat_map.py::view_img - **line 475**
<br>Potential arguments to fix
- [ ] `opacity` `Default=1`
- [ ] `vmin` `Default=None`
- [ ] `vmax` `Default=None`
- [ ] `symmetric_cmap` `Default=True`
- [ ] `black_bg` `Default='auto'`
- [ ] `annotate` `Default=True`
- [ ] `threshold` `Default=1e-06`
- [ ] `colorbar` `Default=True`
- [ ] `cut_coords` `Default=None`

# nilearn/plotting/img_plotting.py
nilearn/plotting/img_plotting.py::plot_img - **line 247**
<br>Potential arguments to fix
- [ ] `cbar_tick_format` `Default='%.2g'`

nilearn/plotting/img_plotting.py::plot_anat - **line 441**
<br>Potential arguments to fix
- [ ] `cbar_tick_format` `Default='%.2g'`
- [ ] `colorbar` `Default=False`
- [ ] `anat_img` `Default=MNI152TEMPLATE`

nilearn/plotting/img_plotting.py::plot_epi - **line 511**
<br>Potential arguments to fix
- [ ] `cbar_tick_format` `Default='%.2g'`
- [ ] `colorbar` `Default=False`
- [ ] `epi_img` `Default=None`

nilearn/plotting/img_plotting.py::plot_roi - **line 611**
<br>Potential arguments to fix
- [ ] `cbar_tick_format` `Default='%i'`
- [ ] `colorbar` `Default=False`
- [ ] `alpha` `Default=0.7`

nilearn/plotting/img_plotting.py::plot_prob_atlas - **line 717**
<br>Potential arguments to fix
- [ ] `alpha` `Default=0.7`
- [ ] `threshold` `Default='auto'`
- [ ] `view_type` `Default='auto'`

nilearn/plotting/img_plotting.py::plot_stat_map - **line 896**
<br>Potential arguments to fix
- [ ] `cbar_tick_format` `Default='%.2g'`

nilearn/plotting/img_plotting.py::plot_glass_brain - **line 991**
<br>Potential arguments to fix
- [ ] `plot_abs` `Default=True`
- [ ] `alpha` `Default=0.7`
- [ ] `cbar_tick_format` `Default='%.2g'`
- [ ] `display_mode` `Default='ortho'`

nilearn/plotting/img_plotting.py::plot_connectome - **line 1106**
<br>Potential arguments to fix
- [ ] `node_kwargs` `Default=None`
- [ ] `edge_kwargs` `Default=None`
- [ ] `alpha` `Default=0.7`
- [ ] `display_mode` `Default='ortho'`
- [ ] `edge_threshold` `Default=None`
- [ ] `edge_vmax` `Default=None`
- [ ] `edge_vmin` `Default=None`
- [ ] `edge_cmap` `Default=cm.bwr`
- [ ] `node_size` `Default=50`
- [ ] `node_color` `Default='auto'`

nilearn/plotting/img_plotting.py::plot_markers - **line 1217**
<br>Potential arguments to fix
- [ ] `node_kwargs` `Default=None`
- [ ] `display_mode` `Default='ortho'`
- [ ] `alpha` `Default=0.7`
- [ ] `node_threshold` `Default=None`
- [ ] `node_vmax` `Default=None`
- [ ] `node_vmin` `Default=None`
- [ ] `node_cmap` `Default=plt.cm.viridis_r`
- [ ] `node_size` `Default='auto'`

nilearn/plotting/img_plotting.py::plot_carpet - **line 1351**
<br>Potential arguments to fix
- [ ] `cmap_labels` `Default=plt.cm.gist_ncar`
- [ ] `detrend` `Default=True`
- [ ] `mask_labels` `Default=None`
- [ ] `mask_img` `Default=None`

nilearn/plotting/img_plotting.py::plot_img_comparison - **line 1601**
<br>Potential arguments to fix
- [ ] `axes` `Default=None`
- [ ] `output_dir` `Default=None`
- [ ] `src_label` `Default='image set 2'`
- [ ] `ref_label` `Default='image set 1'`
- [ ] `log` `Default=True`
- [ ] `plot_hist` `Default=True`

# nilearn/plotting/find_cuts.py
nilearn/plotting/find_cuts.py::find_xyz_cut_coords - **line 31**
<br>Potential arguments to fix
- [ ] `activation_threshold` `Default=None`
- [ ] `mask_img` `Default=None`

nilearn/plotting/find_cuts.py::find_cut_slices - **line 243**
<br>Potential arguments to fix
- [ ] `spacing` `Default='auto'`
- [ ] `n_cuts` `Default=7`
- [ ] `direction` `Default='z'`

nilearn/plotting/find_cuts.py::find_parcellation_cut_coords - **line 408**
<br>Potential arguments to fix
- [ ] `label_hemisphere` `Default='left'`
- [ ] `return_label_names` `Default=False`
- [ ] `background_label` `Default=0`

# nilearn/plotting/html_surface.py
nilearn/plotting/html_surface.py::one_mesh_info - **line 104**
<br>Potential arguments to fix
- [ ] `vmin` `Default=None`
- [ ] `vmax` `Default=None`
- [ ] `darkness` `Default=0.7`
- [ ] `bg_on_data` `Default=False`
- [ ] `symmetric_cmap` `Default=True`
- [ ] `bg_map` `Default=None`
- [ ] `black_bg` `Default=False`
- [ ] `cmap` `Default=cm.cold_hot`
- [ ] `threshold` `Default=None`

nilearn/plotting/html_surface.py::full_brain_info - **line 185**
<br>Potential arguments to fix
- [ ] `vol_to_surf_kwargs` `Default={}`
- [ ] `vmin` `Default=None`
- [ ] `vmax` `Default=None`
- [ ] `darkness` `Default=0.7`
- [ ] `bg_on_data` `Default=False`
- [ ] `symmetric_cmap` `Default=True`
- [ ] `black_bg` `Default=False`
- [ ] `cmap` `Default=cm.cold_hot`
- [ ] `threshold` `Default=None`
- [ ] `mesh` `Default='fsaverage5'`

nilearn/plotting/html_surface.py::view_img_on_surf - **line 215**
<br>Potential arguments to fix
- [ ] `vol_to_surf_kwargs` `Default={}`
- [ ] `title_fontsize` `Default=25`
- [ ] `title` `Default=None`
- [ ] `colorbar_fontsize` `Default=25`
- [ ] `colorbar_height` `Default=0.5`
- [ ] `colorbar` `Default=True`
- [ ] `symmetric_cmap` `Default=True`
- [ ] `vmin` `Default=None`
- [ ] `vmax` `Default=None`
- [ ] `black_bg` `Default=False`
- [ ] `cmap` `Default=cm.cold_hot`
- [ ] `threshold` `Default=None`
- [ ] `surf_mesh` `Default='fsaverage5'`

nilearn/plotting/html_surface.py::view_surf - **line 325**
<br>Potential arguments to fix
- [ ] `title_fontsize` `Default=25`
- [ ] `title` `Default=None`
- [ ] `colorbar_fontsize` `Default=25`
- [ ] `colorbar_height` `Default=0.5`
- [ ] `colorbar` `Default=True`
- [ ] `symmetric_cmap` `Default=True`
- [ ] `vmin` `Default=None`
- [ ] `vmax` `Default=None`
- [ ] `black_bg` `Default=False`
- [ ] `cmap` `Default=cm.cold_hot`
- [ ] `threshold` `Default=None`
- [ ] `bg_map` `Default=None`
- [ ] `surf_map` `Default=None`

# nilearn/plotting/js_plotting_utils.py
nilearn/plotting/js_plotting_utils.py::add_js_lib - **line 25**
<br>Potential arguments to fix
- [ ] `embed_js` `Default=True`

nilearn/plotting/js_plotting_utils.py::colorscale - **line 70**
<br>Potential arguments to fix
- [ ] `vmin` `Default=None`
- [ ] `vmax` `Default=None`
- [ ] `symmetric_cmap` `Default=True`
- [ ] `threshold` `Default=None`

# nilearn/experimental/surface/_datasets.py
nilearn/experimental/surface/_datasets.py::load_fsaverage - **line 17**
<br>Potential arguments to fix
- [ ] `mesh_name` `Default='fsaverage5'`

nilearn/experimental/surface/_datasets.py::fetch_nki - **line 31**
<br>Potential arguments to fix
- [ ] `n_subjects` `Default=1`

# nilearn/experimental/surface/tests/conftest.py
nilearn/experimental/surface/tests/conftest.py::mini_mask - **line: 57**
- [ ] No docstring detected.

nilearn/experimental/surface/tests/conftest.py::mini_img - **line: 63**
- [ ] No docstring detected.

nilearn/experimental/surface/tests/conftest.py::flip - **line: 68**
- [ ] No docstring detected.

nilearn/experimental/surface/tests/conftest.py::flip_img - **line: 80**
- [ ] No docstring detected.

nilearn/experimental/surface/tests/conftest.py::assert_img_equal - **line: 94**
- [ ] No docstring detected.

nilearn/experimental/surface/tests/conftest.py::drop_img_part - **line: 104**
- [ ] No docstring detected.

# nilearn/regions/region_extractor.py
nilearn/regions/region_extractor.py::connected_regions - **line 136**
<br>Potential arguments to fix
- [ ] `mask_img` `Default=None`
- [ ] `min_region_size` `Default=1350`

nilearn/regions/region_extractor.py::connected_label_regions - **line 475**
<br>Potential arguments to fix
- [ ] `labels` `Default=None`
- [ ] `connect_diag` `Default=True`
- [ ] `min_size` `Default=None`

# nilearn/regions/signal_extraction.py
nilearn/regions/signal_extraction.py::img_to_signals_labels - **line 241**
<br>Potential arguments to fix
- [ ] `strategy` `Default='mean'`
- [ ] `order` `Default='F'`
- [ ] `background_label` `Default=0`
- [ ] `mask_img` `Default=None`

nilearn/regions/signal_extraction.py::signals_to_img_labels - **line 340**
<br>Potential arguments to fix
- [ ] `order` `Default='F'`
- [ ] `background_label` `Default=0`
- [ ] `mask_img` `Default=None`

nilearn/regions/signal_extraction.py::img_to_signals_maps - **line 425**
<br>Potential arguments to fix
- [ ] `mask_img` `Default=None`

nilearn/regions/signal_extraction.py::signals_to_img_maps - **line 520**
<br>Potential arguments to fix
- [ ] `mask_img` `Default=None`

# nilearn/regions/hierarchical_kmeans_clustering.py
nilearn/regions/hierarchical_kmeans_clustering.py::hierarchical_k_means - **line 48**
<br>Potential arguments to fix
- [ ] `random_state` `Default=0`
- [ ] `verbose` `Default=0`
- [ ] `max_no_improvement` `Default=10`
- [ ] `n_init` `Default=10`
- [ ] `batch_size` `Default=1000`
- [ ] `init` `Default='k-means++'`

# nilearn/regions/rena_clustering.py
nilearn/regions/rena_clustering.py::recursive_neighbor_agglomeration - **line 364**
<br>Potential arguments to fix
- [ ] `verbose` `Default=0`
- [ ] `threshold` `Default=1e-07`
- [ ] `n_iter` `Default=10`

# nilearn/connectome/group_sparse_cov.py
nilearn/connectome/group_sparse_cov.py::group_sparse_covariance - **line 131**
<br>Potential arguments to fix
- [ ] `debug` `Default=False`
- [ ] `precisions_init` `Default=None`
- [ ] `probe_function` `Default=None`
- [ ] `verbose` `Default=0`
- [ ] `tol` `Default=0.001`
- [ ] `max_iter` `Default=50`

nilearn/connectome/group_sparse_cov.py::empirical_covariances - **line 602**
<br>Potential arguments to fix
- [ ] `standardize` `Default=False`
- [ ] `assume_centered` `Default=False`

nilearn/connectome/group_sparse_cov.py::group_sparse_scores - **line 663**
<br>Potential arguments to fix
- [ ] `debug` `Default=False`
- [ ] `duality_gap` `Default=False`

nilearn/connectome/group_sparse_cov.py::group_sparse_covariance_path - **line 777**
<br>Potential arguments to fix
- [ ] `probe_function` `Default=None`
- [ ] `debug` `Default=False`
- [ ] `verbose` `Default=0`
- [ ] `precisions_init` `Default=None`
- [ ] `max_iter` `Default=10`
- [ ] `tol` `Default=0.001`
- [ ] `test_subjs` `Default=None`

# nilearn/connectome/connectivity_matrices.py
nilearn/connectome/connectivity_matrices.py::sym_matrix_to_vec - **line 211**
<br>Potential arguments to fix
- [ ] `discard_diagonal` `Default=False`

nilearn/connectome/connectivity_matrices.py::vec_to_sym_matrix - **line 248**
<br>Potential arguments to fix
- [ ] `diagonal` `Default=None`

# nilearn/interfaces/fsl.py
nilearn/interfaces/fsl.py::get_design_from_fslmat - **line 6**
<br>Potential arguments to fix
- [ ] `column_names` `Default=None`

# nilearn/interfaces/bids/query.py
nilearn/interfaces/bids/query.py::get_bids_files - **line 151**
<br>Potential arguments to fix
- [ ] `sub_folder` `Default=True`
- [ ] `filters` `Default=None`
- [ ] `modality_folder` `Default='*'`
- [ ] `sub_label` `Default='*'`
- [ ] `file_type` `Default='*'`
- [ ] `file_tag` `Default='*'`

# nilearn/interfaces/bids/glm.py
nilearn/interfaces/bids/glm.py::save_glm_to_bids - **line 13**
<br>Potential arguments to fix
- [ ] `prefix` `Default=None`
- [ ] `out_dir` `Default='.'`
- [ ] `contrast_types` `Default=None`

# nilearn/interfaces/fmriprep/load_confounds_strategy.py
nilearn/interfaces/fmriprep/load_confounds_strategy.py::load_confounds_strategy - **line 52**
<br>Potential arguments to fix
- [ ] `denoise_strategy` `Default='simple'`

# nilearn/interfaces/fmriprep/load_confounds.py
nilearn/interfaces/fmriprep/load_confounds.py::load_confounds - **line 94**
<br>Potential arguments to fix
- [ ] `demean` `Default=True`
- [ ] `ica_aroma` `Default='full'`
- [ ] `n_compcor` `Default='all'`
- [ ] `compcor` `Default='anat_combined'`
- [ ] `global_signal` `Default='basic'`
- [ ] `wm_csf` `Default='basic'`
- [ ] `std_dvars_threshold` `Default=3`
- [ ] `fd_threshold` `Default=0.2`
- [ ] `scrub` `Default=5`
- [ ] `motion` `Default='full'`
- [ ] `strategy` `Default=('motion', 'high_pass', 'wm_csf')`

# nilearn/interfaces/fmriprep/tests/utils.py
nilearn/interfaces/fmriprep/tests/utils.py::get_testdata_path - **line 48**
<br>Potential arguments to fix
- [ ] `non_steady_state` `Default=True`

nilearn/interfaces/fmriprep/tests/utils.py::create_tmp_filepath - **line: 71**
- [ ] No docstring detected.

nilearn/interfaces/fmriprep/tests/utils.py::get_legal_confound - **line 143**
<br>Potential arguments to fix
- [ ] `non_steady_state` `Default=True`
