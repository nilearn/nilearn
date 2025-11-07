"""Run tests on nilearn.utils."""

import contextlib

import pytest

from nilearn.utils.discovery import all_displays, all_estimators, all_functions

with contextlib.suppress(ImportError):
    from rich import print


@pytest.mark.parametrize(
    "type_filter, n_expected",
    [
        (None, 33),
        ("classifier", 3),
        ("regressor", 3),
        ("cluster", 2),
        ("masker", 15),
        ("multi_masker", 6),
        ("transformer", 22),
    ],
)
def test_all_estimators(
    matplotlib_pyplot,  # noqa : ARG001
    type_filter,
    n_expected,
):
    """Check number of estimators in public API."""
    estimators = all_estimators(type_filter=type_filter)
    print(estimators)
    assert len(estimators) == n_expected


# TODO
# for some reason this test is flaky
# and the number of functions found can vary
# usually make_glm_report is missing sometimes
# making sure that make_glm_report is part of the API
# even when matplotlib is not available should help
@pytest.mark.flaky(reruns=5, reruns_delay=2)
def test_all_functions():
    """Check number of functions in public API."""
    fn = [x[0] for x in all_functions()]

    print(fn)

    expected = (
        "all_displays",
        "all_estimators",
        "all_functions",
        "apply_mask",
        "binarize_img",
        "butterworth",
        "check_design_matrix",
        "clean",
        "clean_img",
        "cluster_level_inference",
        "compute_background_mask",
        "compute_brain_mask",
        "compute_contrast",
        "compute_epi_mask",
        "compute_fixed_effects",
        "compute_multi_background_mask",
        "compute_multi_brain_mask",
        "compute_multi_epi_mask",
        "compute_regressor",
        "concat_imgs",
        "connected_label_regions",
        "connected_regions",
        "coord_transform",
        "copy_img",
        "cov_to_corr",
        "crop_img",
        "expression_to_contrast_vector",
        "fdr_threshold",
        "fetch_abide_pcp",
        "fetch_adhd",
        "fetch_atlas_aal",
        "fetch_atlas_allen_2011",
        "fetch_atlas_basc_multiscale_2015",
        "fetch_atlas_craddock_2012",
        "fetch_atlas_destrieux_2009",
        "fetch_atlas_difumo",
        "fetch_atlas_harvard_oxford",
        "fetch_atlas_juelich",
        "fetch_atlas_msdl",
        "fetch_atlas_pauli_2017",
        "fetch_atlas_schaefer_2018",
        "fetch_atlas_smith_2009",
        "fetch_atlas_surf_destrieux",
        "fetch_atlas_talairach",
        "fetch_atlas_yeo_2011",
        "fetch_coords_dosenbach_2010",
        "fetch_coords_power_2011",
        "fetch_coords_seitzman_2018",
        "fetch_development_fmri",
        "fetch_ds000030_urls",
        "fetch_fiac_first_level",
        "fetch_haxby",
        "fetch_icbm152_2009",
        "fetch_icbm152_brain_gm_mask",
        "fetch_language_localizer_demo_dataset",
        "fetch_localizer_button_task",
        "fetch_localizer_calculation_task",
        "fetch_localizer_contrasts",
        "fetch_localizer_first_level",
        "fetch_megatrawls_netmats",
        "fetch_mixed_gambles",
        "fetch_miyawaki2008",
        "fetch_neurovault",
        "fetch_neurovault_auditory_computation_task",
        "fetch_neurovault_ids",
        "fetch_neurovault_motor_task",
        "fetch_oasis_vbm",
        "fetch_openneuro_dataset",
        "fetch_spm_auditory",
        "fetch_spm_multimodal_fmri",
        "fetch_surf_fsaverage",
        "fetch_surf_nki_enhanced",
        "find_cut_slices",
        "find_parcellation_cut_coords",
        "find_probabilistic_atlas_cut_coords",
        "find_xyz_cut_coords",
        "first_level_from_bids",
        "get_bids_files",
        "get_clusters_table",
        "get_data",
        "get_data_dirs",
        "get_design_from_fslmat",
        "get_projector",
        "get_slicer",
        "glover_dispersion_derivative",
        "glover_hrf",
        "glover_time_derivative",
        "group_sparse_covariance",
        "high_variance_confounds",
        "high_variance_confounds",
        "img_to_signals_labels",
        "img_to_signals_maps",
        "index_img",
        "intersect_masks",
        "iter_img",
        "largest_connected_component_img",
        "load_confounds",
        "load_confounds_strategy",
        "load_fsaverage",
        "load_fsaverage_data",
        "load_img",
        "load_mni152_brain_mask",
        "load_mni152_gm_mask",
        "load_mni152_gm_template",
        "load_mni152_wm_mask",
        "load_mni152_wm_template",
        "load_nki",
        "load_surf_data",
        "load_surf_mesh",
        "make_glm_report",
        "make_first_level_design_matrix",
        "make_second_level_design_matrix",
        "math_img",
        "mean_img",
        "mean_scaling",
        "new_img_like",
        "non_parametric_inference",
        "parse_bids_filename",
        "patch_openneuro_dataset",
        "permuted_ols",
        "plot_anat",
        "plot_bland_altman",
        "plot_carpet",
        "plot_connectome",
        "plot_contrast_matrix",
        "plot_design_matrix",
        "plot_design_matrix_correlation",
        "plot_epi",
        "plot_event",
        "plot_glass_brain",
        "plot_img",
        "plot_img_comparison",
        "plot_img_on_surf",
        "plot_markers",
        "plot_matrix",
        "plot_prob_atlas",
        "plot_roi",
        "plot_stat_map",
        "plot_surf",
        "plot_surf_contours",
        "plot_surf_roi",
        "plot_surf_stat_map",
        "prec_to_partial",
        "recursive_neighbor_agglomeration",
        "reorder_img",
        "resample_img",
        "resample_to_img",
        "run_glm",
        "save_glm_to_bids",
        "save_glm_to_bids",
        "select_from_index",
        "show",
        "signals_to_img_labels",
        "signals_to_img_maps",
        "smooth_img",
        "spm_dispersion_derivative",
        "spm_hrf",
        "spm_time_derivative",
        "swap_img_hemispheres",
        "sym_matrix_to_vec",
        "threshold_img",
        "threshold_stats_img",
        "unmask",
        "vec_to_sym_matrix",
        "view_connectome",
        "view_img",
        "view_img_on_surf",
        "view_markers",
        "view_surf",
        "vol_to_surf",
    )
    assert len(fn) == len(expected), (
        f"Difference: {set(fn).symmetric_difference(expected)}"
    )


@pytest.mark.parametrize(
    "type_filter, n_expected",
    [
        (None, 27),
        ("slicer", 24),
        ("axe", 3),
    ],
)
def test_all_displays(
    type_filter,
    n_expected,
    matplotlib_pyplot,  # noqa : ARG001
):
    """Check number of functions in public API."""
    disp = all_displays(type_filter)
    print(disp)
    assert len(disp) == n_expected
