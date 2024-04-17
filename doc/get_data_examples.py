"""Dowxnloads all the data for the examples."""

from nilearn import datasets

datasets.fetch_atlas_allen_2011()
datasets.fetch_atlas_surf_destrieux()
datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=64)
datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=197)
datasets.fetch_atlas_basc_multiscale_2015(version="sym", resolution=444)
datasets.fetch_atlas_destrieux_2009(legacy_format=False)
datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
datasets.fetch_atlas_juelich("maxprob-thr0-1mm")
datasets.fetch_atlas_msdl()
datasets.fetch_atlas_smith_2009(resting=False, dimension=20)
datasets.fetch_atlas_smith_2009(resting=True, dimension=10)
datasets.fetch_atlas_yeo_2011()

datasets.fetch_surf_fsaverage()
datasets.fetch_surf_fsaverage("fsaverage")

datasets.load_mni152_brain_mask(resolution=2)
datasets.fetch_icbm152_2009()
datasets.fetch_icbm152_brain_gm_mask()

datasets.load_sample_motor_activation_image()

datasets.fetch_coords_power_2011()
datasets.fetch_coords_dosenbach_2010(legacy_format=False)

datasets.fetch_adhd(n_subjects=1)
datasets.fetch_development_fmri(n_subjects=60)
datasets.fetch_fiac_first_level()
datasets.fetch_haxby()
datasets.fetch_language_localizer_demo_dataset(legacy_output=False)
datasets.fetch_localizer_button_task(legacy_format=False)
datasets.fetch_localizer_calculation_task(n_subjects=20, legacy_format=False)
# datasets.fetch_localizer_contrasts(
#     ["left button press (auditory cue)"],
#     n_subjects=94,
#     legacy_format=False,
# )
# for contrast in [
#     "vertical checkerboard",
#     "horizontal checkerboard",
#     "left vs right button press",
# ]:
#     datasets.fetch_localizer_contrasts(
#         contrasts=contrast,
#         n_subjects=16,
#         legacy_format=False,
#     )
# datasets.fetch_localizer_first_level()
# datasets.fetch_neurovault_motor_task()
# datasets.fetch_neurovault_ids(
#     image_ids=(151, 3041, 3042, 2676, 2675, 2818, 2834)
# )
# datasets.fetch_neurovault(max_images=30, fetch_neurosynth_words=True)
# datasets.fetch_neurovault_auditory_computation_task()
# datasets.fetch_megatrawls_netmats(
#     dimensionality=300,
#     timeseries="eigen_regression",
#     matrices="partial_correlation",
# )
# datasets.fetch_mixed_gambles(n_subjects=16)
# datasets.fetch_miyawaki2008()
# datasets.fetch_oasis_vbm(n_subjects=100, legacy_format=False)
# datasets.fetch_spm_multimodal_fmri()
# datasets.fetch_spm_auditory()
# datasets.fetch_surf_nki_enhanced(n_subjects=1)

# _, urls = datasets.fetch_ds000030_urls()

# exclusion_patterns = [
#     "*group*",
#     "*phenotype*",
#     "*mriqc*",
#     "*parameter_plots*",
#     "*physio_plots*",
#     "*space-fsaverage*",
#     "*space-T1w*",
#     "*dwi*",
#     "*beh*",
#     "*task-bart*",
#     "*task-rest*",
#     "*task-scap*",
#     "*task-task*",
# ]
# urls = datasets.select_from_index(
#     urls, exclusion_filters=exclusion_patterns, n_subjects=1
# )
# datasets.fetch_openneuro_dataset(urls=urls)
