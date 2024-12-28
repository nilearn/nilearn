"""Downloads the data for building the reports and the examples."""

import sys

from nilearn import datasets


def main(args=sys.argv) -> None:
    build_type = args[1] if len(args) > 1 else "partial"

    print(f"Getting data for a build: {build_type}")

    # %%
    # Even on partial build of the doc,
    # the reports for GLM and maskers need to be built.
    # See doc/visual_testing/reporter_visual_inspection_suite.py
    # The following section downloads the necessary data for this.

    datasets.fetch_icbm152_2009()

    datasets.fetch_atlas_difumo(
        dimension=64, resolution_mm=2, legacy_format=False
    )
    datasets.fetch_atlas_msdl()
    datasets.fetch_atlas_schaefer_2018()
    datasets.fetch_atlas_yeo_2011()

    _, urls = datasets.fetch_ds000030_urls()
    exclusion_patterns = [
        "*group*",
        "*phenotype*",
        "*mriqc*",
        "*parameter_plots*",
        "*physio_plots*",
        "*space-fsaverage*",
        "*space-T1w*",
        "*dwi*",
        "*beh*",
        "*task-bart*",
        "*task-rest*",
        "*task-scap*",
        "*task-task*",
    ]
    urls = datasets.select_from_index(
        urls, exclusion_filters=exclusion_patterns, n_subjects=1
    )
    datasets.fetch_openneuro_dataset(urls=urls)

    datasets.fetch_adhd(n_subjects=1)
    datasets.fetch_development_fmri(n_subjects=5)
    datasets.fetch_fiac_first_level()
    datasets.fetch_miyawaki2008()
    datasets.fetch_oasis_vbm(n_subjects=5)

    if build_type in ["full", "html", "html-strict"]:
        # On full build of the doc we get all the data
        # needed for building all the examples.

        datasets.fetch_atlas_allen_2011()
        datasets.fetch_atlas_surf_destrieux()
        for resolution in [64, 197, 444]:
            datasets.fetch_atlas_basc_multiscale_2015(
                version="sym", resolution=resolution
            )
        datasets.fetch_atlas_destrieux_2009()
        datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
        datasets.fetch_atlas_juelich("maxprob-thr0-1mm")
        for dimension in [10, 20]:
            datasets.fetch_atlas_smith_2009(resting=False, dimension=dimension)
        datasets.fetch_atlas_yeo_2011()

        datasets.fetch_surf_fsaverage()
        datasets.fetch_surf_fsaverage("fsaverage")

        datasets.load_mni152_brain_mask(resolution=2)
        datasets.fetch_icbm152_brain_gm_mask()

        datasets.load_sample_motor_activation_image()

        datasets.fetch_coords_power_2011()
        datasets.fetch_coords_dosenbach_2010()

        datasets.fetch_development_fmri(n_subjects=60)

        datasets.fetch_haxby()
        datasets.fetch_language_localizer_demo_dataset()
        datasets.fetch_localizer_button_task()
        datasets.fetch_localizer_calculation_task(n_subjects=20)
        for contrast, n_subjects in zip(
            [
                "vertical checkerboard",
                "horizontal checkerboard",
                "left vs right button press",
                "left button press (auditory cue)",
            ],
            [16, 16, 16, 94],
        ):
            datasets.fetch_localizer_contrasts(
                contrasts=[contrast],
                n_subjects=n_subjects,
            )
        datasets.fetch_localizer_first_level()
        datasets.fetch_neurovault_motor_task()
        datasets.fetch_neurovault_ids(
            image_ids=(151, 3041, 3042, 2676, 2675, 2818, 2834)
        )
        datasets.fetch_neurovault(max_images=30, fetch_neurosynth_words=True)
        datasets.fetch_neurovault_auditory_computation_task()
        datasets.fetch_megatrawls_netmats(
            dimensionality=300,
            timeseries="eigen_regression",
            matrices="partial_correlation",
        )
        datasets.fetch_mixed_gambles(n_subjects=16)
        datasets.fetch_oasis_vbm(n_subjects=100)
        datasets.fetch_spm_multimodal_fmri()
        datasets.fetch_spm_auditory()
        datasets.fetch_surf_nki_enhanced(n_subjects=1)


if __name__ == "__main__":
    main()
