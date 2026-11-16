"""Downloads the data for building the reports and the examples."""

import sys

from nilearn import datasets
from nilearn._utils.docs import Description, check_content_types

# mismatches between the content of the fetchers' data
# and their description
DESCRIPTION_ERRORS: list[str] = []


def _fetch(fn, *args, **kwargs):
    """Run a fetcher and check its data match their description.

    Mismatches are appended to DESCRIPTION_ERRORS.
    """
    data = fn(*args, **kwargs)
    description = getattr(data, "description", None)
    if isinstance(description, Description):
        DESCRIPTION_ERRORS.extend(
            f"{fn.__name__}({args}, {kwargs}): {msg}"
            for msg in check_content_types(data, description.content)
        )
    return data


def main(args=sys.argv) -> None:
    build_type = args[1] if len(args) > 1 else "partial"

    print(f"Getting data for a build: {build_type}")

    # %%
    # Even on partial build of the doc,
    # the reports for GLM and maskers need to be built.
    # See doc/visual_testing/reporter_visual_inspection_suite.py
    # The following section downloads the necessary data for this.

    _fetch(datasets.fetch_icbm152_2009)

    _fetch(datasets.fetch_atlas_difumo, dimension=64, resolution_mm=2)
    _fetch(datasets.fetch_atlas_schaefer_2018)

    _, urls = _fetch(datasets.fetch_ds000030_urls)
    # Only keep the files for the ``stopsignal`` task that are actually
    # needed by examples/04_glm_first_level/plot_bids_features.py:
    # the raw functional data and events, the relevant fMRIPrep
    # derivatives, and the FSL ``stopsignal.feat`` derivatives used
    # for comparison.
    # Restricting the download with
    # ``inclusion_filters`` this way, rather than trying to list every
    # folder to exclude, avoids pulling in the (much larger)
    # derivatives of the other tasks acquired for this subject.
    inclusion_patterns = ["*sub-*stopsignal*"]
    # Some fMRIPrep and FSL derivatives are are not used in that example.
    exclusion_patterns = [
        "*_space-T1w*",
        "*_space-fsaverage*",
        "*cope*gz",
        "*jpg",
        "*png",
        "*txt",
        "*tiff",
        "*gif",
        "*res4D*",
    ]
    urls = datasets.select_from_index(
        urls,
        inclusion_filters=inclusion_patterns,
        exclusion_filters=exclusion_patterns,
        n_subjects=1,
    )
    _fetch(datasets.fetch_openneuro_dataset, urls=urls)

    _fetch(datasets.fetch_adhd, n_subjects=1)
    _fetch(datasets.fetch_development_fmri, n_subjects=60)
    _fetch(datasets.fetch_fiac_first_level)
    _fetch(datasets.fetch_oasis_vbm, n_subjects=100)
    _fetch(datasets.fetch_localizer_first_level)

    if build_type in ["full", "html", "html-strict"]:
        # On full build of the doc we get all the data
        # needed for building all the examples.

        _fetch(datasets.fetch_atlas_allen_2011)
        _fetch(datasets.fetch_atlas_surf_destrieux)
        for resolution in [64, 197, 444]:
            _fetch(
                datasets.fetch_atlas_basc_multiscale_2015,
                version="sym",
                resolution=resolution,
            )
        _fetch(datasets.fetch_atlas_destrieux_2009)
        _fetch(datasets.fetch_atlas_harvard_oxford, "cort-maxprob-thr25-2mm")
        _fetch(datasets.fetch_atlas_juelich, "maxprob-thr0-1mm")
        for dimension in [10, 20]:
            _fetch(
                datasets.fetch_atlas_smith_2009,
                resting=False,
                dimension=dimension,
            )
        _fetch(datasets.fetch_atlas_yeo_2011, n_networks=7)
        _fetch(datasets.fetch_atlas_yeo_2011, n_networks=17)
        _fetch(datasets.fetch_atlas_msdl)

        _fetch(datasets.fetch_surf_fsaverage)
        _fetch(datasets.fetch_surf_fsaverage, "fsaverage")

        datasets.load_mni152_brain_mask(resolution=2)
        _fetch(datasets.fetch_icbm152_brain_gm_mask)

        datasets.load_sample_motor_activation_image()

        _fetch(datasets.fetch_coords_power_2011)
        _fetch(datasets.fetch_coords_dosenbach_2010)

        _fetch(datasets.fetch_haxby)
        _fetch(datasets.fetch_language_localizer_demo_dataset)
        _fetch(datasets.fetch_localizer_button_task)
        for contrast, n_subjects in zip(
            [
                "calculation (auditory and visual cue)",
                "vertical checkerboard",
                "horizontal checkerboard",
                "left vs right button press",
                "left button press (auditory cue)",
            ],
            [20, 16, 16, 16, 94],
            strict=False,
        ):
            _fetch(
                datasets.fetch_localizer_contrasts,
                contrasts=[contrast],
                n_subjects=n_subjects,
            )
        _fetch(
            datasets.fetch_neurovault_ids,
            image_ids=(151, 3041, 3042, 2676, 2675, 2818, 2834),
        )
        _fetch(
            datasets.fetch_neurovault,
            max_images=30,
            fetch_neurosynth_words=True,
        )
        _fetch(datasets.fetch_neurovault_auditory_computation_task)
        _fetch(
            datasets.fetch_megatrawls_netmats,
            dimensionality=300,
            timeseries="eigen_regression",
            matrices="partial_correlation",
        )
        _fetch(datasets.fetch_mixed_gambles, n_subjects=16)
        _fetch(datasets.fetch_miyawaki2008)
        _fetch(datasets.fetch_spm_multimodal_fmri)
        _fetch(datasets.fetch_spm_auditory)
        _fetch(datasets.fetch_surf_nki_enhanced, n_subjects=1)

    if DESCRIPTION_ERRORS:
        raise TypeError(
            "Content of fetched data does not match their description:\n- "
            + "\n- ".join(DESCRIPTION_ERRORS)
        )


if __name__ == "__main__":
    main()
