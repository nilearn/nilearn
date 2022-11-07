#!/bin/bash -ef

DOWNLOAD_DATA=$(cat <<END
from nilearn import datasets

# Download all datasets used in examples
# Specifying parameters such as n_subjects if amount of data
# downloaded depends on them
datasets.fetch_neurovault_motor_task()
datasets.fetch_atlas_smith_2009()
datasets.fetch_haxby()
datasets.fetch_haxby(subjects=[], fetch_stimuli=True)
datasets.fetch_spm_auditory()
datasets.fetch_surf_fsaverage()
datasets.fetch_surf_fsaverage('fsaverage')
datasets.fetch_atlas_surf_destrieux()
datasets.fetch_atlas_destrieux_2009()
datasets.fetch_atlas_harvard_oxford('cort-prob-2mm')
datasets.fetch_atlas_juelich('prob-2mm')
datasets.fetch_adhd(n_subjects=1)
datasets.fetch_icbm152_2009()
datasets.fetch_localizer_button_task()
datasets.fetch_atlas_basc_multiscale_2015(version='sym')
datasets.fetch_atlas_msdl()
datasets.fetch_atlas_allen_2011()
datasets.fetch_atlas_pauli_2017()
datasets.fetch_atlas_difumo(dimension=64)
datasets.fetch_surf_nki_enhanced(n_subjects=1)
datasets.fetch_megatrawls_netmats()
datasets.fetch_mixed_gambles()
datasets.fetch_miyawaki2008()
datasets.fetch_oasis_vbm()
datasets.fetch_atlas_yeo_2011()
datasets.fetch_development_fmri(n_subjects=60)
datasets.fetch_fiac_first_level()
datasets.fetch_localizer_first_level()
datasets.fetch_spm_multimodal_fmri()
datasets.fetch_localizer_contrasts(
    ['left button press (auditory cue)'],
    n_subjects=94
)
datasets.fetch_localizer_contrasts(
    contrasts=[
        "left vs right button press",
        "vertical checkerboard",
        "horizontal checkerboard"
    ],
    n_subjects=16,
    get_tmaps=True
)
datasets.fetch_localizer_calculation_task(n_subjects=20)
datasets.fetch_neurovault_auditory_computation_task()
datasets.fetch_language_localizer_demo_dataset()
datasets.fetch_neurovault(max_images=30, fetch_neurosynth_words=True)
datasets.fetch_neurovault_ids(
    image_ids=(151, 3041, 3042, 2676, 2675, 2818, 2834)
)

# Minimal download of openneuro dataset
_, urls = datasets.fetch_ds000030_urls()
exclusion_patterns = [
    '*group*', '*phenotype*', '*mriqc*',
    '*parameter_plots*', '*physio_plots*',
    '*space-fsaverage*', '*space-T1w*',
    '*dwi*', '*beh*', '*task-bart*',
    '*task-rest*', '*task-scap*', '*task-task*'
]
urls = datasets.select_from_index(
    urls, exclusion_filters=exclusion_patterns, n_subjects=1
)
datasets.fetch_openneuro_dataset(urls=urls)

END
)

python3 -c "$DOWNLOAD_DATA"
