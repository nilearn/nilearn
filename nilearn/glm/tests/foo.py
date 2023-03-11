from nilearn._utils.data_gen import (basic_paradigm, create_fake_bids_dataset,
                                     generate_fake_fmri_data_and_design,
                                     write_fake_fmri_data_and_design)

n_sub = 2
n_ses = 2
n_runs = [3]

bids_path = create_fake_bids_dataset(
    n_sub=n_sub,
    n_ses=n_ses,
    tasks=["main"],
    n_runs=n_runs,
    entities={"acq": ["A", "B"]},
)