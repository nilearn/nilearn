"""Contain a bunch of functions run via __main__().

The functions represent feature comprehensive examples
to visualize, inspect, and test the functionality
of nilearn.reporting.make_glm_reports().

Disable any of the function calls in the __main__()
to run a specific script and save time.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

import nilearn
from nilearn import datasets
from nilearn.glm.first_level import FirstLevelModel, first_level_from_bids
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import resample_to_img
from nilearn.interfaces.fsl import get_design_from_fslmat
from nilearn.maskers import NiftiSpheresMasker
from nilearn.reporting import make_glm_report

REPORTS_DIR = (
    Path(__file__).parent.parent / "modules" / "generated_glm_reports"
)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def report_flm_adhd_dmn():
    t_r = 2.0
    slice_time_ref = 0.0
    n_scans = 176
    pcc_coords = (0, -53, 26)
    adhd_dataset = nilearn.datasets.fetch_adhd(n_subjects=1)
    seed_masker = NiftiSpheresMasker(
        [pcc_coords],
        radius=10,
        detrend=True,
        standardize=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.0,
        memory="nilearn_cache",
        memory_level=1,
        verbose=0,
    )
    seed_time_series = seed_masker.fit_transform(adhd_dataset.func[0])
    frametimes = np.linspace(0, (n_scans - 1) * t_r, n_scans)
    design_matrix = make_first_level_design_matrix(
        frametimes,
        hrf_model="spm",
        add_regs=seed_time_series,
        add_reg_names=["pcc_seed"],
    )
    dmn_contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))
    contrasts = {"seed_based_glm": dmn_contrast}

    first_level_model = FirstLevelModel(t_r=t_r, slice_time_ref=slice_time_ref)
    first_level_model = first_level_model.fit(
        run_imgs=adhd_dataset.func[0], design_matrices=design_matrix
    )

    report = make_glm_report(
        first_level_model,
        contrasts=contrasts,
        title="ADHD DMN Report",
        cluster_threshold=15,
        height_control="bonferroni",
        min_distance=8.0,
        plot_type="glass",
        report_dims=(1200, "a"),
    )
    output_filename = "flm_adhd_dmn.html"
    output_filepath = REPORTS_DIR / output_filename
    report.save_as_html(output_filepath)
    report.get_iframe()


###########################################################################


def _fetch_bids_data():
    _, urls = datasets.func.fetch_openneuro_dataset_index()

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
    urls = datasets.func.select_from_index(
        urls, exclusion_filters=exclusion_patterns, n_subjects=1
    )

    data_dir, _ = datasets.func.fetch_openneuro_dataset(urls=urls)
    return data_dir


def _make_flm(data_dir):
    task_label = "stopsignal"
    space_label = "MNI152NLin2009cAsym"
    derivatives_folder = "derivatives/fmriprep"
    (
        models,
        models_run_imgs,
        models_events,
        models_confounds,
    ) = first_level_from_bids(
        data_dir,
        task_label,
        space_label,
        smoothing_fwhm=5.0,
        derivatives_folder=derivatives_folder,
    )

    model, imgs, _, _ = (
        models[0],
        models_run_imgs[0],
        models_events[0],
        models_confounds[0],
    )
    subject = f"sub-{model.subject_label}"
    design_matrix = _make_design_matrix_for_bids_feature(data_dir, subject)
    model.fit(imgs, design_matrices=[design_matrix])
    return model, subject


def _make_design_matrix_for_bids_feature(data_dir, subject):
    fsl_design_matrix_path = (
        Path(data_dir)
        / "derivatives"
        / "task"
        / subject
        / "stopsignal.feat"
        / "design.mat"
    )
    design_matrix = get_design_from_fslmat(
        fsl_design_matrix_path, column_names=None
    )

    design_columns = [
        f"cond_{i:02d}" for i in range(len(design_matrix.columns))
    ]
    design_columns[0] = "Go"
    design_columns[4] = "StopSuccess"
    design_matrix.columns = design_columns
    return design_matrix


def report_flm_bids_features():
    data_dir = _fetch_bids_data()
    model, _ = _make_flm(data_dir)
    title = "FLM Bids Features Stat maps"
    report = make_glm_report(
        model=model,
        contrasts="StopSuccess - Go",
        title=title,
        cluster_threshold=3,
    )
    output_filename = "flm_bids_features.html"
    output_filepath = REPORTS_DIR / output_filename
    report.save_as_html(output_filepath)
    report.get_iframe()


###########################################################################


def report_flm_fiac():
    data = datasets.func.fetch_fiac_first_level()
    fmri_img = [data["func1"], data["func2"]]

    from nilearn.image import mean_img

    mean_img_ = mean_img(fmri_img[0])

    design_files = [data["design_matrix1"], data["design_matrix2"]]
    design_matrices = [pd.DataFrame(np.load(df)["X"]) for df in design_files]

    fmri_glm = FirstLevelModel(mask_img=data["mask"], minimize_memory=True)
    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices)

    n_columns = design_matrices[0].shape[1]

    contrasts = {
        "SStSSp_minus_DStDSp": np.array([[1, 0, 0, -1]]),
        "DStDSp_minus_SStSSp": np.array([[-1, 0, 0, 1]]),
        "DSt_minus_SSt": np.array([[-1, -1, 1, 1]]),
        "DSp_minus_SSp": np.array([[-1, 1, -1, 1]]),
        "DSt_minus_SSt_for_DSp": np.array([[0, -1, 0, 1]]),
        "DSp_minus_SSp_for_DSt": np.array([[0, 0, -1, 1]]),
        "Deactivation": np.array([[-1, -1, -1, -1, 4]]),
        "Effects_of_interest": np.eye(n_columns)[:5, :],  # An F-contrast
    }
    report = make_glm_report(
        fmri_glm,
        contrasts,
        bg_img=mean_img_,
        height_control="fdr",
    )
    output_filename = "flm_fiac.html"
    output_filepath = REPORTS_DIR / output_filename
    report.save_as_html(output_filepath)
    report.get_iframe()


###########################################################################


def _make_design_matrix_slm_oasis(oasis_dataset, n_subjects):
    age = oasis_dataset.ext_vars["age"].astype(float)
    sex = oasis_dataset.ext_vars["mf"] == "F"
    intercept = np.ones(n_subjects)
    design_matrix = pd.DataFrame(
        np.vstack((age, sex, intercept)).T, columns=["age", "sex", "intercept"]
    )
    return design_matrix


def report_slm_oasis():
    n_subjects = 5  # more subjects requires more memory
    oasis_dataset = nilearn.datasets.fetch_oasis_vbm(n_subjects=n_subjects)
    # Resample the images, since this mask has a different resolution
    mask_img = resample_to_img(
        nilearn.datasets.fetch_icbm152_brain_gm_mask(),
        oasis_dataset.gray_matter_maps[0],
        interpolation="nearest",
    )
    design_matrix = _make_design_matrix_slm_oasis(oasis_dataset, n_subjects)
    second_level_model = SecondLevelModel(
        smoothing_fwhm=2.0, mask_img=mask_img
    )
    second_level_model.fit(
        oasis_dataset.gray_matter_maps, design_matrix=design_matrix
    )

    # contrast = [[1, 0, 0], [0, 1, 0]]
    contrast = ["age", "sex"]
    report = make_glm_report(
        model=second_level_model,
        contrasts=contrast,
        bg_img=nilearn.datasets.fetch_icbm152_2009()["t1"],
        height_control=None,
    )
    output_filename = "slm_oasis.html"
    output_filepath = REPORTS_DIR / output_filename
    report.save_as_html(output_filepath)
    report.get_iframe()


if __name__ == "__main__":

    print("\nGenerating GLM reports templates\n")

    t0 = time.time()

    report_flm_adhd_dmn()
    report_flm_bids_features()
    report_flm_fiac()
    report_slm_oasis()

    t1 = time.time()

    print(f"\nRun took: {(t1-t0)} seconds\n")
