"""Runs shorter version of several examples to generate their reports.

Can be for GLM reports or masker reports.

This should represent feature comprehensive examples
to visualize, inspect, and test the functionality
of nilearn.reporting.make_glm_reports().
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from nilearn.datasets import (
    fetch_adhd,
    fetch_atlas_difumo,
    fetch_atlas_msdl,
    fetch_atlas_schaefer_2018,
    fetch_atlas_surf_destrieux,
    fetch_atlas_yeo_2011,
    fetch_development_fmri,
    fetch_ds000030_urls,
    fetch_fiac_first_level,
    fetch_icbm152_2009,
    fetch_icbm152_brain_gm_mask,
    fetch_miyawaki2008,
    fetch_oasis_vbm,
    fetch_openneuro_dataset,
    load_fsaverage,
    load_nki,
    select_from_index,
)
from nilearn.glm.first_level import FirstLevelModel, first_level_from_bids
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix,
)
from nilearn.glm.second_level import SecondLevelModel
from nilearn.image import mean_img, resample_to_img
from nilearn.interfaces.fsl import get_design_from_fslmat
from nilearn.maskers import (
    MultiNiftiLabelsMasker,
    MultiNiftiMapsMasker,
    MultiNiftiMasker,
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    SurfaceLabelsMasker,
    SurfaceMasker,
)
from nilearn.reporting import make_glm_report
from nilearn.surface import SurfaceImage

REPORTS_DIR = Path(__file__).parent.parent / "modules" / "generated_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# %%
# Adapted from examples/04_glm_first_level/plot_adhd_dmn.py
def report_flm_adhd_dmn():
    t_r = 2.0
    slice_time_ref = 0.0
    n_scans = 176

    pcc_coords = (0, -53, 26)

    seed_masker = NiftiSpheresMasker(
        [pcc_coords],
        radius=10,
        detrend=True,
        standardize=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=t_r,
        memory="nilearn_cache",
    )

    adhd_dataset = fetch_adhd(n_subjects=1)
    seed_time_series = seed_masker.fit_transform(adhd_dataset.func[0])

    masker_report = seed_masker.generate_report()
    masker_report.save_as_html(REPORTS_DIR / "nifti_sphere_masker.html")

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

    glm_report = make_glm_report(
        first_level_model,
        contrasts=contrasts,
        title="ADHD DMN Report",
        cluster_threshold=15,
        alpha=0.0009,
        height_control="bonferroni",
        min_distance=8.0,
        plot_type="glass",
        report_dims=(1200, "a"),
    )
    glm_report.save_as_html(REPORTS_DIR / "flm_adhd_dmn.html")

    return masker_report, glm_report


# %%
# Adapted from examples/04_glm_first_level/plot_bids_features.py
def _fetch_bids_data():
    _, urls = fetch_ds000030_urls()

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
    urls = select_from_index(
        urls, exclusion_filters=exclusion_patterns, n_subjects=1
    )

    data_dir, _ = fetch_openneuro_dataset(urls=urls)
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
        plot_type="glass",
    )

    report.save_as_html(REPORTS_DIR / "flm_bids_features.html")
    return report


# %%
# adapted from examples/04_glm_first_level/plot_two_runs_model.py
def report_flm_fiac():
    data = fetch_fiac_first_level()
    fmri_img = [data["func1"], data["func2"]]

    mean_img_ = mean_img(fmri_img[0], copy_header=True)

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
    report.save_as_html(REPORTS_DIR / "flm_fiac.html")
    return report


# %%
# adatapted from examples/05_glm_second_level/plot_oasis.py
def _make_design_matrix_slm_oasis(oasis_dataset, n_subjects):
    age = oasis_dataset.ext_vars["age"].astype(float)
    sex = oasis_dataset.ext_vars["mf"] == "F"
    intercept = np.ones(n_subjects)
    design_matrix = pd.DataFrame(
        np.vstack((age, sex, intercept)).T, columns=["age", "sex", "intercept"]
    )
    return design_matrix


def report_slm_oasis():
    # more subjects requires more memory
    n_subjects = 5
    oasis_dataset = fetch_oasis_vbm(n_subjects=n_subjects)

    # Resample the images, since this mask has a different resolution
    mask_img = resample_to_img(
        fetch_icbm152_brain_gm_mask(),
        oasis_dataset.gray_matter_maps[0],
        interpolation="nearest",
        copy_header=True,
        force_resample=True,
    )

    design_matrix = _make_design_matrix_slm_oasis(oasis_dataset, n_subjects)

    second_level_model = SecondLevelModel(
        smoothing_fwhm=2.0, mask_img=mask_img
    )
    second_level_model.fit(
        oasis_dataset.gray_matter_maps, design_matrix=design_matrix
    )

    # TODO the following crashes
    # contrast = [np.array([1, 0]), np.array([0, 1])]
    # contrast = [[1, 0, 0], [0, 1, 0]]

    # The following are equivalent
    # contrast = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    contrast = ["age", "sex"]

    report = make_glm_report(
        model=second_level_model,
        contrasts=contrast,
        bg_img=fetch_icbm152_2009()["t1"],
        height_control=None,
        plot_type="glass",
    )
    report.save_as_html(REPORTS_DIR / "slm_oasis.html")
    return report


# %%
# Adapted from examples/03_connectivity/plot_probabilistic_atlas_extraction.py
def report_nifti_maps_masker():
    atlas = fetch_atlas_msdl()
    atlas_filename = atlas["maps"]

    data = fetch_development_fmri(n_subjects=1)

    masker = NiftiMapsMasker(
        maps_img=atlas_filename,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory="nilearn_cache",
        cmap="gray",
    )
    masker.fit(data.func[0])

    report = masker.generate_report(displayed_maps=[2, 6, 7, 16, 21])
    report.save_as_html(REPORTS_DIR / "nifti_maps_masker.html")
    return report


#  %%
# Adapted from examples/06_manipulating_images/plot_nifti_labels_simple.py
def report_nifti_labels_masker():
    atlas = fetch_atlas_schaefer_2018()

    atlas.labels = np.insert(atlas.labels, 0, "Background")

    masker = NiftiLabelsMasker(
        atlas.maps,
        labels=atlas.labels,
        standardize="zscore_sample",
    )
    masker.fit()
    report = masker.generate_report()
    report.save_as_html(REPORTS_DIR / "nifti_labels_masker_atlas.html")

    data = fetch_development_fmri(n_subjects=1)
    masker.fit(data.func[0])
    report = masker.generate_report()
    report.save_as_html(REPORTS_DIR / "nifti_labels_masker_fitted.html")
    return report


#  %%
# Adapted from examples/06_manipulating_images/plot_nifti_simple.py
def report_nifti_masker():
    masker = NiftiMasker(
        standardize="zscore_sample",
        mask_strategy="epi",
        memory="nilearn_cache",
        memory_level=2,
        smoothing_fwhm=8,
        cmap="gray",
    )

    data = fetch_development_fmri(n_subjects=1)
    masker.fit(data.func[0])
    report = masker.generate_report()
    report.save_as_html(REPORTS_DIR / "nifti_masker.html")
    return report


# %%
# Adapted from examples/02_decoding/plot_miyawaki_encoding.py
def report_multi_nifti_masker():
    data = fetch_miyawaki2008()

    masker = MultiNiftiMasker(
        mask_img=data.mask,
        detrend=True,
        standardize="zscore_sample",
        n_jobs=2,
        cmap="gray",
    )
    masker.fit()
    empty_report = masker.generate_report()
    empty_report.save_as_html(REPORTS_DIR / "multi_nifti_masker.html")

    fmri_random_runs_filenames = data.func[12:]
    masker.fit(fmri_random_runs_filenames)
    report = masker.generate_report()
    report.save_as_html(REPORTS_DIR / "multi_nifti_masker_fitted.html")
    return empty_report, report


#  %%
#  Adapted from examples/03_connectivity/plot_atlas_comparison.py
def report_multi_nifti_labels_masker():
    yeo = fetch_atlas_yeo_2011()

    data = fetch_development_fmri(n_subjects=2)

    masker = MultiNiftiLabelsMasker(
        labels_img=yeo["thick_17"],
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory="nilearn_cache",
        n_jobs=2,
    )

    masker.fit()
    report = masker.generate_report()
    report.save_as_html(REPORTS_DIR / "multi_nifti_labels_masker_atlas.html")

    _ = masker.fit_transform(data.func, confounds=data.confounds)
    report = masker.generate_report()
    report.save_as_html(REPORTS_DIR / "multi_nifti_labels_masker_fitted.html")
    return report


#  %%
#  Adapted from examples/03_connectivity/plot_atlas_comparison.py
def report_multi_nifti_maps_masker():
    difumo = fetch_atlas_difumo(
        dimension=64, resolution_mm=2, legacy_format=False
    )

    data = fetch_development_fmri(n_subjects=2)

    masker = MultiNiftiMapsMasker(
        maps_img=difumo.maps,
        standardize="zscore_sample",
        standardize_confounds="zscore_sample",
        memory="nilearn_cache",
        n_jobs=2,
    )

    masker.fit()
    empty_report = masker.generate_report()
    empty_report.save_as_html(
        REPORTS_DIR / "multi_nifti_maps_masker_atlas.html"
    )

    _ = masker.fit_transform(data.func, confounds=data.confounds)
    report = masker.generate_report()
    report.save_as_html(REPORTS_DIR / "multi_nifti_maps_masker_fitted.html")

    return empty_report, report


def report_surface_masker():
    masker = SurfaceMasker()
    img = load_nki(mesh_type="inflated")[0]
    masker.fit_transform(img)
    surface_masker_report = masker.generate_report()
    surface_masker_report.save_as_html(REPORTS_DIR / "surface_masker.html")

    fsaverage = load_fsaverage("fsaverage5")
    destrieux = fetch_atlas_surf_destrieux()
    destrieux_atlas = SurfaceImage(
        mesh=fsaverage["inflated"],
        data={
            "left": destrieux["map_left"],
            "right": destrieux["map_right"],
        },
    )

    mask = destrieux_atlas
    for part in mask.data.parts:
        mask.data.parts[part] = mask.data.parts[part] == 34

    masker = SurfaceMasker(mask)
    masker.fit_transform(img)
    surface_masker_with_mask_report = masker.generate_report()
    surface_masker_with_mask_report.save_as_html(
        REPORTS_DIR / "surface_masker_with_mask.html"
    )

    return (
        surface_masker_report,
        surface_masker_with_mask_report,
    )


def report_surface_label_masker():
    fsaverage = load_fsaverage("fsaverage5")
    destrieux = fetch_atlas_surf_destrieux()
    labels_img = SurfaceImage(
        mesh=fsaverage["inflated"],
        data={
            "left": destrieux["map_left"],
            "right": destrieux["map_right"],
        },
    )
    label_names = [x.decode("utf-8") for x in destrieux.labels]

    labels_masker = SurfaceLabelsMasker(labels_img, label_names).fit()
    labels_masker_report_unfitted = labels_masker.generate_report()
    labels_masker_report_unfitted.save_as_html(
        REPORTS_DIR / "surface_label_masker_unfitted.html"
    )

    img = load_nki(mesh_type="inflated")[0]

    labels_masker.transform(img)
    labels_masker_report = labels_masker.generate_report()
    labels_masker_report.save_as_html(
        REPORTS_DIR / "surface_label_masker.html"
    )

    return (
        labels_masker_report_unfitted,
        labels_masker_report,
    )


# %%
if __name__ == "__main__":
    print("\nGenerating masker reports templates\n")
    t0 = time.time()

    report_surface_masker()
    report_surface_label_masker()
    report_nifti_masker()
    report_nifti_maps_masker()
    report_nifti_labels_masker()
    report_multi_nifti_masker()
    report_multi_nifti_labels_masker()
    report_multi_nifti_maps_masker()

    t1 = time.time()
    print(f"\nTook: {(t1 - t0)} seconds\n")

    print("\nGenerating GLM reports templates\n")
    t0 = time.time()

    report_flm_adhd_dmn()
    report_flm_bids_features()
    report_flm_fiac()
    report_slm_oasis()

    t1 = time.time()
    print(f"\nTook: {(t1 - t0)} seconds\n")
