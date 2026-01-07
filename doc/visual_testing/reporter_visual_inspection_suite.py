"""Runs shorter version of several examples to generate their reports.

Can be for GLM reports or masker reports.
"""

import contextlib
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import clone

from nilearn.datasets import (
    fetch_adhd,
    fetch_atlas_difumo,
    fetch_atlas_schaefer_2018,
    fetch_atlas_surf_destrieux,
    fetch_development_fmri,
    fetch_ds000030_urls,
    fetch_fiac_first_level,
    fetch_icbm152_2009,
    fetch_icbm152_brain_gm_mask,
    fetch_localizer_first_level,
    fetch_oasis_vbm,
    fetch_openneuro_dataset,
    load_fsaverage,
    load_fsaverage_data,
    load_mni152_gm_template,
    load_sample_motor_activation_image,
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
    NiftiLabelsMasker,
    NiftiMapsMasker,
    NiftiMasker,
    NiftiSpheresMasker,
    SurfaceLabelsMasker,
    SurfaceMapsMasker,
    SurfaceMasker,
)
from nilearn.reporting.glm_reporter import HTMLReport
from nilearn.surface import SurfaceImage

with contextlib.suppress(Exception):
    from rich import print

REPORTS_DIR = Path(__file__).parent.parent / "modules" / "generated_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def verbose_save(report, file: str) -> None:
    """Save reportas html  and say where it was saved."""
    report.save_as_html(REPORTS_DIR / file)
    print(f"Report saved to {REPORTS_DIR / file}")


# %%%%%%%%%% GLM REPORTS %%%%%%%%%%


# %%
# Adapted from examples/04_glm_first_level/plot_adhd_dmn.py
def report_flm_adhd_dmn(build_type):
    if build_type == "partial":
        _generate_dummy_html(filenames=["flm_adhd_dmn.html"])
        return None

    adhd_dataset = fetch_adhd(n_subjects=1)

    pcc_coords = (0, -53, 26)

    seed_masker = NiftiSpheresMasker(
        [pcc_coords],
        radius=10,
        detrend=True,
        standardize="zscore_sample",
        low_pass=0.1,
        high_pass=0.01,
        t_r=adhd_dataset.t_r,
        memory="nilearn_cache",
        memory_level=1,
    )

    seed_time_series = seed_masker.fit_transform(adhd_dataset.func[0])
    n_scans = seed_time_series.shape[0]

    frametimes = np.linspace(0, (n_scans - 1) * adhd_dataset.t_r, n_scans)

    design_matrix = make_first_level_design_matrix(
        frametimes,
        hrf_model="spm",
        add_regs=seed_time_series,
        add_reg_names=["pcc_seed"],
    )
    dmn_contrast = np.array([1] + [0] * (design_matrix.shape[1] - 1))
    contrasts = {"seed_based_glm": dmn_contrast}

    first_level_model = FirstLevelModel(slice_time_ref=0)
    first_level_model = first_level_model.fit(
        run_imgs=adhd_dataset.func[0], design_matrices=design_matrix
    )

    report = first_level_model.generate_report(
        contrasts=contrasts,
        title="ADHD DMN Report",
        cluster_threshold=15,
        alpha=0.0009,
        height_control="bonferroni",
        min_distance=8.0,
        plot_type="glass",
        report_dims=(1200, "a"),
    )

    verbose_save(report, "flm_adhd_dmn.html")

    return report


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


def report_flm_bids_features(build_type):
    if build_type == "partial":
        _generate_dummy_html(filenames=["flm_bids_features.html"])
        return None

    data_dir = _fetch_bids_data()
    model, _ = _make_flm(data_dir)
    title = "FLM Bids Features Stat maps"

    report = model.generate_report(
        contrasts="StopSuccess - Go",
        title=title,
        cluster_threshold=3,
        plot_type="glass",
    )

    verbose_save(report, "flm_bids_features.html")

    return report


# %%
# adapted from examples/04_glm_first_level/plot_two_runs_model.py
def report_flm_fiac(build_type):
    if build_type == "partial":
        _generate_dummy_html(filenames=["flm_fiac.html"])
        return None

    data = fetch_fiac_first_level()
    fmri_img = [data["func1"], data["func2"]]

    mean_img_ = mean_img(fmri_img[0])

    design_matrices = [data["design_matrix1"], data["design_matrix2"]]

    fmri_glm = FirstLevelModel(mask_img=data["mask"], minimize_memory=True)
    fmri_glm = fmri_glm.fit(fmri_img, design_matrices=design_matrices)

    n_columns = design_matrices[0].shape[1]

    contrasts = {
        "SStSSp_minus_DStDSp": np.array([[1, 0, 0, -1]]),
        "Deactivation": np.array([[-1, -1, -1, -1, 4]]),
        "Effects_of_interest": np.eye(n_columns)[:5, :],  # An F-contrast
    }
    report = fmri_glm.generate_report(
        contrasts,
        bg_img=mean_img_,
        height_control="fdr",
    )

    verbose_save(report, "flm_fiac.html")

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


def report_slm_oasis(build_type):
    if build_type == "partial":
        _generate_dummy_html(filenames=["slm_oasis.html"])
        return None

    # more subjects requires more memory
    n_subjects = 5
    oasis_dataset = fetch_oasis_vbm(n_subjects=n_subjects)

    # Resample the images, since this mask has a different resolution
    mask_img = resample_to_img(
        fetch_icbm152_brain_gm_mask(),
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

    contrast = ["age", "sex"]

    report = second_level_model.generate_report(
        contrasts=contrast,
        bg_img=fetch_icbm152_2009()["t1"],
        height_control=None,
        plot_type="glass",
    )

    verbose_save(report, "slm_oasis.html")

    return report


def report_surface_flm(build_type):
    """FirstLevelGLM surface reports."""
    flm = FirstLevelModel(mask_img=SurfaceMasker())
    report_flm_empty = flm.generate_report(height_control=None)

    verbose_save(report_flm_empty, "flm_surf_empty.html")

    if build_type == "partial":
        _generate_dummy_html(filenames=["flm_surf.html"])
        return report_flm_empty, None

    data = fetch_localizer_first_level()

    fsaverage5 = load_fsaverage()
    surface_image = SurfaceImage.from_volume(
        mesh=fsaverage5["pial"],
        volume_img=data.epi_img,
    )

    t_r = 2.4
    slice_time_ref = 0.5

    glm = FirstLevelModel(
        t_r=t_r,
        slice_time_ref=slice_time_ref,
        hrf_model="glover + derivative",
        minimize_memory=False,
        verbose=1,
    ).fit(run_imgs=surface_image, events=data.events)

    design_matrix = glm.design_matrices_[0]
    contrast_matrix = np.eye(design_matrix.shape[1])

    basic_contrasts = {
        column: contrast_matrix[i]
        for i, column in enumerate(design_matrix.columns)
    }

    contrasts = {
        "(left - right) button press": (
            basic_contrasts["audio_left_hand_button_press"]
            - basic_contrasts["audio_right_hand_button_press"]
            + basic_contrasts["visual_left_hand_button_press"]
            - basic_contrasts["visual_right_hand_button_press"]
        )
    }

    report_flm = glm.generate_report(
        contrasts,
        threshold=3.0,
        bg_img=load_fsaverage_data(data_type="sulcal", mesh_type="inflated"),
        height_control=None,
    )

    verbose_save(report_flm, "flm_surf.html")

    return report_flm, report_flm_empty


def report_surface_slm():
    slm = SecondLevelModel(mask_img=SurfaceMasker())
    report_slm_empty = slm.generate_report(height_control="bonferroni")

    verbose_save(report_slm_empty, "slm_surf_empty.html")

    return report_slm_empty


# %%%%%%%%%% MASKER REPORTS %%%%%%%%%%


def _generate_masker_report_files_partial(masker, **kwargs) -> HTMLReport:
    """Generate masker report files for partial doc build.

    Should generate reports for:
    - unfitted masker
    - unfitted masker with reports=False

    Returns
    -------
    HTMLReport: First element is the unfitted masker report.
    """
    masker_class_name = masker.__class__.__name__

    unfitted_report = masker.generate_report(
        title=f"{masker_class_name} unfitted", **kwargs
    )
    verbose_save(unfitted_report, f"{masker_class_name}_unfitted.html")

    masker.reports = False
    unfitted_report_no_reporting = masker.generate_report(
        title=f"{masker_class_name} unfitted - reports=False", **kwargs
    )
    verbose_save(
        unfitted_report_no_reporting,
        f"{masker_class_name}_unfitted_reports-False.html",
    )

    _generate_dummy_html(filenames=[f"{masker_class_name}_fitted.html"])

    return unfitted_report


def _generate_masker_report_files(
    masker, data, **kwargs
) -> tuple[HTMLReport, HTMLReport]:
    """Generate masker report files.

    Should generate reports for:
    - unfitted masker
    - unfitted masker with reports=False
    - fitted masker

    Returns
    -------
    tuple[HTMLReport, None]
        First element is the unfitted masker report.
        Second element is the fitted masker report.
    """
    masker_class_name = masker.__class__.__name__

    unfitted_report = _generate_masker_report_files_partial(masker, **kwargs)

    masker.reports = True
    masker.fit(data)
    report = masker.generate_report(**kwargs)
    verbose_save(report, f"{masker_class_name}_fitted.html")

    return unfitted_report, report


def report_nifti_masker(build_type):
    gm_template = load_mni152_gm_template()
    masker = NiftiMasker(
        mask_img=gm_template,
        standardize="zscore_sample",
        memory="nilearn_cache",
        memory_level=1,
        target_affine=gm_template.affine,
        target_shape=gm_template.shape,
    )

    if build_type == "partial":
        return _generate_masker_report_files_partial(masker), None
    else:
        data = load_sample_motor_activation_image()
        return _generate_masker_report_files(masker, data=data)


def report_nifti_labels_masker(build_type):
    atlas = fetch_atlas_schaefer_2018()
    masker = NiftiLabelsMasker(
        atlas.maps, lut=atlas.lut, standardize="zscore_sample"
    )

    if build_type == "partial":
        return _generate_masker_report_files_partial(masker), None
    else:
        data = fetch_development_fmri(n_subjects=1)
        return _generate_masker_report_files(masker, data=data.func[0])


def report_nifti_maps_masker(build_type):
    atlas = fetch_atlas_difumo(dimension=64, resolution_mm=2)
    atlas_filename = atlas["maps"]

    masker = NiftiMapsMasker(
        maps_img=atlas_filename,
        standardize="zscore_sample",
        standardize_confounds=True,
        memory="nilearn_cache",
        memory_level=1,
    )

    if build_type == "partial":
        return _generate_masker_report_files_partial(masker), None
    else:
        data = fetch_development_fmri(n_subjects=1)
        return _generate_masker_report_files(
            masker, data=data.func[0], displayed_maps=[2, 6, 7]
        )


def report_sphere_masker(build_type):
    """Generate masker with 3 spheres but only 2 in the report."""
    masker = NiftiSpheresMasker(
        seeds=[(0, -53, 26), (5, 53, -26), (0, 0, 0)],
        radius=10,
        detrend=True,
        standardize="zscore_sample",
        low_pass=0.1,
        high_pass=0.01,
        memory="nilearn_cache",
        memory_level=1,
    )

    if build_type == "partial":
        return _generate_masker_report_files_partial(masker), None
    else:
        data = fetch_development_fmri(n_subjects=1)
        masker.t_r = data.t_r
        return _generate_masker_report_files(
            masker, data.func[0], displayed_spheres=[0, 2]
        )


def load_sample_motor_activation_image_on_surface():
    """Load sample motor activation image projected on fsaverage surface."""
    stat_img = load_sample_motor_activation_image()
    fsaverage_meshes = load_fsaverage()
    return SurfaceImage.from_volume(
        mesh=fsaverage_meshes["pial"],
        volume_img=stat_img,
    )


def report_surface_masker(build_type):
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

    if build_type == "partial":
        return _generate_masker_report_files_partial(masker), None
    else:
        surface_stat_image = load_sample_motor_activation_image_on_surface()
        return _generate_masker_report_files(masker, surface_stat_image)


def report_surface_label_masker(build_type):
    fsaverage = load_fsaverage("fsaverage5")
    destrieux = fetch_atlas_surf_destrieux()
    labels_img = SurfaceImage(
        mesh=fsaverage["inflated"],
        data={
            "left": destrieux["map_left"],
            "right": destrieux["map_right"],
        },
    )
    masker = SurfaceLabelsMasker(labels_img, lut=destrieux.lut)

    if build_type == "partial":
        return _generate_masker_report_files_partial(masker), None
    else:
        surface_stat_image = load_sample_motor_activation_image_on_surface()
        return _generate_masker_report_files(masker, surface_stat_image)


def report_surface_maps_masker(build_type):
    atlas = fetch_atlas_difumo(dimension=64, resolution_mm=2)
    fsaverage5_mesh = load_fsaverage("fsaverage5")["pial"]
    surf_atlas = SurfaceImage.from_volume(
        volume_img=atlas.maps, mesh=fsaverage5_mesh
    )

    masker = SurfaceMapsMasker(surf_atlas)

    if build_type == "partial":
        _generate_dummy_html(
            filenames=[
                "SurfaceMapsMasker_fitted_plotly.html",
                "SurfaceMapsMasker_fitted_matplotlib.html",
            ]
        )
        return _generate_masker_report_files_partial(masker), None
    else:
        empty_report = _generate_masker_report_files_partial(masker)

        surface_stat_image = load_sample_motor_activation_image_on_surface()

        print("Use mpl")
        _, matplotlib_reports = _generate_masker_report_files(
            masker,
            surface_stat_image,
            engine="matplotlib",
            displayed_maps=[6, 2],
        )
        verbose_save(
            matplotlib_reports, "SurfaceMapsMasker_fitted_matplotlib.html"
        )

        print("Use plotly")
        masker = clone(masker)
        _, plotly_reports = _generate_masker_report_files(
            masker,
            surface_stat_image,
            engine="plotly",
            displayed_maps=[6, 2],
        )
        verbose_save(plotly_reports, "SurfaceMapsMasker_fitted_plotly.html")

        return empty_report, matplotlib_reports, plotly_reports


def _generate_dummy_html(filenames: list[str]):
    for x in filenames:
        with (REPORTS_DIR / x).open("w") as f:
            f.write("""
<!doctype html>
<html lang="en">
<head>
    <title>dummy content</title>
    <meta
    name="viewport"
    content="width = device-width, initial-scale = 1"
    charset="UTF-8"
    />
</head>
<body>
    <p>No content displayed on partial doc build.</p>
</body>
</html>""")


def cli_parser():
    parser = ArgumentParser(
        description="Build all types of nilearn reports.",
    )
    parser.add_argument(
        "--build_type",
        help="""
        build_type.
        """,
        choices=["full", "partial"],
        default="partial",
        type=str,
        nargs=1,
    )
    return parser


def main(args=sys.argv):
    parser = cli_parser()
    args = parser.parse_args(args[1:])
    build_type = args.build_type

    print(f"Generating reports for a build: {build_type}")

    print("\nGenerating masker reports templates\n")
    t0 = time.time()

    report_nifti_masker(build_type)
    report_nifti_maps_masker(build_type)
    report_nifti_labels_masker(build_type)
    report_sphere_masker(build_type)
    report_surface_masker(build_type)
    report_surface_label_masker(build_type)
    report_surface_maps_masker(build_type)

    t1 = time.time()
    print(f"\nTook: {t1 - t0:0.2f} seconds\n")

    print("\nGenerating GLM reports templates\n")
    t0 = time.time()

    report_flm_adhd_dmn(build_type)
    report_flm_bids_features(build_type)
    report_flm_fiac(build_type)
    report_slm_oasis(build_type)
    report_surface_flm(build_type)
    report_surface_slm()

    t1 = time.time()
    print(f"\nTook: {t1 - t0:0.2f} seconds\n")


if __name__ == "__main__":
    main()
