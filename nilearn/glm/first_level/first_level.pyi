import pathlib
from typing import Union, Optional
from _typeshed import Incomplete
from nilearn._utils import (
    fill_doc as fill_doc,
    stringify_path as stringify_path,
)
from nilearn._utils.niimg_conversions import check_niimg as check_niimg
from nilearn.glm._base import BaseGLM as BaseGLM
from nilearn.glm.contrasts import (
    expression_to_contrast_vector as expression_to_contrast_vector,
)
from nilearn.glm.first_level.design_matrix import (
    make_first_level_design_matrix as make_first_level_design_matrix,
)
from nilearn.glm.regression import (
    ARModel as ARModel,
    OLSModel as OLSModel,
    RegressionResults as RegressionResults,
    SimpleRegressionResults as SimpleRegressionResults,
)
from nilearn.image import get_data as get_data
from nilearn.interfaces.bids import (
    get_bids_files as get_bids_files,
    parse_bids_filename as parse_bids_filename,
)

def mean_scaling(Y, axis: int = ...): ...
def run_glm(
    Y,
    X,
    noise_model: str = ...,
    bins: int = ...,
    n_jobs: int = ...,
    verbose: int = ...,
    random_state: Incomplete | None = ...,
): ...

class FirstLevelModel(BaseGLM):
    t_r: Incomplete
    slice_time_ref: Incomplete
    hrf_model: Incomplete
    drift_model: Incomplete
    high_pass: Incomplete
    drift_order: Incomplete
    fir_delays: Incomplete
    min_onset: Incomplete
    mask_img: Incomplete
    target_affine: Incomplete
    target_shape: Incomplete
    smoothing_fwhm: Incomplete
    memory: Incomplete
    memory_level: Incomplete
    standardize: Incomplete
    signal_scaling: Incomplete
    noise_model: Incomplete
    verbose: Incomplete
    n_jobs: Incomplete
    minimize_memory: Incomplete
    labels_: Incomplete
    results_: Incomplete
    subject_label: Incomplete
    random_state: Incomplete
    def __init__(
        self,
        t_r: Incomplete | None = ...,
        slice_time_ref: float = ...,
        hrf_model: str = ...,
        drift_model: str = ...,
        high_pass: float = ...,
        drift_order: int = ...,
        fir_delays=...,
        min_onset: int = ...,
        mask_img: Incomplete | None = ...,
        target_affine: Incomplete | None = ...,
        target_shape: Incomplete | None = ...,
        smoothing_fwhm: Incomplete | None = ...,
        memory=...,
        memory_level: int = ...,
        standardize: bool = ...,
        signal_scaling: int = ...,
        noise_model: str = ...,
        verbose: int = ...,
        n_jobs: int = ...,
        minimize_memory: bool = ...,
        subject_label: Incomplete | None = ...,
        random_state: Incomplete | None = ...,
    ) -> None: ...
    @property
    def scaling_axis(self): ...
    masker_: Incomplete
    def fit(
        self,
        run_imgs,
        events: Incomplete | None = ...,
        confounds: Incomplete | None = ...,
        sample_masks: Incomplete | None = ...,
        design_matrices: Incomplete | None = ...,
        bins: int = ...,
    ): ...
    def compute_contrast(
        self,
        contrast_def,
        stat_type: Incomplete | None = ...,
        output_type: str = ...,
    ): ...

def first_level_from_bids(
    dataset_path: Union[str, pathlib.Path],
    task_label,
    space_label: Incomplete | None = ...,
    sub_labels: Incomplete | None = ...,
    img_filters: Incomplete | None = ...,
    t_r: Incomplete | None = ...,
    slice_time_ref: float = ...,
    hrf_model: str = ...,
    drift_model: str = ...,
    high_pass: float = ...,
    drift_order: int = ...,
    fir_delays=...,
    min_onset: int = ...,
    mask_img: Incomplete | None = ...,
    target_affine: Incomplete | None = ...,
    target_shape: Incomplete | None = ...,
    smoothing_fwhm: Incomplete | None = ...,
    memory=...,
    memory_level: int = ...,
    standardize: bool = ...,
    signal_scaling: int = ...,
    noise_model: str = ...,
    verbose: int = ...,
    n_jobs: int = ...,
    minimize_memory: bool = ...,
    derivatives_folder: str = ...,
): ...
def _list_valid_subjects(
    derivatives_path: str, sub_labels: list[str] | None
) -> list[str]:
    ...,

def _report_found_files(
    files: list[str], text: str, sub_label: str, filters: list[tuple[str, str]]
) -> None:
    ...,

def _get_processed_imgs(
    derivatives_path: str,
    sub_label: str,
    task_label: str,
    space_label: str,
    img_filters: list[tuple[str, str]],
    verbose: int,
) -> list[str]:
    ...,

def _get_events_files(
    dataset_path: str,
    sub_label: str,
    task_label: str,
    img_filters: list[tuple[str, str]],
    imgs: list[str],
    verbose: int,
) -> list[str]:
    ...,

def _get_confounds(
    derivatives_path: str,
    sub_label: str,
    task_label: str,
    img_filters: list[tuple[str, str]],
    imgs: list[str],
    verbose: int,
) -> Optional[list[str]]:
    ...,

def _check_confounds_list(confounds: list[str], imgs: list[str]) -> None:
    ...,

def _check_args_first_level_from_bids(
    dataset_path: str | pathlib.Path,
    task_label: str,
    space_label: str | None,
    sub_labels: list[str] | None,
    img_filters: list[tuple[str, str]],
    derivatives_folder: str,
) -> None:
    ...,

def _make_bids_files_filter(
    task_label: str,
    space_label: str | None = None,
    supported_filters: list[str] | None = None,
    extra_filter: list[tuple[str, str]] | None = None,
) -> list[tuple[str, str]]:
    ...,

def _check_bids_image_list(
    imgs: list[str] | None, sub_label: str, filters: list[tuple[str, str]]
) -> None:
    ...,

def _check_bids_events_list(
    events: list[str] | None,
    imgs: list[str],
    sub_label: str,
    task_label: str,
    dataset_path: str,
    events_filters: list[tuple[str, str]],
) -> None:
    ...,
