import numpy as np
from _typeshed import Incomplete
from nilearn import (
    datasets as datasets,
    image as image,
    maskers as maskers,
    masking as masking,
)
from nilearn._utils import as_ndarray as as_ndarray, logger as logger
from nilearn.interfaces.bids._utils import (
    _bids_entities as _bids_entities,
    _check_bids_label as _check_bids_label,
)
from pathlib import Path
from typing import Any, Union

def generate_mni_space_img(
    n_scans: int = ...,
    res: int = ...,
    random_state: int = ...,
    mask_dilation: int = ...,
): ...
def generate_timeseries(n_timepoints, n_features, random_state: int = ...): ...
def generate_regions_ts(
    n_features,
    n_regions,
    overlap: int = ...,
    random_state: int = ...,
    window: str = ...,
): ...
def generate_maps(
    shape,
    n_regions,
    overlap: int = ...,
    border: int = ...,
    window: str = ...,
    random_state: int = ...,
    affine=...,
): ...
def generate_labeled_regions(
    shape,
    n_regions,
    random_state: int = ...,
    labels: Incomplete | None = ...,
    affine=...,
    dtype: str = ...,
): ...
def generate_fake_fmri(
    shape=...,
    length: int = ...,
    kind: str = ...,
    affine=...,
    n_blocks: Incomplete | None = ...,
    block_size: Incomplete | None = ...,
    block_type: str = ...,
    random_state: int = ...,
): ...
def generate_fake_fmri_data_and_design(
    shapes, rk: int = ..., affine=..., random_state: int = ...
): ...
def write_fake_fmri_data_and_design(
    shapes, rk: int = ..., affine=..., random_state: int = ...
): ...
def write_fake_bold_img(
    file_path, shape, affine=..., random_state: int = ...
): ...
def generate_signals_from_precisions(
    precisions,
    min_n_samples: int = ...,
    max_n_samples: int = ...,
    random_state: int = ...,
): ...
def generate_group_sparse_gaussian_graphs(
    n_subjects: int = ...,
    n_features: int = ...,
    min_n_samples: int = ...,
    max_n_samples: int = ...,
    density: float = ...,
    random_state: int = ...,
    verbose: int = ...,
): ...
def basic_paradigm(condition_names_have_spaces: bool = ...): ...
def basic_confounds(length, random_state: int = ...): ...
def generate_random_img(shape, affine=..., random_state=...): ...
def create_fake_bids_dataset(
    base_dir: Union[str, Path] = ...,
    n_sub: int = ...,
    n_ses: int = ...,
    tasks: list[str] = ...,
    n_runs: list[int] = ...,
    with_derivatives: bool = ...,
    with_confounds: bool = ...,
    confounds_tag: Union[str, None] = ...,
    no_session: bool = ...,
    random_state: int = ...,
    entities: Union[dict[str, list[str]], None] = ...,
) -> Path: ...
def _check_entities_and_labels(entities: dict) -> None: ...
def _mock_bids_dataset(
    bids_path: Path,
    n_sub: int,
    n_ses: int,
    tasks: list[str],
    n_runs: list[int],
    entities: dict[str, list[str]],
    n_voxels: int,
    rand_gen: np.random.RandomState,
) -> None: ...
def _mock_bids_derivatives(
    bids_path: Path,
    n_sub: int,
    n_ses: int,
    tasks: list[str],
    n_runs: list[int],
    confounds_tag: Union[str, None],
    entities: dict[str, list[str]],
    n_voxels: int,
    rand_gen: np.random.RandomState,
) -> None: ...
def _listify(n: int) -> list[str]: ...
def _create_bids_filename(
    fields: dict[str, Any], entities_to_include: Union[list[str], None] = ...
) -> str: ...
def _init_fields(
    subject: str, session: str, task: str, run: str
) -> dict[str, Any]: ...
def _write_bids_raw_anat(
    subses_dir: Path, subject: str, session: str
) -> None: ...
def _write_bids_raw_func(
    func_path: Path,
    fields: dict[str, Any],
    n_voxels: int,
    rand_gen: np.random.RandomState,
) -> None: ...
def _write_bids_derivative_func(
    func_path: Path,
    fields: dict[str, Any],
    n_voxels: int,
    rand_gen: np.random.RandomState,
    confounds_tag: Union[str, None],
) -> None: ...
