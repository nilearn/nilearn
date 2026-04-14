# AGENTS.md

This file provides guidance to AI coding agents working with this repository.

## What is Nilearn

Nilearn is a Python library for statistical learning on neuroimaging data. It provides tools for fMRI analysis, brain decoding, connectivity analysis, and visualization, built on top of scikit-learn.

## Git

Never commit directly to `main`. Always work on a feature branch.

Fix any remaining issues flagged by pre-commit before committing again.

## Commands

### Setup

Create a virtual environment before running any script, test, doc build...

```bash
uv sync
```

```bash
pip install -e ".[plotting]" --group dev  # Dev install with plotting dependencies
pre-commit install              # Install git hooks
```

### Testing

```bash
tox -e latest -- nilearn/                        # Run all tests
tox -e latest -- nilearn/glm/tests/              # Run a specific module's tests
tox -e latest -- nilearn/glm/tests/test_first_level_model.py  # Run a single test file
tox -e latest -- nilearn/glm/tests/test_first_level_model.py::test_name  # Run single test
tox -e latest                           # Full test suite (latest deps)
tox -e plotting                         # With plotting dependencies
tox -e min                              # Minimum supported dependencies
```

### Linting, Formatting & Architecture Validation

```bash
pre-commit run --all-files # Run all pre-commit hooks
```

### Documentation

```bash
tox -e doc                 # Build documentation
```

## Code Architecture

### Layered Import Structure

The import architecture is enforced via `import-linter` contracts defined in `pyproject.toml`. Layers must only import from lower layers:

```
Layer 5 (Top):    nilearn.glm, nilearn.decoding, nilearn.connectome, nilearn.decomposition
Layer 4:          nilearn.mass_univariate
Layer 3 (Mid):    nilearn.maskers, nilearn.image, nilearn.surface, nilearn.plotting,
                  nilearn.masking, nilearn.reporting, nilearn.datasets, nilearn.interfaces
Layer 2:          nilearn.signal
Layer 1 (Base):   nilearn.typing, nilearn.exceptions, nilearn._utils.versions
```

Violating these contracts will fail the `lint-imports` check.

### Main Modules

- **`nilearn.datasets`** — Download neuroimaging datasets and atlases
- **`nilearn.image`** — Volumetric operations (resample, crop, math on NIfTI images)
- **`nilearn.masking`** — Brain mask computation
- **`nilearn.signal`** — Time-series preprocessing (confound removal, filtering, detrending)
- **`nilearn.maskers`** — Scikit-learn–compatible region extractors (`NiftiMasker`, `NiftiLabelsMasker`, `NiftiSpheresMasker`, surface variants, etc.)
- **`nilearn.surface`** — Surface mesh I/O (`SurfaceImage`, `SurfaceMesh`, `vol_to_surf`)
- **`nilearn.glm`** — General Linear Models: `first_level` (single-subject) and `second_level` (group)
- **`nilearn.decoding`** — ML decoders (`Decoder`, `SearchLight`, `SpaceNet*`)
- **`nilearn.connectome`** — Functional connectivity (`ConnectivityMeasure`, `GroupSparseCovariance`)
- **`nilearn.decomposition`** — `CanICA`, `DictLearning`
- **`nilearn.mass_univariate`** — Permutation-based mass univariate testing
- **`nilearn.regions`** — Region extraction, parcellation, clustering
- **`nilearn.plotting`** — Static and interactive brain visualization (matplotlib + plotly)
- **`nilearn.reporting`** — HTML analysis reports
- **`nilearn.interfaces`** — fMRIPrep/BIDS pipeline integration
- **`nilearn.utils`** — Utilities for nilearn users.
- **`nilearn._utils`** — Internal utilities (not public API); `data_gen` for test fixtures

### Key Design Patterns

**Scikit-learn API**: All estimators implement `fit()`. Many implement `transform()` (or `fit_transform()`). Fitted attributes have trailing underscores (e.g., `masker.mask_img_`). See `nilearn._utils.estimator_checks` for sklearn compatibility checks and additional systematic checks to apply to all Nilearn estimators.

**Maskers as transformers**: Maskers extract signals from brain images into 2D arrays (samples × features) suitable for sklearn pipelines.

**Caching**: `CacheMixin` (from `nilearn._utils.cache_mixin`) provides joblib-based caching for expensive computations and downloads.

**BIDS support**: The `interfaces` and `datasets` modules handle BIDS file conventions.

**Surface vs. Volume**: The library handles both 3D/4D NIfTI volumes and surface meshes (`.gii`). Surface equivalents exist for most maskers and plotters.

## Coding Conventions

- **Style**: PEP8, ruff-enforced, 79-char line limit, double quotes
- **Docstrings**: NumPy format (use `.. nilearn_versionadded::` and `.. nilearn_versionchanged::` for API changes)
- **Naming**: `snake_case` for functions/variables, `CamelCase` for classes, leading `_` for private
- **Imports**: Absolute imports only; no new top-level dependencies without discussion
- **Backward compatibility**: Required; use deprecation warnings before removing features

## Test Conventions

- Test data generated with `nilearn._utils.data_gen` (no real data in tests) or via fixtures most often stored in a `conftest.py` file: run `pytest nilearn --fixtures` to get a list of all available fixtures.
- Markers:
  - `@pytest.mark.slow`: for tests that exceed the timeout allowed for each test
  - `@pytest.mark.single_process`: for test that require testing n_jobs>1
  - `@pytest.mark.thread_unsafe`: marker used by pytest-run-parallel
- Keep tests fast (mocked/synthetic data).
- `xfail_strict = true` — unexpected passes are errors

## Changelog

Each PR must add an entry to `doc/changes/latest.rst` with a badge, PR link, and author. Badge types:

- :bdg-primary:`Doc`
- :bdg-secondary:`Maint`
- :bdg-success:`API`
- :bdg-info:`Plotting`
- :bdg-warning:`Test`
- :bdg-danger:`Deprecation`
- :bdg-dark:`Code`

## PR Tags

Prefix PR titles with: `[FIX]`, `[ENH]`, `[DOC]`, `[MAINT]`, `[WIP]`.

## CI Commit Message Controls

- `[skip test]` — Skip test workflow
- `[skip doc]` — Skip doc build
- `[full doc]` — Force full doc build on a PR
- `[example] name.py` — Build a specific gallery example
