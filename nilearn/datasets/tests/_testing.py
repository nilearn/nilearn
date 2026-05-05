"""Utilities for testing the dataset fetchers."""

import pathlib
import pickle
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from nibabel import Nifti1Image
from sklearn.utils import Bunch

from nilearn.surface.surface import PolyMesh, SurfaceImage


def check_fetcher_verbosity(fn, capsys, **kwargs):
    """Check verbosity behavior of fetcher.

    - Default verbosity == 1
    - Verbose 0 is quiet
    """
    capsys.readouterr()  # necessary to flush what is already in system output

    fn(**kwargs)
    captured_default = capsys.readouterr().out

    fn(verbose=1, **kwargs)
    captured_verbose = capsys.readouterr().out

    assert captured_default == captured_verbose

    fn(verbose=0, **kwargs)
    captured_verbose_0 = capsys.readouterr().out
    assert captured_verbose_0 == ""


def _add_to_archive(path, content):
    path.parent.mkdir(exist_ok=True, parents=True)
    if hasattr(content, "to_filename"):
        content.to_filename(str(path))
    elif hasattr(content, "is_dir") and hasattr(content, "is_file"):
        if content.is_file():
            shutil.copy(str(content), str(path))
        elif content.is_dir():
            shutil.copytree(str(content), str(path))
        else:
            raise FileNotFoundError(
                f"Not found or not a regular file or a directory {content}"
            )
    elif isinstance(content, str):
        with path.open("w") as f:
            f.write(content)
    elif isinstance(content, bytes):
        with path.open("wb") as f:
            f.write(content)
    else:
        with path.open("wb") as f:
            pickle.dump(content, f)


def dict_to_archive(data, archive_format="gztar"):
    """Transform a {path: content} dict to an archive.

    Parameters
    ----------
    data : dict
        Keys are strings or `pathlib.Path` objects and specify paths inside the
        archive. (If strings, must use the system path separator.)
        Values determine the contents of these files and can be:
          - an object with a `to_filename` method (e.g. a Nifti1Image): it is
            serialized to .nii.gz
          - a `pathlib.Path`: the contents are copied inside the archive (can
            point to a file or a directory). (can also be anything that has
            `is_file` and `is_directory` attributes, e.g. a `pathlib2.Path`)
          - a `str` or `bytes`: the contents of the file
          - anything else is pickled.

    archive_format : str, default="gztar"
        The archive format. See `shutil` documentation for available formats.

    Returns
    -------
    bytes : the contents of the resulting archive file, to be used for example
        as the contents of a mock response object (see Sender).

    Examples
    --------
    if `data` is `{"README.txt": "hello", Path("Data") / "labels.csv": "a,b"}`,
    the resulting archive has this structure:
        .
        ├── Data
        │   └── labels.csv
        └── README.txt

    where labels.csv and README.txt contain the corresponding values in `data`

    """
    with tempfile.TemporaryDirectory() as root_tmp_dir:
        root_tmp_dir = Path(root_tmp_dir)
        tmp_dir = root_tmp_dir / "tmp"
        tmp_dir.mkdir()
        for path, content in data.items():
            _add_to_archive(tmp_dir / path, content)
        archive_path = shutil.make_archive(
            str(root_tmp_dir / "archive"), archive_format, str(tmp_dir)
        )
        with Path(archive_path).open("rb") as f:
            return f.read()


def list_to_archive(sequence, archive_format="gztar", content=""):
    """Transform a list of paths to an archive.

    This invokes dict_to_archive with the `sequence` items as keys and
    `content` (by default '') as values.

    For example, if `sequence` is
    `["README.txt", Path("Data") / "labels.csv"]`,
    the resulting archive has this structure:
        .
        ├── Data
        │   └── labels.csv
        └── README.txt

    and "labels.csv" and "README.txt" contain the value of `content`.

    """
    return dict_to_archive(
        dict.fromkeys(sequence, content), archive_format=archive_format
    )


def check_type_fetcher(data):
    """Check type content of datasets.

    Recursively checks the content returned by fetchers
    to make sure they do not contain only some allowed type of objects.

    If the data is a Bunch and contains a dataset description,
    ensures the description is not empty.
    """
    if isinstance(
        data,
        (
            str,
            int,
            float,
            Nifti1Image,
            SurfaceImage,
            pd.DataFrame,
            PolyMesh,
            pathlib.Path,
        ),
    ):
        pass
    elif isinstance(data, (Bunch, dict)):
        for k, v in data.items():
            if k == "description":
                assert isinstance(v, str)
                assert v != ""
            if not check_type_fetcher(v):
                raise TypeError(f"Found {k} : {v.__class__.__name__}")
    elif isinstance(data, (set, list, tuple)):
        for v in data:
            if not check_type_fetcher(v):
                raise TypeError(f"{type(v)}")
    else:
        return False
    return True
