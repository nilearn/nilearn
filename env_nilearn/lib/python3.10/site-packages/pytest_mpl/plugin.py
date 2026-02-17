# Copyright (c) 2015, Thomas P. Robitaille
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# The code below includes code adapted from WCSAxes, which is released
# under a 3-clause BSD license and can be found here:
#
#   https://github.com/astrofrog/wcsaxes

import io
import os
import json
import uuid
import shutil
import hashlib
import logging
import tempfile
import warnings
import contextlib
from pathlib import Path
from urllib.request import urlopen

import pytest
from packaging.version import Version

from pytest_mpl.summary.html import generate_summary_basic_html, generate_summary_html

DEFAULT_STYLE = "classic"
DEFAULT_TOLERANCE = 2
DEFAULT_BACKEND = "agg"

SUPPORTED_FORMATS = {"html", "json", "basic-html"}

SHAPE_MISMATCH_ERROR = """Error: Image dimensions did not match.
  Expected shape: {expected_shape}
    {expected_path}
  Actual shape: {actual_shape}
    {actual_path}"""

PYTEST_LT_7 = Version(pytest.__version__) < Version("7.0.0")

# The following are the subsets of formats supported by the Matplotlib image
# comparison machinery
RASTER_IMAGE_FORMATS = ['png']
VECTOR_IMAGE_FORMATS = ['eps', 'pdf', 'svg']
ALL_IMAGE_FORMATS = RASTER_IMAGE_FORMATS + VECTOR_IMAGE_FORMATS


def _get_item_dir(item):
    path = Path(item.fspath) if PYTEST_LT_7 else item.path
    return path.parent


def _hash_file(in_stream):
    """
    Hashes an already opened file.
    """
    in_stream.seek(0)
    buf = in_stream.read()
    hasher = hashlib.sha256()
    hasher.update(buf)
    return hasher.hexdigest()


def pathify(path):
    """
    Remove non-path safe characters.
    """
    path = Path(path)
    ext = ''
    if path.suffixes[-1][1:] in ALL_IMAGE_FORMATS:
        ext = path.suffixes[-1]
        path = str(path).split(ext)[0]
    path = str(path)
    path = path.replace('[', '_').replace(']', '_')
    path = path.replace('/', '_')
    if path.endswith('_'):
        path = path[:-1]
    return Path(path + ext)


def generate_test_name(item):
    """
    Generate a unique name for the hash for this test.
    """
    if item.cls is not None:
        name = f"{item.module.__name__}.{item.cls.__name__}.{item.name}"
    else:
        name = f"{item.module.__name__}.{item.name}"
    return name


def wrap_figure_interceptor(plugin, item):
    """
    Intercept and store figures returned by test functions.
    """
    # Only intercept figures on marked figure tests
    if get_compare(item) is not None:

        # Use the full test name as a key to ensure correct figure is being retrieved
        test_name = generate_test_name(item)

        def figure_interceptor(store, obj):
            def wrapper(*args, **kwargs):
                store.return_value[test_name] = obj(*args, **kwargs)
            return wrapper

        item.obj = figure_interceptor(plugin, item.obj)


def pytest_report_header():
    import matplotlib
    import matplotlib.ft2font
    return ["Matplotlib: {0}".format(matplotlib.__version__),
            "Freetype: {0}".format(matplotlib.ft2font.__freetype_version__)]


def pytest_addoption(parser):
    group = parser.getgroup("matplotlib image comparison")

    msg = "Enable comparison of matplotlib figures to reference files"
    group.addoption("--mpl", help=msg, action="store_true")

    msg = "directory to generate reference images in, relative to location where py.test is run"
    group.addoption("--mpl-generate-path", help=msg, action="store")

    msg = "filepath to save a generated hash library, relative to location where py.test is run"
    group.addoption("--mpl-generate-hash-library", help=msg, action="store")

    msg = (
        "directory containing baseline images, relative to "
        "location where py.test is run unless --mpl-baseline-relative is given. "
        "This can also be a URL or a set of comma-separated URLs (in case "
        "mirrors are specified)"
    )
    option = "mpl-baseline-path"
    group.addoption(f"--{option}", help=msg, action="store")
    parser.addini(option, help=msg)

    msg = "interpret the baseline directory as relative to the test location."
    group.addoption("--mpl-baseline-relative", help=msg, action="store_true")

    msg = "json library of image hashes, relative to location where py.test is run"
    option = "mpl-hash-library"
    group.addoption(f"--{option}", help=msg, action="store")
    parser.addini(option, help=msg)

    msg = (
        "Generate a summary report of any failed tests"
        ", in --mpl-results-path. The type of the report should be "
        "specified. Supported types are `html`, `json` and `basic-html`. "
        "Multiple types can be specified separated by commas."
    )
    option = "mpl-generate-summary"
    group.addoption(f"--{option}", help=msg, action="store")
    parser.addini(option, help=msg)

    msg = "directory for test results, relative to location where py.test is run"
    option = "mpl-results-path"
    group.addoption(f"--{option}", help=msg, action="store")
    parser.addini(option, help=msg)

    msg = (
        "Always compare to baseline images and save result images, even for passing tests. "
        "This option is automatically applied when generating a HTML summary."
    )
    option = "mpl-results-always"
    group.addoption(f"--{option}", help=msg, action="store_true")
    parser.addini(option, help=msg)

    msg = "use fully qualified test name as the filename."
    option = "mpl-use-full-test-name"
    group.addoption(f"--{option}", help=msg, action="store_true")
    parser.addini(option, help=msg, type="bool")

    msg = "default style to use for tests, unless specified in the mpl_image_compare decorator"
    option = "mpl-default-style"
    group.addoption(f"--{option}", help=msg, action="store")
    parser.addini(option, help=msg)

    msg = "default tolerance to use for tests, unless specified in the mpl_image_compare decorator"
    option = "mpl-default-tolerance"
    group.addoption(f"--{option}", help=msg, action="store")
    parser.addini(option, help=msg)

    msg = "whether to make the image file metadata deterministic"
    option_true = "mpl-deterministic"
    option_false = "mpl-no-deterministic"
    group.addoption(f"--{option_true}", help=msg, action="store_true")
    group.addoption(f"--{option_false}", help=msg, action="store_true")
    parser.addini(option_true, help=msg, type="bool", default=None)

    msg = "default backend to use for tests, unless specified in the mpl_image_compare decorator"
    option = "mpl-default-backend"
    group.addoption(f"--{option}", help=msg, action="store")
    parser.addini(option, help=msg)


class XdistPlugin:
    def pytest_configure_node(self, node):
        node.workerinput["pytest_mpl_uid"] = node.config.pytest_mpl_uid
        node.workerinput["pytest_mpl_results_dir"] = node.config.pytest_mpl_results_dir


def pytest_configure(config):

    config.addinivalue_line(
        "markers",
        "mpl_image_compare: Compares matplotlib figures against a baseline image",
    )

    if (
        config.getoption("--mpl")
        or config.getoption("--mpl-generate-path") is not None
        or config.getoption("--mpl-generate-hash-library") is not None
    ):

        def get_cli_or_ini(name, default=None):
            return config.getoption(f"--{name}") or config.getini(name) or default

        generate_dir = config.getoption("--mpl-generate-path")
        generate_hash_lib = config.getoption("--mpl-generate-hash-library")

        baseline_dir = get_cli_or_ini("mpl-baseline-path")
        if config.getoption("--mpl-baseline-relative"):
            baseline_relative_dir = config.getoption("--mpl-baseline-path")
        else:
            baseline_relative_dir = None
        use_full_test_name = get_cli_or_ini("mpl-use-full-test-name")

        hash_library = get_cli_or_ini("mpl-hash-library")
        _hash_library_from_cli = bool(config.getoption("--mpl-hash-library"))  # for backwards compatibility

        default_tolerance = get_cli_or_ini("mpl-default-tolerance", DEFAULT_TOLERANCE)
        if isinstance(default_tolerance, str):
            if default_tolerance.isdigit():  # prefer int if possible
                default_tolerance = int(default_tolerance)
            else:
                default_tolerance = float(default_tolerance)

        deterministic_ini = config.getini("mpl-deterministic")
        deterministic_flag_true = config.getoption("--mpl-deterministic")
        deterministic_flag_false = config.getoption("--mpl-no-deterministic")
        if deterministic_flag_true and deterministic_flag_false:
            raise ValueError("Only one of `--mpl-deterministic` and `--mpl-no-deterministic` can be set.")
        if deterministic_flag_true:
            deterministic = True
        elif deterministic_flag_false:
            deterministic = False
        elif isinstance(deterministic_ini, bool):
            deterministic = deterministic_ini
        else:
            deterministic = None

        default_style = get_cli_or_ini("mpl-default-style", DEFAULT_STYLE)
        default_backend = get_cli_or_ini("mpl-default-backend", DEFAULT_BACKEND)

        results_dir = get_cli_or_ini("mpl-results-path")
        results_always = get_cli_or_ini("mpl-results-always")
        generate_summary = get_cli_or_ini("mpl-generate-summary")

        if generate_dir is not None:
            if baseline_dir is not None:
                warnings.warn("Ignoring --mpl-baseline-path since --mpl-generate-path is set")

        if baseline_dir is not None and not baseline_dir.startswith(("https", "http")):
            baseline_dir = os.path.abspath(baseline_dir)
        if generate_dir is not None:
            baseline_dir = os.path.abspath(generate_dir)
        if results_dir is not None:
            results_dir = os.path.abspath(results_dir)
        if hash_library is not None:
            # For backwards compatibility, don't make absolute if set via CLI option
            if not _hash_library_from_cli:
                hash_library = os.path.abspath(hash_library)

        if not hasattr(config, "workerinput"):
            uid = uuid.uuid4().hex
            results_dir_path = results_dir or tempfile.mkdtemp()
            config.pytest_mpl_uid = uid
            config.pytest_mpl_results_dir = results_dir_path

        if config.pluginmanager.hasplugin("xdist"):
            config.pluginmanager.register(XdistPlugin(), name="pytest_mpl_xdist_plugin")

        plugin = ImageComparison(
            config,
            baseline_dir=baseline_dir,
            baseline_relative_dir=baseline_relative_dir,
            generate_dir=generate_dir,
            hash_library=hash_library,
            generate_hash_library=generate_hash_lib,
            generate_summary=generate_summary,
            results_always=results_always,
            use_full_test_name=use_full_test_name,
            default_style=default_style,
            default_tolerance=default_tolerance,
            deterministic=deterministic,
            default_backend=default_backend,
            _hash_library_from_cli=_hash_library_from_cli,
        )
        config.pluginmanager.register(plugin)

    else:
        config.pluginmanager.register(FigureCloser(config))


@contextlib.contextmanager
def switch_backend(backend):
    import matplotlib
    import matplotlib.pyplot as plt
    prev_backend = matplotlib.get_backend().lower()
    if prev_backend != backend.lower():
        plt.switch_backend(backend)
        yield
        plt.switch_backend(prev_backend)
    else:
        yield


def close_mpl_figure(fig):
    "Close a given matplotlib Figure. Any other type of figure is ignored"

    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    # We only need to close actual Matplotlib figure objects. If
    # we are dealing with a figure-like object that provides
    # savefig but is not a real Matplotlib object, we shouldn't
    # try closing it here.
    if isinstance(fig, Figure):
        plt.close(fig)


def get_compare(item):
    """
    Return the mpl_image_compare marker for the given item.
    """
    return item.get_closest_marker("mpl_image_compare")


def path_is_not_none(apath):
    return Path(apath) if apath is not None else apath


class ImageComparison:
    def __init__(
        self,
        config,
        baseline_dir=None,
        baseline_relative_dir=None,
        generate_dir=None,
        hash_library=None,
        generate_hash_library=None,
        generate_summary=None,
        results_always=False,
        use_full_test_name=False,
        default_style=DEFAULT_STYLE,
        default_tolerance=DEFAULT_TOLERANCE,
        deterministic=None,
        default_backend=DEFAULT_BACKEND,
        _hash_library_from_cli=False,  # for backwards compatibility
    ):
        self.config = config
        self.baseline_dir = baseline_dir
        self.baseline_relative_dir = path_is_not_none(baseline_relative_dir)
        self.generate_dir = path_is_not_none(generate_dir)
        self.results_dir = None
        self.hash_library = path_is_not_none(hash_library)
        self._hash_library_from_cli = _hash_library_from_cli  # for backwards compatibility
        self.generate_hash_library = path_is_not_none(generate_hash_library)
        if generate_summary:
            generate_summary = {i.lower() for i in generate_summary.split(',')}
            unsupported_formats = generate_summary - SUPPORTED_FORMATS
            if len(unsupported_formats) > 0:
                raise ValueError(f"The mpl summary type(s) '{sorted(unsupported_formats)}' "
                                 "are not supported.")
            # When generating HTML always apply `results_always`
            if generate_summary & {'html', 'basic-html'}:
                results_always = True
        self.generate_summary = generate_summary
        self.results_always = results_always
        self.use_full_test_name = use_full_test_name

        self.default_style = default_style
        self.default_tolerance = default_tolerance
        self.deterministic = deterministic
        self.default_backend = default_backend

        # Decide what to call the downloadable results hash library
        if self.hash_library is not None:
            self.results_hash_library_name = self.hash_library.name
        else:  # Use the first filename encountered in a `hash_library=` kwarg
            self.results_hash_library_name = None

        # We need global state to store all the hashes generated over the run
        self._generated_hash_library = {}
        self._test_results = {}
        self._test_stats = None
        self.return_value = {}

    def pytest_sessionstart(self, session):
        config = session.config
        if hasattr(config, "workerinput"):
            config.pytest_mpl_uid = config.workerinput["pytest_mpl_uid"]
            config.pytest_mpl_results_dir = config.workerinput["pytest_mpl_results_dir"]
        self.results_dir = Path(config.pytest_mpl_results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_logger(self):
        # configure a separate logger for this pluggin which is independent
        # of the options that are configured for pytest or for the code that
        # is tested; turn debug prints on only if "-vv" or more passed
        level = logging.DEBUG if self.config.option.verbose > 1 else logging.INFO
        if self.config.option.log_cli_format is not None:
            fmt = self.config.option.log_cli_format
        else:
            # use pytest's default fmt
            fmt = "%(levelname)-8s %(name)s:%(filename)s:%(lineno)d %(message)s"
        formatter = logging.Formatter(fmt)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger = logging.getLogger('pytest-mpl')
        logger.propagate = False
        logger.setLevel(level)
        logger.addHandler(handler)
        return logger

    def _file_extension(self, item):
        compare = get_compare(item)
        savefig_kwargs = compare.kwargs.get('savefig_kwargs', {})
        return savefig_kwargs.get('format', 'png')

    def generate_filename(self, item):
        """
        Given a pytest item, generate the figure filename.
        """
        ext = self._file_extension(item)
        if self.use_full_test_name:
            filename = generate_test_name(item) + f'.{ext}'
        else:
            compare = get_compare(item)
            # Find test name to use as plot name
            filename = compare.kwargs.get('filename', None)
            if filename is None:
                filename = item.name + f'.{ext}'

        filename = str(pathify(filename))
        return filename

    def make_test_results_dir(self, item):
        """
        Generate the directory to put the results in.
        """
        test_name = pathify(generate_test_name(item))
        results_dir = self.results_dir / test_name
        results_dir.mkdir(exist_ok=True, parents=True)
        return results_dir

    def baseline_directory_specified(self, item):
        """
        Returns `True` if a non-default baseline directory is specified.
        """
        compare = get_compare(item)
        item_baseline_dir = compare.kwargs.get('baseline_dir', None)
        return item_baseline_dir or self.baseline_dir or self.baseline_relative_dir

    def get_baseline_directory(self, item):
        """
        Return a full path to the baseline directory, either local or remote.

        Using the global and per-test configuration return the absolute
        baseline dir, if the baseline file is local else return base URL.
        """
        compare = get_compare(item)
        baseline_dir = compare.kwargs.get('baseline_dir', None)
        if baseline_dir is None:
            if self.baseline_dir is None:
                baseline_dir = _get_item_dir(item) / 'baseline'
            else:
                if self.baseline_relative_dir:
                    # baseline dir is relative to the current test
                    baseline_dir = _get_item_dir(item) / self.baseline_relative_dir
                else:
                    # baseline dir is relative to where pytest was run
                    baseline_dir = self.baseline_dir

        baseline_remote = (isinstance(baseline_dir, str) and  # noqa
                           baseline_dir.startswith(('http://', 'https://')))
        if not baseline_remote:
            return _get_item_dir(item) / baseline_dir

        return baseline_dir

    def _download_file(self, baseline, filename):
        # Note that baseline can be a comma-separated list of URLs that we can
        # then treat as mirrors
        for base_url in baseline.split(','):
            try:
                u = urlopen(base_url + filename)
                content = u.read()
            except Exception as e:
                self.get_logger().info(f'Downloading {base_url + filename} failed: {repr(e)}')
            else:
                break
        else:  # Could not download baseline image from any of the available URLs
            return
        result_dir = Path(tempfile.mkdtemp())
        filename = result_dir / 'downloaded'
        with open(str(filename), 'wb') as tmpfile:
            tmpfile.write(content)
        return Path(filename)

    def obtain_baseline_image(self, item):
        """
        Copy the baseline image to our working directory.

        If the image is remote it is downloaded, if it is local it is copied to
        ensure it is kept in the event of a test failure.
        """
        filename = self.generate_filename(item)
        baseline_dir = self.get_baseline_directory(item)
        baseline_remote = (isinstance(baseline_dir, str) and  # noqa
                           baseline_dir.startswith(('http://', 'https://')))
        if baseline_remote:
            # baseline_dir can be a list of URLs when remote, so we have to
            # pass base and filename to download
            baseline_image = self._download_file(baseline_dir, filename)
        else:
            baseline_image = (baseline_dir / filename).absolute()

        return baseline_image

    def generate_baseline_image(self, item, fig):
        """
        Generate reference figures.
        """

        if not os.path.exists(self.generate_dir):
            os.makedirs(self.generate_dir)

        baseline_filename = self.generate_filename(item)
        baseline_path = (self.generate_dir / baseline_filename).absolute()
        self.save_figure(item, fig, baseline_path)
        close_mpl_figure(fig)

        return baseline_path

    def generate_image_hash(self, item, fig):
        """
        For a `matplotlib.figure.Figure`, returns the SHA256 hash as a hexadecimal
        string.
        """

        imgdata = io.BytesIO()
        self.save_figure(item, fig, imgdata)
        out = _hash_file(imgdata)
        imgdata.close()

        close_mpl_figure(fig)
        return out

    def compare_image_to_baseline(self, item, fig, result_dir, summary=None):
        """
        Compare a test image to a baseline image.
        """
        from matplotlib.image import imread
        from matplotlib.testing.compare import compare_images

        if summary is None:
            summary = {}

        compare = get_compare(item)
        tolerance = compare.kwargs.get('tolerance', self.default_tolerance)

        ext = self._file_extension(item)

        test_image = (result_dir / f"result.{ext}").absolute()
        self.save_figure(item, fig, test_image)

        if ext in ['png', 'svg']:  # Use original file
            summary['result_image'] = test_image.relative_to(self.results_dir).as_posix()
        else:
            summary['result_image'] = (result_dir / f"result_{ext}.png").relative_to(self.results_dir).as_posix()

        baseline_image_ref = self.obtain_baseline_image(item)

        baseline_missing = None
        if baseline_image_ref is None:
            baseline_missing = ("Could not download the baseline image from "
                                "any of the available URLs.\n")
        elif not os.path.exists(baseline_image_ref):
            baseline_missing = ("Image file not found for comparison test in: \n\t"
                                f"{self.get_baseline_directory(item)}\n")

        if baseline_missing:
            summary['status'] = 'failed'
            summary['image_status'] = 'missing'
            error_message = (baseline_missing +
                             "(This is expected for new tests.)\n"
                             "Generated Image: \n\t"
                             f"{test_image}")
            summary['status_msg'] = error_message
            return error_message

        # setuptools may put the baseline images in non-accessible places,
        # copy to our tmpdir to be sure to keep them in case of failure
        baseline_image = (result_dir / f"baseline.{ext}").absolute()
        shutil.copyfile(baseline_image_ref, baseline_image)

        if ext in ['png', 'svg']:  # Use original file
            summary['baseline_image'] = baseline_image.relative_to(self.results_dir).as_posix()
        else:
            summary['baseline_image'] = (result_dir / f"baseline_{ext}.png").relative_to(self.results_dir).as_posix()

        # Compare image size ourselves since the Matplotlib
        # exception is a bit cryptic in this case and doesn't show
        # the filenames. However imread won't work for vector graphics so we
        # only do this for raster files.
        if ext in RASTER_IMAGE_FORMATS:
            expected_shape = imread(str(baseline_image)).shape[:2]
            actual_shape = imread(str(test_image)).shape[:2]
            if expected_shape != actual_shape:
                summary['status'] = 'failed'
                summary['image_status'] = 'diff'
                error_message = SHAPE_MISMATCH_ERROR.format(expected_path=baseline_image,
                                                            expected_shape=expected_shape,
                                                            actual_path=test_image,
                                                            actual_shape=actual_shape)
                summary['status_msg'] = error_message
                return error_message

        results = compare_images(str(baseline_image), str(test_image), tol=tolerance, in_decorator=True)

        summary['tolerance'] = tolerance
        if results is None:
            summary['status'] = 'passed'
            summary['image_status'] = 'match'
            summary['status_msg'] = 'Image comparison passed.'
            return None
        else:
            summary['status'] = 'failed'
            summary['image_status'] = 'diff'
            summary['rms'] = results['rms']
            summary['diff_image'] = Path(results['diff']).relative_to(self.results_dir).as_posix()
            template = ['Error: Image files did not match.',
                        'RMS Value: {rms}',
                        'Expected:  \n    {expected}',
                        'Actual:    \n    {actual}',
                        'Difference:\n    {diff}',
                        'Tolerance: \n    {tol}', ]
            error_message = '\n  '.join([line.format(**results) for line in template])
            summary['status_msg'] = error_message
            return error_message

    def load_hash_library(self, library_path):
        with open(str(library_path)) as fp:
            return json.load(fp)

    def save_figure(self, item, fig, filename):
        if isinstance(filename, Path):
            filename = str(filename)
        compare = get_compare(item)
        savefig_kwargs = compare.kwargs.get('savefig_kwargs', {})
        deterministic = compare.kwargs.get('deterministic', self.deterministic)

        original_source_date_epoch = os.environ.get('SOURCE_DATE_EPOCH', None)

        extra_rcparams = {}

        ext = self._file_extension(item)

        if deterministic is None:

            # The deterministic option should only matter for hash-based tests,
            # so we first check if a hash library is being used

            if self.hash_library or compare.kwargs.get('hash_library', None):

                if ext == 'png':
                    if 'metadata' not in savefig_kwargs or 'Software' not in savefig_kwargs['metadata']:
                        warnings.warn("deterministic option not set (currently defaulting to False), "
                                      "in future this will default to True to give consistent "
                                      "hashes across Matplotlib versions. To suppress this warning, "
                                      "set deterministic to True if you are happy with the future "
                                      "behavior or to False if you want to preserve the old behavior.",
                                      FutureWarning)
                    else:
                        # Set to False but in practice because Software is set to a constant value
                        # by the caller, the output will be deterministic (we don't want to change
                        # Software to None if the caller set it to e.g. 'test')
                        deterministic = False
                else:
                    deterministic = True

            else:

                # We can just default to True since it shouldn't matter and in
                # case generated images are somehow used in future to compute
                # hashes

                deterministic = True

        if deterministic:

            # Make sure we don't modify the original dictionary in case is a common
            # object used by different tests
            savefig_kwargs = savefig_kwargs.copy()

            if 'metadata' not in savefig_kwargs:
                savefig_kwargs['metadata'] = {}

            if ext == 'png':
                extra_metadata = {"Software": None}
            elif ext == 'pdf':
                extra_metadata = {"Creator": None, "Producer": None, "CreationDate": None}
            elif ext == 'eps':
                extra_metadata = {"Creator": "test"}
                os.environ['SOURCE_DATE_EPOCH'] = '1680254601'
            elif ext == 'svg':
                extra_metadata = {"Date": None}
                extra_rcparams["svg.hashsalt"] = "test"

            savefig_kwargs['metadata'].update(extra_metadata)

        import matplotlib.pyplot as plt

        with plt.rc_context(rc=extra_rcparams):
            fig.savefig(filename, **savefig_kwargs)

        if original_source_date_epoch is not None:
            os.environ['SOURCE_DATE_EPOCH'] = original_source_date_epoch

    def compare_image_to_hash_library(self, item, fig, result_dir, summary=None):
        hash_comparison_pass = False
        if summary is None:
            summary = {}

        compare = get_compare(item)

        ext = self._file_extension(item)

        if not self.results_hash_library_name:
            # Use hash library name of current test as results hash library name
            self.results_hash_library_name = Path(compare.kwargs.get("hash_library", "")).name

        # Order of precedence for hash library: CLI, kwargs, INI (for backwards compatibility)
        hash_library_filename = compare.kwargs.get("hash_library", None) or self.hash_library
        if self._hash_library_from_cli:  # for backwards compatibility
            hash_library_filename = self.hash_library
        hash_library_filename = _get_item_dir(item) / hash_library_filename

        if not Path(hash_library_filename).exists():
            pytest.fail(f"Can't find hash library at path {hash_library_filename}")

        hash_library = self.load_hash_library(hash_library_filename)
        hash_name = generate_test_name(item)
        baseline_hash = hash_library.get(hash_name, None)
        summary['baseline_hash'] = baseline_hash

        test_hash = self.generate_image_hash(item, fig)
        summary['result_hash'] = test_hash

        if baseline_hash is None:  # hash-missing
            summary['status'] = 'failed'
            summary['hash_status'] = 'missing'
            summary['status_msg'] = (f"Hash for test '{hash_name}' not found in {hash_library_filename}. "
                                     f"Generated hash is {test_hash}.")
        elif test_hash == baseline_hash:  # hash-match
            hash_comparison_pass = True
            summary['status'] = 'passed'
            summary['hash_status'] = 'match'
            summary['status_msg'] = 'Test hash matches baseline hash.'
        else:  # hash-diff
            summary['status'] = 'failed'
            summary['hash_status'] = 'diff'
            summary['status_msg'] = (f"Hash {test_hash} doesn't match hash "
                                     f"{baseline_hash} in library "
                                     f"{hash_library_filename} for test {hash_name}.")

        # Save the figure for later summary (will be removed later if not needed)
        test_image = (result_dir / f"result.{ext}").absolute()
        self.save_figure(item, fig, test_image)
        summary['result_image'] = test_image.relative_to(self.results_dir).as_posix()

        # Hybrid mode (hash and image comparison)
        if self.baseline_directory_specified(item):

            # Skip image comparison if hash matches (unless `--mpl-results-always`)
            if hash_comparison_pass and not self.results_always:
                return

            # Run image comparison
            baseline_summary = {}  # summary for image comparison to merge with hash comparison summary
            try:  # Ignore all errors as success does not influence the overall test result
                baseline_comparison = self.compare_image_to_baseline(item, fig, result_dir,
                                                                     summary=baseline_summary)
            except Exception as baseline_error:  # Append to test error later
                summary['image_status'] = 'diff'  # (not necessarily diff, but makes user aware)
                baseline_comparison = str(baseline_error)
            else:  # Update main summary
                for k in ['image_status', 'baseline_image', 'diff_image',
                          'rms', 'tolerance', 'result_image']:
                    summary[k] = summary[k] or baseline_summary.get(k)

            # Append the log from image comparison
            r = baseline_comparison or "The comparison to the baseline image succeeded."
            summary['status_msg'] += ("\n\n"
                                      "Image comparison test\n"
                                      "---------------------\n") + r

        if hash_comparison_pass:  # Return None to indicate test passed
            return
        return summary['status_msg']

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):  # noqa

        compare = get_compare(item)

        if compare is None:
            yield
            return

        import matplotlib.pyplot as plt
        try:
            from matplotlib.testing.decorators import remove_ticks_and_titles
        except ImportError:
            from matplotlib.testing.decorators import ImageComparisonTest as MplImageComparisonTest
            remove_ticks_and_titles = MplImageComparisonTest.remove_text

        style = compare.kwargs.get('style', self.default_style)
        remove_text = compare.kwargs.get('remove_text', False)
        backend = compare.kwargs.get('backend', self.default_backend)

        ext = self._file_extension(item)

        with plt.style.context(style, after_reset=True), switch_backend(backend):

            test_name = generate_test_name(item)

            # Store fallback summary in case of exceptions
            summary = {
                'status': 'failed',
                'image_status': None,
                'hash_status': None,
                'status_msg': 'An exception was raised while testing the figure.',
                'baseline_image': None,
                'diff_image': None,
                'rms': None,
                'tolerance': None,
                'result_image': None,
                'baseline_hash': None,
                'result_hash': None,
            }
            self._test_results[test_name] = summary

            # Run test and get figure object
            wrap_figure_interceptor(self, item)

            # See https://github.com/pytest-dev/pytest/issues/11714
            result = yield
            try:
                if test_name not in self.return_value:
                    # Test function did not complete successfully
                    summary['status'] = 'failed'
                    summary['status_msg'] = ('Test function raised an exception '
                                             'before returning a figure.')
                    self._test_results[test_name] = summary
                    return
                fig = self.return_value[test_name]

                if remove_text:
                    remove_ticks_and_titles(fig)

                result_dir = self.make_test_results_dir(item)

                # What we do now depends on whether we are generating the
                # reference images or simply running the test.
                if self.generate_dir is not None:
                    summary['status'] = 'skipped'
                    summary['image_status'] = 'generated'
                    summary['status_msg'] = 'Skipped test, since generating image.'
                    generate_image = self.generate_baseline_image(item, fig)
                    if self.results_always:  # Make baseline image available in HTML
                        result_image = (result_dir / f"baseline.{ext}").absolute()
                        shutil.copy(generate_image, result_image)
                        summary['baseline_image'] = \
                            result_image.relative_to(self.results_dir).as_posix()

                if self.generate_hash_library is not None:
                    summary['hash_status'] = 'generated'
                    image_hash = self.generate_image_hash(item, fig)
                    self._generated_hash_library[test_name] = image_hash
                    summary['baseline_hash'] = image_hash

                # Only test figures if not generating images
                if self.generate_dir is None:
                    # Compare to hash library
                    if self.hash_library or compare.kwargs.get('hash_library', None):
                        msg = self.compare_image_to_hash_library(item, fig, result_dir, summary=summary)

                    # Compare against a baseline if specified
                    else:
                        msg = self.compare_image_to_baseline(item, fig, result_dir, summary=summary)

                    close_mpl_figure(fig)

                    if msg is None:
                        if not self.results_always:
                            shutil.rmtree(result_dir)
                            for image_type in ['baseline_image', 'diff_image', 'result_image']:
                                summary[image_type] = None  # image no longer exists
                    else:
                        self._test_results[test_name] = summary
                        pytest.fail(msg, pytrace=False)

                close_mpl_figure(fig)

                self._test_results[test_name] = summary

                if summary['status'] == 'skipped':
                    pytest.skip(summary['status_msg'])
            except BaseException as e:
                if hasattr(result, "force_exception"):  # pluggy>=1.2.0
                    result.force_exception(e)
                else:
                    result._result = None
                    result._excinfo = (type(e), e, e.__traceback__)

    def generate_hash_library_json(self):
        if hasattr(self.config, "workerinput"):
            uid = self.config.pytest_mpl_uid
            worker_id = os.environ.get("PYTEST_XDIST_WORKER")
            json_file = self.results_dir / f"generated-hashes-xdist-{uid}-{worker_id}.json"
        else:
            json_file = Path(self.config.rootdir) / self.generate_hash_library
            json_file.parent.mkdir(parents=True, exist_ok=True)
        with open(json_file, 'w') as f:
            json.dump(self._generated_hash_library, f, indent=2)
        return json_file

    def generate_summary_json(self):
        filename = "results.json"
        if hasattr(self.config, "workerinput"):
            uid = self.config.pytest_mpl_uid
            worker_id = os.environ.get("PYTEST_XDIST_WORKER")
            filename = f"results-xdist-{uid}-{worker_id}.json"
        json_file = self.results_dir / filename
        with open(json_file, 'w') as f:
            json.dump(self._test_results, f, indent=2)
        return json_file

    def pytest_sessionfinish(self, session):
        """
        Save out the hash library at the end of the run.
        """
        config = session.config
        is_xdist_worker = hasattr(config, "workerinput")
        is_xdist_controller = (
                config.pluginmanager.hasplugin("xdist")
                and not is_xdist_worker
                and getattr(config.option, "dist", "") != "no"
        )

        if is_xdist_controller:  # Merge results from workers
            uid = config.pytest_mpl_uid
            for worker_hashes in self.results_dir.glob(f"generated-hashes-xdist-{uid}-*.json"):
                with worker_hashes.open() as f:
                    self._generated_hash_library.update(json.load(f))
            for worker_results in self.results_dir.glob(f"results-xdist-{uid}-*.json"):
                with worker_results.open() as f:
                    self._test_results.update(json.load(f))

        result_hash_library = self.results_dir / (self.results_hash_library_name or "temp.json")
        if self.generate_hash_library is not None:
            hash_library_path = self.generate_hash_library_json()
            if self.results_always and not is_xdist_worker:  # Make accessible in results directory
                # Use same name as generated
                result_hash_library = self.results_dir / hash_library_path.name
                shutil.copy(hash_library_path, result_hash_library)
        elif self.results_always and self.results_hash_library_name and not is_xdist_worker:
            result_hashes = {k: v['result_hash'] for k, v in self._test_results.items()
                             if v['result_hash']}
            if len(result_hashes) > 0:  # At least one hash comparison test
                with open(result_hash_library, "w") as fp:
                    json.dump(result_hashes, fp, indent=2)

        if self.generate_summary:
            if is_xdist_worker:
                self.generate_summary_json()
                return
            kwargs = {}
            if 'json' in self.generate_summary:
                summary = self.generate_summary_json()
                print(f"A JSON report can be found at: {summary}")
            if result_hash_library.exists():  # link to it in the HTML
                kwargs["hash_library"] = result_hash_library.name
            if 'html' in self.generate_summary:
                summary = generate_summary_html(self._test_results, self.results_dir, **kwargs)
                print(f"A summary of test results can be found at: {summary}")
            if 'basic-html' in self.generate_summary:
                summary = generate_summary_basic_html(self._test_results, self.results_dir,
                                                      **kwargs)
                print(f"A summary of test results can be found at: {summary}")


class FigureCloser:
    """
    This is used in place of ImageComparison when the --mpl option is not used,
    to make sure that we still close figures returned by tests.
    """

    def __init__(self, config):
        self.config = config
        self.return_value = {}

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_call(self, item):
        wrap_figure_interceptor(self, item)
        yield
        if get_compare(item) is not None:
            test_name = generate_test_name(item)
            if test_name not in self.return_value:
                # Test function did not complete successfully
                return
            fig = self.return_value[test_name]
            close_mpl_figure(fig)
