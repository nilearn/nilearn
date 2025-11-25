"""Generate HTML reports."""

import uuid
import warnings
from copy import deepcopy
from string import Template
from typing import Any, ClassVar

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.html_document import HTMLDocument
from nilearn._utils.logger import find_stack_level
from nilearn._version import __version__
from nilearn.reporting._utils import (
    dataframe_to_html,
    model_attributes_to_dataframe,
)
from nilearn.reporting.utils import (
    CSS_PATH,
    TEMPLATE_ROOT_PATH,
    figure_to_svg_base64,
)

UNFITTED_MSG = (
    "\nThis estimator has not been fit yet.\n"
    "Make sure to run `fit` before inspecting reports."
)

MISSING_ENGINE_MSG = (
    "\nNo plotting back-end detected.\nReport will be missing figures."
)


def _update_defaults(base_dict: dict, update_dict: dict):
    """Return a new dictionary with the values of dictionary base_dict updated
    recursively with the values of update_dict.
    """
    new_dict = deepcopy(base_dict)
    for k, v in update_dict.items():
        if (
            k in new_dict
            and isinstance(new_dict[k], dict)
            and isinstance(v, dict)
        ):
            v = _update_defaults(new_dict[k], v)
        new_dict[k] = v
    return new_dict


class ReportMixin:
    """Mixin class to be inherited by the estimators with reporting
    functionality.

    Provides interfaces and implementations for common methods.

    Each inheriting estimator has below fields in its reporting content:

    title: Title of the report.
           Can be set by the user at report generation; otherwise set to class
           name.
    description: Description of the report for that specific estimator.
    summary: Summary of the report.
           Created depending on report data when generating report.
    warning_messages: Warnings while generating the report.
           If there are warnings an empty report displaying the warnings is
           generated.

    Inheriting estimators can define additional fields or update existing
    fields defining _REPORT_DEFAULTS class variable.


    Ex.

    class Reportable1(ReportMixin):
        _REPORT_DEFAULTS = {
            "description": (
                "This report shows the input Nifti image overlaid "
                "with the outlines of the mask (in green). We "
                "recommend to inspect the report for the overlap "
                "between the mask and its input image. "
            ),
            "n_elements": 0,
            "coverage": 0,
        }

    A non-empty report has the following section:

    - head (title, description, estimator params, etc)
    - plots
    - summary

    If empty:

    - head (title, description, estimator params, etc)
    - warnings

    """

    _REPORT_DEFAULTS: ClassVar[dict[str, Any]] = {
        "title": None,
        "description": "",
        "summary": {},
        "warning_messages": [],
        "engine": "matplotlib",
        "has_plotting_engine": True,
    }

    def __init_subclass__(cls):
        super().__init_subclass__()
        # sets implementing class _REPORT_DEFAULTS
        # updating the base class value with implementing class value
        cls._REPORT_DEFAULTS = _update_defaults(
            ReportMixin._REPORT_DEFAULTS, cls._REPORT_DEFAULTS
        )

    def _reset_report(self):
        self._report_content = deepcopy(self._REPORT_DEFAULTS)
        if self._has_report_data():
            del self._reporting_data

    def _has_report_data(self):
        """
        Check if the model is fitted and _reporting_data is populated.

        Returns
        -------
        bool
            True if reporting is enabled, the model is fitted and
        _reporting_data is populated; False otherwise.
        """
        return hasattr(self, "_reporting_data")

    def _append_warning(self, warning):
        """Append the specified warning to the warning list of the report.

        Parameters
        ----------
        warning: str
            warning to be added to the list of warnings.
        """
        self._report_content["warning_messages"].append(warning)

    def _get_warnings(self):
        """Return the sorted list of report warnings.

        Returns
        -------
        list of str
            the list of warnings, empty list if there are no warnings
        """
        return sorted(set(self._report_content["warning_messages"]))

    def _dict_to_html(self, dict_cvrt):
        """Creates html content from the specified dictionary content. The
        dictionary is expected to be key value pairs without depth.
        """
        df_cvrt = pd.DataFrame.from_dict(dict_cvrt)
        return dataframe_to_html(
                df_cvrt,
                precision=2,
                header=True,
                index=False,
                sparsify=False,
            )

    def _get_body_template(self, estimator_type):
        env = return_jinja_env()

        body_tpl_path = f"html/{estimator_type}/{self._template_name}"
        return env.get_template(body_tpl_path)

    def _get_partial_template(self, estimator_type, tpl_name, is_common=False):
        env = return_jinja_env()
        loc = f"/{estimator_type}" if not is_common else ""
        return env.get_template(f"html{loc}/partials/{tpl_name}.jinja")

    def _model_params_to_html(self):
        parameters = model_attributes_to_dataframe(self)
        with pd.option_context("display.max_colwidth", 100):
            parameters = dataframe_to_html(
                parameters,
                precision=2,
                header=True,
                sparsify=False,
            )
        return parameters

    def _embed_img(self, img):
        return embed_img(img)

    def _assemble_report(self, body, title):
        return assemble_report(body, title)

    def _is_notebook(self):
        """Detect if we are running in a notebook.

        From https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
        """
        try:
            shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
            return shell == "ZMQInteractiveShell"
        except NameError:
            return False  # Probably standard Python interpreter

    def generate_report(self, title: str | None = None):
        """Generate an HTML report for the current object.

        Parameters
        ----------
        title : :obj:`str` or None, default=None
            title for the report. If None, title will be the class name.

        Returns
        -------
        report : `nilearn.reporting.html_report.HTMLReport`
            HTML report for the masker.
        """
        _run_report_checks(self)

        report = self._report_content

        # Generate a unique ID for report
        report["unique_id"] = str(uuid.uuid4()).replace("-", "")

        # Set title for report
        report["title"] = title

        # TODO clean up docstring from RST formatting
        report["docstring"] = self.__doc__.split("Parameters\n")[0]

        return self._create_report()


class HTMLReport(HTMLDocument):
    """A report written as HTML.

    Methods such as ``save_as_html``, or ``open_in_browser``
    are inherited from class ``nilearn.plotting.html_document.HTMLDocument``.

    Parameters
    ----------
    head_tpl : str.Template or Jinja Template
        This is meant for display as a full page, like writing on disk.
        This is the Template object used to generate the HTML head
        section of the report. The template should be filled with:

            - title: The title of the HTML page.
            - body: The full body of the HTML page. Provided through
                the ``body`` input.

    body : :obj:`str`
        This parameter is used for embedding in the provided
        ``head_tpl`` template. It contains the full body of the
        HTML page.

    head_values : :obj:`dict`, default=None
        Additional substitutions in ``head_tpl``.
        if ``None`` is passed, defaults to ``{}``

        .. note::
            This can be used to provide additional values
            with custom templates.

    """

    def __init__(self, head_tpl, body, head_values=None):
        """Construct the ``HTMLReport`` class."""
        if head_values is None:
            head_values = {}

        if isinstance(head_tpl, Template):
            html = head_tpl.safe_substitute(body=body, **head_values)
            self.head_tpl = head_tpl.safe_substitute(**head_values)
        else:
            # in this case we are working with jinja template
            html = head_tpl.render(body=body, **head_values)
            self.head_tpl = head_tpl.render(**head_values)

        self.body = body

        super().__init__(html)

    def _repr_html_(self):
        """Return body of the report.

        Method used by the Jupyter notebook.
        Users normally won't call this method explicitly.
        """
        return self.body

    def __str__(self):
        return self.body


def return_jinja_env() -> Environment:
    """Set up the jinja Environment."""
    return Environment(
        loader=FileSystemLoader(TEMPLATE_ROOT_PATH),
        autoescape=select_autoescape(),
        lstrip_blocks=True,
        trim_blocks=True,
    )


def embed_img(display):
    """Embed an image or just return its instance if already embedded.

    Parameters
    ----------
    display : obj
        A Nilearn plotting object to display.

    Returns
    -------
    embed : str
        Binary image string.

    """
    if display is None:  # no image to display
        return None
    # If already embedded, simply return as is
    if isinstance(display, str):
        return display
    return figure_to_svg_base64(display.frame_axes.figure)


def assemble_report(body: str, title: str) -> HTMLReport:
    """Put together head and body of report."""
    env = return_jinja_env()

    head_tpl = env.get_template("html/head.jinja")

    head_css_file_path = CSS_PATH / "head.css"
    with head_css_file_path.open(encoding="utf-8") as head_css_file:
        head_css = head_css_file.read()

    return HTMLReport(
        body=body,
        head_tpl=head_tpl,
        head_values={
            "head_css": head_css,
            "version": __version__,
            "page_title": title,
            "display_footer": "style='display: none'" if is_notebook() else "",
        },
    )


def _run_report_checks(estimator):

    if estimator.reports is False:
        estimator._append_warning(
            "\nReport generation not enabled!\nNo visual outputs created."
        )

    if not estimator.__sklearn_is_fitted__():
        estimator._append_warning(UNFITTED_MSG)

    report = estimator._report_content
    if estimator.__sklearn_is_fitted__() and not report["reports_at_fit_time"]:
        estimator._append_warning(
            "\nReport generation was disabled when fit was run. "
            "No reporting data is available.\n"
            "Make sure to set estimator.reports=True before fit."
        )

    report["has_plotting_engine"] = is_matplotlib_installed()

    if not is_matplotlib_installed():
        estimator._append_warning(MISSING_ENGINE_MSG)

    report_warnings = estimator._get_warnings()
    if report_warnings:
        for msg in report_warnings:
            warnings.warn(
                msg,
                stacklevel=find_stack_level(),
                category=UserWarning,
            )


def generate_report(estimator) -> HTMLReport:
    """Generate a report for Nilearn objects.

    Reports are useful to visualize steps in a processing pipeline.
    Example use case: visualize the overlap of a mask and reference image
    in NiftiMasker.

    Parameters
    ----------
    estimator : Object instance of BaseEstimator.
        Object for which the report should be generated.

    Returns
    -------
    report : HTMLReport

    """
    data = {}
    if hasattr(estimator, "_report_content"):
        data = estimator._report_content

    # Generate a unique ID for this report
    data["unique_id"] = str(uuid.uuid4()).replace("-", "")

    if data["title"] is None:
        data["title"] = estimator.__class__.__name__

    _run_report_checks(estimator)

    return _create_report(estimator, data)


def _insert_figure_partial(
    engine, content, displayed_maps, unique_id: str
) -> str:
    env = return_jinja_env()

    tpl = env.get_template("html/maskers/partials/figure.jinja")

    if not isinstance(content, list):
        content = [content]
    return tpl.render(
        engine=engine,
        content=content,
        displayed_maps=displayed_maps,
        unique_id=unique_id,
    )


def _create_report(
    estimator,
    data: dict[str, Any],
) -> HTMLReport:
    embeded_images = None
    image = estimator._reporting()
    if image is None:
        embeded_images = None
    elif not isinstance(image, list):
        embeded_images = embed_img(image)
    elif all(x is None for x in image):
        embeded_images = None
    else:
        embeded_images = [embed_img(i) for i in image]

    summary_html = estimator._get_summary_html()

    parameters = model_attributes_to_dataframe(estimator)
    with pd.option_context("display.max_colwidth", 100):
        parameters = dataframe_to_html(
            parameters,
            precision=2,
            header=True,
            sparsify=False,
        )

    if not isinstance(data["coverage"], str):
        data["coverage"] = f"{data['coverage']:0.1f}"

    if "overlay" in data:
        data["overlay"] = embed_img(data["overlay"])

    # TODO clean up docstring from RST formatting
    docstring = estimator.__doc__.split("Parameters\n")[0]

    body_tpl = estimator._get_body_template("maskers")

    body = body_tpl.render(
        content=embeded_images,
        docstring=docstring,
        parameters=parameters,
        figure=(
            _insert_figure_partial(
                data["engine"],
                embeded_images,
                data["displayed_maps"],
                data["unique_id"],
            )
            if "engine" in data
            else None
        ),
        summary_html=summary_html,
        **data,
    )

    return assemble_report(body, f"{data['title']} report")
