"""Generate HTML reports."""

import uuid
import warnings
from string import Template
from typing import Any

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

    data["has_plotting_engine"] = is_matplotlib_installed()
    if not is_matplotlib_installed():
        data["warning_messages"].append(MISSING_ENGINE_MSG)

    if estimator.reports is False:
        data["warning_messages"].append(
            "\nReport generation not enabled!\nNo visual outputs created."
        )

    if not estimator.__sklearn_is_fitted__():
        data["warning_messages"].append(UNFITTED_MSG)

    if estimator.__sklearn_is_fitted__() and not data["reports_at_fit_time"]:
        data["warning_messages"].append(
            "\nReport generation was disabled when fit was run. "
            "No reporting data is available.\n"
            "Make sure to set estimator.reports=True before fit."
        )

    if data["warning_messages"]:
        data["warning_messages"] = sorted(set(data["warning_messages"]))
        for msg in data["warning_messages"]:
            warnings.warn(
                msg,
                stacklevel=find_stack_level(),
                category=UserWarning,
            )

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

    summary_html: None | dict | str = None
    # only convert summary to html table if summary exists
    if "summary" in data and data["summary"] is not None:
        # convert region summary to html table
        # for Surface maskers create a table for each part
        if "Surface" in estimator.__class__.__name__:
            summary_html = {}
            for part in data["summary"]:
                summary_html[part] = pd.DataFrame.from_dict(
                    data["summary"][part]
                )
                summary_html[part] = dataframe_to_html(
                    summary_html[part],
                    precision=2,
                    header=True,
                    index=False,
                    sparsify=False,
                )
        # otherwise we just have one table
        elif "Nifti" in estimator.__class__.__name__:
            summary_html = dataframe_to_html(
                pd.DataFrame.from_dict(data["summary"]),
                precision=2,
                header=True,
                index=False,
                sparsify=False,
            )
    parameters = model_attributes_to_dataframe(estimator)
    with pd.option_context("display.max_colwidth", 100):
        parameters = dataframe_to_html(
            parameters,
            precision=2,
            header=True,
            sparsify=False,
        )

    if "n_elements" not in data:
        data["n_elements"] = 0

    if "coverage" not in data:
        data["coverage"] = ""
    if not isinstance(data["coverage"], str):
        data["coverage"] = f"{data['coverage']:0.1f}"

    if "overlay" in data:
        data["overlay"] = embed_img(data["overlay"])

    # TODO clean up docstring from RST formatting
    docstring = estimator.__doc__.split("Parameters\n")[0]

    env = return_jinja_env()

    body_tpl_path = f"html/maskers/{estimator._template_name}"
    body_tpl = env.get_template(body_tpl_path)

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
        is_notebook=is_notebook(),
        **data,
    )

    return assemble_report(body, f"{data['title']} report")


def is_notebook() -> bool:
    """Detect if we are running in a notebook.

    Adapted from https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
    except NameError:
        shell = False

    try:
        import marimo as mo

        is_marimo = mo.running_in_notebook()
    except ImportError:
        is_marimo = False

    if shell:
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)

    if is_marimo:
        return is_marimo

    return False
