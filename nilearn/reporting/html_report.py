"""Generate HTML reports."""

import uuid
import warnings
from string import Template

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

ESTIMATOR_TEMPLATES = {
    "NiftiLabelsMasker": "body_nifti_labels_masker.jinja",
    "MultiNiftiLabelsMasker": "body_nifti_labels_masker.jinja",
    "NiftiMapsMasker": "body_nifti_maps_masker.jinja",
    "MultiNiftiMapsMasker": "body_nifti_maps_masker.jinja",
    "NiftiSpheresMasker": "body_nifti_spheres_masker.jinja",
    "SurfaceMasker": "body_surface_masker.jinja",
    "MultiSurfaceMasker": "body_surface_masker.jinja",
    "SurfaceLabelsMasker": "body_surface_masker.jinja",
    "MultiSurfaceLabelsMasker": "body_surface_masker.jinja",
    "SurfaceMapsMasker": "body_surface_maps_masker.jinja",
    "MultiSurfaceMapsMasker": "body_surface_maps_masker.jinja",
}


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


def _update_template(
    docstring,
    content,
    overlay,
    parameters,
    data,
    summary_html=None,
    template_name=None,
    warning_messages=None,
) -> HTMLReport:
    """Populate a report with content.

    Parameters
    ----------
    docstring : str
        The introductory docstring for the reported object.

    content : img
        The content to display.

    overlay : img
        Overlaid content, to appear on hover.

    parameters : dict
        A dictionary of object parameters and their values.

    data : dict
        A dictionary holding the data to be added to the report.
        The keys must match exactly the ones used in the template.
        The default template accepts the following:
            - title (str) : Title of the report
            - description (str) : Description of the content.
            - warning_message (str) : An optional warning
              message to be displayed in red. This is used
              for example when no image was provided to the
              estimator when fitting.
        The NiftiLabelsMasker template accepts the additional
        fields:
            - summary (dict) : A summary description of the
              region labels and sizes. This will be displayed
              as an expandable table in the report.

    summary_html : dict if estimator is Surface masker str otherwise, optional
        Summary of the region labels and sizes converted to html table.

    template_name : str, optional
        The name of the template to use. If not provided, the
        default template `"body_masker.jinja"` will be
        used.

    Returns
    -------
    report : HTMLReport
        An instance of a populated HTML report.

    """
    if template_name is None:
        template_name = "body_masker.jinja"

    body_tpl_path = f"html/maskers/{template_name}"

    env = return_jinja_env()

    body_tpl = env.get_template(body_tpl_path)

    if "n_elements" not in data:
        data["n_elements"] = 0

    if "coverage" not in data:
        data["coverage"] = ""
    if not isinstance(data["coverage"], str):
        data["coverage"] = f"{data['coverage']:0.1f}"

    body = body_tpl.render(
        content=content,
        overlay=overlay,
        docstring=docstring,
        parameters=parameters,
        figure=(
            _insert_figure_partial(
                data["engine"],
                content,
                data["displayed_maps"],
                data["unique_id"],
            )
            if "engine" in data
            else None
        ),
        **data,
        carousel=False,
        warning_messages=warning_messages,
        summary_html=summary_html,
    )

    return assemble_report(body, f"{data['title']} report")


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


def _define_overlay(estimator):
    """Determine whether an overlay was provided and \
    update the report text as appropriate.
    """
    from nilearn.maskers import NiftiSpheresMasker

    displays = estimator._reporting()

    if len(displays) == 1:  # set overlay to None
        return None, displays[0]

    elif isinstance(estimator, NiftiSpheresMasker):
        return None, displays

    elif len(displays) == 2:
        return displays[0], displays[1]

    return None, displays


def generate_report(estimator) -> list[None] | HTMLReport:
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
    if not is_matplotlib_installed():
        with warnings.catch_warnings():
            mpl_unavail_msg = (
                "Matplotlib is not imported! No reports will be generated."
            )
            warnings.filterwarnings("always", message=mpl_unavail_msg)
            warnings.warn(
                category=ImportWarning,
                message=mpl_unavail_msg,
                stacklevel=find_stack_level(),
            )
            return [None]

    data = {}
    if hasattr(estimator, "_report_content"):
        data = estimator._report_content

    # Generate a unique ID for this report
    data["unique_id"] = str(uuid.uuid4()).replace("-", "")

    if data.get("title") is None:
        data["title"] = estimator.__class__.__name__

    warning_messages = []

    if estimator.reports is False:
        warning_messages.append(
            "\nReport generation not enabled!\nNo visual outputs created."
        )

    if not estimator.__sklearn_is_fitted__():
        warning_messages.append(
            "\nThis report was not generated.\n"
            "Make sure to run `fit` before inspecting reports."
        )

    if warning_messages:
        for msg in warning_messages:
            warnings.warn(
                msg,
                stacklevel=find_stack_level(),
            )

        data["title"] = "Empty Report"

        return _update_template(
            docstring="Empty Report",
            content=embed_img(None),
            overlay=None,
            parameters={},
            data=data,
            warning_messages=warning_messages,
        )

    return _create_report(estimator, data)


def _insert_figure_partial(engine, content, displayed_maps, unique_id=None):
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


def _create_report(estimator, data) -> HTMLReport:
    template_name = ESTIMATOR_TEMPLATES.get(estimator.__class__.__name__, None)

    # note that some surface images are passed via data
    # for surface maps masker
    overlay, image = _define_overlay(estimator)
    embeded_images = (
        [embed_img(i) for i in image]
        if isinstance(image, list)
        else embed_img(image)
    )

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
    docstring = estimator.__doc__
    snippet = docstring.partition("Parameters\n    ----------\n")[0]

    return _update_template(
        docstring=snippet,
        content=embeded_images,
        overlay=embed_img(overlay),
        parameters=parameters,
        data=data,
        template_name=template_name,
        summary_html=summary_html,
    )


def is_notebook() -> bool:
    """Detect if we are running in a notebook.

    From https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False  # Probably standard Python interpreter
