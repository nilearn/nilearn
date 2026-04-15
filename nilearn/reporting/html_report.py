"""Generate HTML reports."""

from string import Template

from nilearn._assets import get_template
from nilearn._utils.html_document import HTMLDocument
from nilearn._version import __version__

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


def assemble_report(body: str, page_title: str) -> HTMLReport:
    """Put together head and body of report."""
    head_tpl = get_template("html/head.jinja")

    return HTMLReport(
        body=body,
        head_tpl=head_tpl,
        head_values={
            "head_css": True,
            "version": __version__,
            "page_title": page_title,
            "display_footer": "style='display: none'" if is_notebook() else "",
        },
    )


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
