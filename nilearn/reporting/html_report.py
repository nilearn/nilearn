"""Generate HTML reports."""

import abc
import uuid
import warnings
from copy import deepcopy
from string import Template
from typing import Any, ClassVar

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.html_document import (
    HEIGHT_DEFAULT,
    WIDTH_DEFAULT,
    HTMLDocument,
)
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

    def _dataframe_to_html(
        self, df_cvrt, precision=2, header=True, index=False, sparsify=False
    ):
        """Create html content from the specified dataframe content."""
        return dataframe_to_html(
            df_cvrt,
            precision=precision,
            header=header,
            index=index,
            sparsify=sparsify,
        )

    def _dict_to_html(
        self, dict_cvrt, precision=2, header=True, index=False, sparsify=False
    ):
        """Create html content from the specified dictionary content. The
        dictionary is expected to be key value pairs without depth.
        """
        df_cvrt = pd.DataFrame.from_dict(dict_cvrt)
        return self._dataframe_to_html(
            df_cvrt,
            precision=precision,
            header=header,
            index=index,
            sparsify=sparsify,
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
        return is_notebook()

    def _run_report_checks(self):
        if self.reports is False:
            self._append_warning(
                "\nReport generation not enabled!\nNo visual outputs created."
            )

        if not self.__sklearn_is_fitted__():
            self._append_warning(UNFITTED_MSG)

        report = self._report_content
        if self.__sklearn_is_fitted__() and not report["reports_at_fit_time"]:
            self._append_warning(
                "\nReport generation was disabled when fit was run. "
                "No reporting data is available.\n"
                "Make sure to set self.reports=True before fit."
            )

        if not is_matplotlib_installed():
            self._append_warning(MISSING_ENGINE_MSG)

    def _display_report_warnings(self):
        report_warnings = self._get_warnings()
        if report_warnings:
            for msg in report_warnings:
                warnings.warn(
                    msg,
                    stacklevel=find_stack_level(),
                    category=UserWarning,
                )

    def _set_report_basics(self, title):
        report = self._report_content

        # Generate a unique ID for report
        report["unique_id"] = str(uuid.uuid4()).replace("-", "")

        # Set title for report
        report["title"] = title if title else self.__class__.__name__

        # TODO clean up docstring from RST formatting
        report["docstring"] = self.__doc__.split("Parameters\n")[0]

        report["has_plotting_engine"] = is_matplotlib_installed()

    @abc.abstractmethod
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
        raise NotImplementedError()


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


def is_notebook():
    """Detect if we are running in a notebook.

    From https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
    """
    try:
        shell = get_ipython().__class__.__name__  # type: ignore[name-defined]
        return shell == "ZMQInteractiveShell"
    except NameError:
        return False  # Probably standard Python interpreter


@fill_doc
def make_glm_report(
    model,
    contrasts=None,
    first_level_contrast=None,
    title=None,
    bg_img="MNI152TEMPLATE",
    threshold=3.09,
    alpha=0.001,
    cluster_threshold=0,
    height_control="fpr",
    two_sided=False,
    min_distance=8.0,
    plot_type="slice",
    cut_coords=None,
    display_mode=None,
    report_dims=(WIDTH_DEFAULT, HEIGHT_DEFAULT),
) -> HTMLReport:
    """Return HTMLReport object \
    for a report which shows all important aspects of a fitted GLM.

    The object can be opened in a browser, displayed in a notebook,
    or saved to disk as a standalone HTML file.

    Examples
    --------
    report = make_glm_report(model, contrasts)
    report.open_in_browser()
    report.save_as_html(destination_path)

    Parameters
    ----------
    model : FirstLevelModel or SecondLevelModel object
        A fitted first or second level model object.
        Must have the computed design matrix(ces).

    contrasts : :obj:`dict` with :obj:`str` - ndarray key-value pairs \
        or :obj:`str` \
        or :obj:`list` of :obj:`str` \
        or ndarray or \
        :obj:`list` of ndarray, Default=None

        Contrasts information for a first or second level model.

        Example:

            Dict of :term:`contrast` names and coefficients,
            or list of :term:`contrast` names
            or list of :term:`contrast` coefficients
            or :term:`contrast` name
            or :term:`contrast` coefficient

            Each :term:`contrast` name must be a string.
            Each :term:`contrast` coefficient must be a list
            or numpy array of ints.

        Contrasts are passed to ``contrast_def`` for FirstLevelModel
        (:func:`nilearn.glm.first_level.FirstLevelModel.compute_contrast`)
        & second_level_contrast for SecondLevelModel
        (:func:`nilearn.glm.second_level.SecondLevelModel.compute_contrast`)

    %(first_level_contrast)s

        .. nilearn_versionadded:: 0.12.0

    title : :obj:`str`, default=None
        If string, represents the web page's title and primary heading,
        model type is sub-heading.
        If None, page titles and headings are autogenerated
        using :term:`contrast` names.

    bg_img : Niimg-like object, default='MNI152TEMPLATE'
        See :ref:`extracting_data`.
        The background image for mask and stat maps to be plotted on upon.
        To turn off background image, just pass "bg_img=None".

    threshold : :obj:`float`, default=3.09
        Cluster forming threshold in same scale as `stat_img` (either a
        t-scale or z-scale value). Used only if height_control is None.

        .. note::

            - When ``two_sided`` is True:

              ``'threshold'`` cannot be negative.

              The given value should be within the range of minimum and
              maximum intensity of the input image.
              All intensities in the interval ``[-threshold, threshold]``
              will be set to zero.

            - When ``two_sided`` is False:

              - If the threshold is negative:

                It should be greater than the minimum intensity
                of the input data.
                All intensities greater than or equal
                to the specified threshold will be set to zero.
                All other intensities keep their original values.

              - If the threshold is positive:

                It should be less than the maximum intensity
                of the input data.
                All intensities less than or equal
                to the specified threshold will be set to zero.
                All other intensities keep their original values.

    alpha : :obj:`float`, default=0.001
        Number controlling the thresholding (either a p-value or q-value).
        Its actual meaning depends on the height_control parameter.
        This function translates alpha to a z-scale threshold.

    %(cluster_threshold)s

    height_control :  :obj:`str`, default='fpr'
        false positive control meaning of cluster forming
        threshold: 'fpr' or 'fdr' or 'bonferroni' or None.

    two_sided : :obj:`bool`, default=False
        Whether to employ two-sided thresholding or to evaluate positive
        values only.

    min_distance : :obj:`float`, default=8.0
        For display purposes only.
        Minimum distance between subpeaks in mm.

    plot_type : :obj:`str`, {'slice', 'glass'}, default='slice'
        Specifies the type of plot to be drawn for the statistical maps.

    %(cut_coords)s

    display_mode :  :obj:`str`, default=None
        Default is 'z' if plot_type is 'slice'; '
        ortho' if plot_type is 'glass'.

        Choose the direction of the cuts:
        'x' - sagittal, 'y' - coronal, 'z' - axial,
        'l' - sagittal left hemisphere only,
        'r' - sagittal right hemisphere only,
        'ortho' - three cuts are performed in orthogonal directions.

        Possible values are:
        'ortho', 'x', 'y', 'z', 'xz', 'yx', 'yz',
        'l', 'r', 'lr', 'lzr', 'lyr', 'lzry', 'lyrz'.

    report_dims : Sequence[:obj:`int`, :obj:`int`], default=(1600, 800)
        Specifies width, height (in pixels) of report window within a
        notebook.
        Only applicable when inserting the report into a Jupyter notebook.
        Can be set after report creation using report.width, report.height.

    Returns
    -------
    report_text : HTMLReport Object
        Contains the HTML code for the :term:`GLM` Report.

    """
    return model._make_glm_report(
        contrasts=contrasts,
        first_level_contrast=first_level_contrast,
        title=title,
        bg_img=bg_img,
        threshold=threshold,
        alpha=alpha,
        cluster_threshold=cluster_threshold,
        height_control=height_control,
        two_sided=two_sided,
        min_distance=min_distance,
        plot_type=plot_type,
        cut_coords=cut_coords,
        display_mode=display_mode,
        report_dims=report_dims,
    )

