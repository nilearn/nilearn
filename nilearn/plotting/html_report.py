import os
import tempita
from . import js_plotting_utils as plot_utils


class ReportMixin():
    """
    Generate a report for Nilearn objects.

    Report is useful to visualize steps in a processing pipeline.
    Example use case: visualize the overlap of a mask and reference image
    in NiftiMasker.

    Returns
    -------
    report : HTML report with embedded content
    """

    def _update_template(name, content, description=None):
        """
        Populate a report with content.

        Parameters
        ----------
        name: str
            The name for the report
        content: img
            The content to display
        description: str
            An optional description of the content

        Returns
        -------
        html : populated HTML report
        """
        template_name = 'report_template.html'
        template_path = os.path.join(
            os.path.dirname(__file__), 'data', 'html', template_name)
        tpl = tempita.HTMLTemplate.from_filename(template_path,
                                                 encoding='utf-8')

        html = tpl.substitute(name=name, content=content,
                              description=description)
        return plot_utils.HTMLDocument(html)
