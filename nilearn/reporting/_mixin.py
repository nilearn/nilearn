"""Module that provides mixin class to add reporting functionality to
estimators.
"""

import abc

from nilearn.reporting.html_report import generate_report


class ReportingMixin:
    """A mixin class to be used with classes that require reporting
    functionality.
    """

    def _init_report(self):
        if not hasattr(self, "_report_content"):
            self._report_content = {
                "description": None,
                "summary": None,
                "warning_message": None,
            }
        if not (hasattr(self, "_reporting_data")):
            self._reporting_data = None

    def generate_report(self, title=None):
        """Generate an HTML report for the current object.

        Parameters
        ----------
        title : :obj:`str`, default=None
            title for the report. If None, title will be the class name.

        Returns
        -------
        report : `nilearn.reporting.html_report.HTMLReport`
            HTML report for the masker.
        """
        self._init_report()
        self._report_content["title"] = title
        return generate_report(self)

    @abc.abstractmethod
    def _reporting(self):
        raise NotImplementedError()
