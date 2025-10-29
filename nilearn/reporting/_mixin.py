"""Module that provides mixin class to add reporting functionality to
estimators.
"""

import abc

from nilearn.reporting.html_report import generate_report


class ReportingMixin:
    """A mixin class to be used with classes that require reporting
    functionality.
    """

    def has_report_data(self):
        return hasattr(self, "_reporting_data")

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
        self._report_content["title"] = title
        return generate_report(self)

    def _reporting(self):
        # if report is disabled or the model is not yet fitted
        if not self.reports or not self.__sklearn_is_fitted__:
            self._report_content["summary"] = None
            return [None]
        return self._get_displays()

    @abc.abstractmethod
    def _get_displays(self):
        raise NotImplementedError()
