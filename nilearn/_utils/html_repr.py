import itertools

from packaging.version import parse

from nilearn._version import __version__


class _NilearnHTMLDocumentationLinkMixin:
    """Mixin class allowing to generate a link to the API documentation.

    This mixin relies on three attributes:
    - `_doc_link_module`: it corresponds to the root module (e.g. `sklearn`).
      Using this
      mixin, the default value is `nilearn`.
    - `_doc_link_template`: it corresponds to the template used to generate the
      link to the API documentation. Using this mixin, the default value is
      `"https://scikit-learn.org/{version_url}/modules/generated/
      {estimator_module}.{estimator_name}.html"`.
    - `_doc_link_url_param_generator`: it corresponds to a function
      that generates the
      parameters to be used in the template when the estimator module
      and name are not
      sufficient.

    The method :meth:`_get_doc_link`
    generates the link to the API documentation for a given estimator.

    This mixin provides all the necessary states for
    :func:`sklearn.utils.estimator_html_repr` to generate a link to the API
    documentation for the estimator HTML diagram.
    """

    _doc_link_module = "nilearn"
    _doc_link_url_param_generator = None

    @property
    def _doc_link_template(self):
        nil_version = parse(__version__)
        if nil_version.dev is None:
            version_url = (
                f"{nil_version.major}.{nil_version.minor}.{nil_version.minor}"
            )
        else:
            version_url = "dev"
        return getattr(
            self,
            "__doc_link_template",
            (
                f"https://nilearn.github.io/{version_url}/modules/generated/"
                "{estimator_module}.{estimator_name}.html"
            ),
        )

    def _get_doc_link(self):
        """Generate a link to the API documentation for a given estimator.

        This method generates the link to the estimator's documentation page
        by using the template defined by the attribute `_doc_link_template`.

        Returns
        -------
        url : str
            The URL to the API documentation for this estimator.
            If the estimator does not belong to module
            `_doc_link_module`, the empty string (i.e. `""`) is returned.
        """
        if self.__class__.__module__.split(".")[0] != self._doc_link_module:
            return ""

        if self._doc_link_url_param_generator is None:
            estimator_name = self.__class__.__name__
            tmp = list(
                itertools.takewhile(
                    lambda part: not part.startswith("_"),
                    self.__class__.__module__.split("."),
                )
            )
            estimator_module = ".".join([tmp[0], tmp[1]])
            return self._doc_link_template.format(
                estimator_module=estimator_module,
                estimator_name=estimator_name,
            )

        return self._doc_link_template.format(
            **self._doc_link_url_param_generator()
        )
