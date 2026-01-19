"""Base classes for all estimators and various utility functions."""

import itertools

from packaging.version import parse
from sklearn import __version__ as sklearn_version
from sklearn.base import BaseEstimator

from nilearn._version import __version__

SKLEARN_GTE_1_7 = parse(sklearn_version).release[1] >= 7
SKLEARN_LT_1_6 = parse(sklearn_version).release[1] < 6


class _NilearnHTMLDocumentationLinkMixin:
    """Mixin class allowing to help a link to the API documentation.

    See the sklearn.utils._repr_html.base.HTMLDocumentationLinkMixin
    for more details.
    """

    _doc_link_module = "nilearn"

    @property
    def _doc_link_template(self):
        version_url = "dev"
        nil_version = parse(__version__)
        if nil_version.dev is None:
            version_url = (
                f"{nil_version.major}.{nil_version.minor}.{nil_version.micro}"
            )

        return (
            f"https://nilearn.github.io/{version_url}/modules/generated/"
            "{estimator_module}.{estimator_name}.html"
        )

    def _doc_link_url_param_generator(self, *args):  # noqa : ARG002
        """Generate a link to the API documentation \
            for a given estimator.

        # TODO (sklearn >= 1.7) remove *args from signature
        """
        estimator_name = self.__class__.__name__
        tmp = list(
            itertools.takewhile(
                lambda part: not part.startswith("_"),
                self.__class__.__module__.split("."),
            )
        )
        estimator_module = ".".join([tmp[0], tmp[1]])

        return {
            "estimator_module": estimator_module,
            "estimator_name": estimator_name,
        }


class NilearnBaseEstimator(_NilearnHTMLDocumentationLinkMixin, BaseEstimator):
    """Base estimator for all Nilearn estimators."""

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(niimg_like=False)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(niimg_like=False)
        return tags

    def _more_tags(self):
        """Return estimator tags.

        TODO (sklearn >= 1.6.0) remove
        """
        return self.__sklearn_tags__()
