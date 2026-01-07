import itertools

from packaging.version import parse
from sklearn import __version__ as sklearn_version

from nilearn._version import __version__

SKLEARN_GTE_1_7 = parse(sklearn_version).release[1] >= 7


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
