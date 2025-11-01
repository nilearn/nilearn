"""Replacement for sklearn mixins."""

import datetime
import uuid

import numpy as np
from sklearn.preprocessing import LabelBinarizer

from nilearn._utils.docs import fill_doc
from nilearn._utils.helpers import is_matplotlib_installed
from nilearn._utils.param_validation import (
    check_params,
)
from nilearn._utils.tags import SKLEARN_LT_1_6
from nilearn._version import __version__
from nilearn.maskers import SurfaceMasker
from nilearn.reporting.glm_reporter import _mask_to_plot
from nilearn.reporting.html_report import (
    HTMLReport,
    is_notebook,
    return_jinja_env,
)
from nilearn.reporting.utils import CSS_PATH
from nilearn.surface import SurfaceImage

MNI152TEMPLATE = None
if is_matplotlib_installed():
    pass


class _ReportMixin:
    def _is_volume_glm(self):
        """Return if model is run on volume data or not."""
        return not (
            (
                hasattr(self, "mask")
                and isinstance(self.mask, (SurfaceMasker, SurfaceImage))
            )
            or (
                self.__sklearn_is_fitted__()
                and hasattr(self, "masker_")
                and isinstance(self.masker_, SurfaceMasker)
            )
        )

    def generate_report(
        self, title: str | None = None, bg_img=None, cut_coords=None
    ) -> HTMLReport:
        if title is None:
            title = self.__class__.__name__

        mask_plot = _mask_to_plot(self, bg_img=bg_img, cut_coords=cut_coords)

        env = return_jinja_env()

        body_tpl = env.get_template("html/decoding/body_decoders.jinja")

        docstring = "" if self.__doc__ is None else self.__doc__

        body = body_tpl.render(
            docstring=docstring.partition("Parameters\n    ----------\n")[0],
            date=datetime.datetime.now().replace(microsecond=0).isoformat(),
            title=title,
            unique_id=str(uuid.uuid4()).replace("-", ""),
            mask_plot=mask_plot,
        )

        head_tpl = env.get_template("html/head.jinja")

        head_css_file_path = CSS_PATH / "head.css"
        with head_css_file_path.open(encoding="utf-8") as head_css_file:
            head_css = head_css_file.read()

        report = HTMLReport(
            body=body,
            head_tpl=head_tpl,
            head_values={
                "head_css": head_css,
                "version": __version__,
                "page_title": title,
                "display_footer": "style='display: none'"
                if is_notebook()
                else "",
            },
        )

        return report


class _ClassifierMixin:
    _estimator_type = "classifier"  # TODO (sklearn >= 1.6) remove

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn >= 1.6) get rid of if block
        # when bumping sklearn_version > 1.5
        # see https://github.com/scikit-learn/scikit-learn/pull/29677
        tags = super().__sklearn_tags__()
        if SKLEARN_LT_1_6:
            return tags

        from sklearn.utils import ClassifierTags

        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()

        return tags

    def _n_problems(self):
        if self.n_classes_ > 2:
            return self.n_classes_
        else:
            return 1

    def _binarize_y(self, y):
        """Encode target classes as -1 and 1.

        Helper function invoked just before fitting a classifier.
        """
        y = np.array(y)

        self._enc = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self._enc.fit_transform(y)
        self.classes_ = self._enc.classes_
        self.n_classes_ = len(self.classes_)
        return y

    def _get_classes(self):
        return self.classes_


class _RegressorMixin:
    _estimator_type = "regressor"  # TODO (sklearn >= 1.6) remove

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn >= 1.6) get rid of if block
        # when bumping sklearn_version > 1.5
        # see https://github.com/scikit-learn/scikit-learn/pull/29677
        tags = super().__sklearn_tags__()
        if SKLEARN_LT_1_6:
            tags["multioutput"] = True
            return tags
        from sklearn.utils import RegressorTags

        tags.estimator_type = "regressor"
        tags.regressor_tags = RegressorTags()

        return tags

    @fill_doc
    def fit(self, X, y, groups=None):
        """Fit the decoder (learner).

        Parameters
        ----------
        X : list of Niimg-like or :obj:`~nilearn.surface.SurfaceImage` objects
            See :ref:`extracting_data`.
            Data on which model is to be fitted.
            If this is a list,
            the affine is considered the same for all.

        y : numpy.ndarray of shape=(n_samples) or list of length n_samples
            The dependent variable (age, sex, IQ, yes/no, etc.).
            Target variable to predict.
            Must have exactly as many elements as the input images.

        %(groups)s

        """
        check_params(self.__dict__)
        self._classes_ = ["beta"]
        return super().fit(X, y, groups=groups)

    def _n_problems(self):
        return 1

    def _binarize_y(self, y):
        return y[:, np.newaxis]

    def _get_classes(self):
        return self._classes_
