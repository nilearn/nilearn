"""Mixin classes for maskers."""

from nilearn._utils.docs import fill_doc
from nilearn._utils.tags import SKLEARN_LT_1_6


class _MultiMixin:
    """Mixin class to add common MultiMasker functionalities."""

    def __sklearn_tags__(self):
        """Return estimator tags.

        See the sklearn documentation for more details on tags
        https://scikit-learn.org/1.6/developers/develop.html#estimator-tags
        """
        # TODO (sklearn  >= 1.6.0) remove if block
        if SKLEARN_LT_1_6:
            from nilearn._utils.tags import tags

            return tags(masker=True, multi_masker=True)

        from nilearn._utils.tags import InputTags

        tags = super().__sklearn_tags__()
        tags.input_tags = InputTags(masker=True, multi_masker=True)
        return tags

    @fill_doc
    def fit_transform(self, imgs, y=None, confounds=None, sample_mask=None):
        """
        Fit to data, then transform it.

        Parameters
        ----------
        imgs : Image object, or a :obj:`list` of Image objects
            See :ref:`extracting_data`.
            Data to be preprocessed

        y : None
            This parameter is unused.
            It is solely included for scikit-learn compatibility.

        %(confounds_multi)s

        %(sample_mask_multi)s

            .. versionadded:: 0.8.0

        Returns
        -------
        %(signals_transform_multi_nifti)s
        """
        return self.fit(imgs, y=y).transform(
            imgs, confounds=confounds, sample_mask=sample_mask
        )
