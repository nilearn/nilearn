"""Transformer for computing ROI signals of multiple 4D images."""

from nilearn._utils.docs import fill_doc
from nilearn.maskers._mixin import _MultiMixin
from nilearn.maskers.nifti_labels_masker import NiftiLabelsMasker


@fill_doc
class MultiNiftiLabelsMasker(_MultiMixin, NiftiLabelsMasker):
    """Class for extracting data from multiple Niimg-like objects \
       using labels of non-overlapping brain regions.

    MultiNiftiLabelsMasker is useful when data from non-overlapping volumes
    and from different subjects should be extracted (contrary to
    :class:`nilearn.maskers.NiftiLabelsMasker`).

    For more details on the definitions of labels in Nilearn,
    see the :ref:`region` section.

    Parameters
    ----------
    labels_img : Niimg-like object or None, default=None
        See :ref:`extracting_data`.
        Region definitions, as one image of labels.

    labels : :obj:`list` of :obj:`str`, optional
        Full labels corresponding to the labels image. This is used
        to improve reporting quality if provided.

        .. warning::
            The labels must be consistent with the label
            values provided through `labels_img`.

    %(masker_lut)s

    background_label : :obj:`int` or :obj:`float`, default=0
        Label used in labels_img to represent background.

        .. warning::

            This value must be consistent with label values and image provided.

    mask_img : Niimg-like object, optional
        See :ref:`extracting_data`.
        Mask to apply to regions before extracting signals.

    %(smoothing_fwhm)s

    %(standardize_false)s

    %(standardize_confounds)s

    high_variance_confounds : :obj:`bool`, default=False
        If True, high variance confounds are computed on provided image with
        :func:`nilearn.image.high_variance_confounds` and default parameters
        and regressed out.

    %(detrend)s

    %(low_pass)s

    %(high_pass)s

    %(t_r)s

    %(dtype)s

    resampling_target : {"data", "labels", None}, default="data"
        Defines which image gives the final shape/size:

        - ``"data"`` means the atlas is resampled
          to the shape of the data if needed.
        - ``"labels"`` means that the ``mask_img`` and images provided
          to ``fit()`` are resampled to the shape and affine of ``labels_img``.
        - ``"None"`` means no resampling:
          if shapes and affines do not match, a :obj:`ValueError` is raised.

    %(memory)s

    %(memory_level1)s

    %(verbose0)s

    %(strategy)s

    %(keep_masked_labels)s

    reports : :obj:`bool`, default=True
        If set to True, data is saved in order to produce a report.

    %(cmap)s
        default="CMRmap_r"
        Only relevant for the report figures.

    %(n_jobs)s

    %(clean_args)s

    Attributes
    ----------
    %(clean_args_)s

    labels_img_ : :obj:`nibabel.nifti1.Nifti1Image`
        The labels image.

    lut_ : :obj:`pandas.DataFrame`
        Look-up table derived from the ``labels`` or ``lut``
        or from the values of the label image.

    %(nifti_mask_img_)s

    memory_ : joblib memory cache

    See Also
    --------
    nilearn.maskers.NiftiMasker
    nilearn.maskers.NiftiLabelsMasker

    """

    def __init__(
        self,
        labels_img=None,
        labels=None,
        lut=None,
        background_label=0,
        mask_img=None,
        smoothing_fwhm=None,
        standardize=False,
        standardize_confounds=True,
        high_variance_confounds=False,
        detrend=False,
        low_pass=None,
        high_pass=None,
        t_r=None,
        dtype=None,
        resampling_target="data",
        memory=None,
        memory_level=1,
        verbose=0,
        strategy="mean",
        keep_masked_labels=False,
        reports=True,
        cmap="CMRmap_r",
        n_jobs=1,
        clean_args=None,
    ):
        self.n_jobs = n_jobs
        super().__init__(
            labels_img,
            labels=labels,
            lut=lut,
            background_label=background_label,
            mask_img=mask_img,
            smoothing_fwhm=smoothing_fwhm,
            standardize=standardize,
            standardize_confounds=standardize_confounds,
            high_variance_confounds=high_variance_confounds,
            low_pass=low_pass,
            high_pass=high_pass,
            detrend=detrend,
            t_r=t_r,
            dtype=dtype,
            resampling_target=resampling_target,
            memory=memory,
            memory_level=memory_level,
            verbose=verbose,
            strategy=strategy,
            reports=reports,
            cmap=cmap,
            clean_args=clean_args,
            keep_masked_labels=keep_masked_labels,
        )
