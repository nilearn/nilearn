"""Nilearn tags for estimators."""

from dataclasses import dataclass

from packaging.version import parse
from sklearn import __version__ as sklearn_version

ver = parse(sklearn_version)
if ver.release[1] >= 6:
    from sklearn.utils import InputTags as SkInputTags

    @dataclass
    class InputTags(SkInputTags):
        """Tags for the input data.

        Nilearn version of sklearn.utils.InputTags
        https://scikit-learn.org/1.6/modules/generated/sklearn.utils.InputTags.html#sklearn.utils.InputTags
        """

        # same as base input tags of
        # sklearn.utils.InputTags
        one_d_array: bool = False
        two_d_array: bool = True
        three_d_array: bool = False
        sparse: bool = False
        categorical: bool = False
        string: bool = False
        dict: bool = False
        positive_only: bool = False
        allow_nan: bool = False
        pairwise: bool = False

        # nilearn specific things
        # estimator accepts for str, Path to .nii[.gz] file
        # or NiftiImage object
        niimg_like: bool = True
        # estimator accepts SurfaceImage object
        surf_img: bool = False
else:

    def tags(niimg_like=True, surf_img=False, **kwargs):
        """Add nilearn tags to estimator.

        See also: InputTags

        TODO remove when dropping sklearn 1.5
        """
        X_types = []
        if "X_types" in kwargs:
            X_types = kwargs["X_types"]
        X_types.append("2darray")
        if niimg_like:
            X_types.append("niimg_like")
        if surf_img:
            X_types.append("surf_img")
        X_types = list(set(X_types))
        return dict(X_types=X_types, **kwargs)
