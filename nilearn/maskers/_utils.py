import warnings

from nilearn import image
from nilearn._utils.logger import find_stack_level


def sanitize_cleaning_parameters(masker):
    """Make sure that clarning parameters are passed via clean_args.

    TODO simplify after nilearn 0.13.2
    """
    if hasattr(masker, "clean_kwargs"):
        if masker.clean_kwargs:
            warnings.warn(
                f"You passed some kwargs to {type(masker)}"
                "This behavior is deprecated "
                "and will be removed in version 0.13.2.",
                DeprecationWarning,
                stacklevel=find_stack_level(),
            )
            if masker.clean_args:
                raise ValueError(
                    "Passing arguments via 'kwargs' "
                    "is mutually exclusive with using 'clean_args'"
                )
        masker.clean_kwargs_ = {
            k[7:]: v
            for k, v in masker.clean_kwargs.items()
            if k.startswith("clean__")
        }
    if masker.clean_args is None:
        masker.clean_args_ = {}
    else:
        masker.clean_args_ = masker.clean_args

    return masker


def _check_dims(imgs):
    # check dims of one image if given a list
    if isinstance(imgs, list):
        im = imgs[0]
        dim = image.load_img(im).shape
        # in case of 4D (timeseries) + 1D (subjects) return first subject
        return (im, (*dim, 1)) if len(dim) == 4 else (imgs, (*dim, 1))
    else:
        dim = image.load_img(imgs).shape
        return imgs, dim


def compute_middle_image(img):
    """Compute middle image of timeseries (4D data)."""
    img, dim = _check_dims(img)
    if len(dim) in {4, 5}:
        img = image.index_img(img, dim[-1] // 2)
    return img, len(dim)
