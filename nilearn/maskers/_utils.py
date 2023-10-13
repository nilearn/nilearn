from nilearn import image


def compute_middle_image(img):
    """Compute middle image of timeseries (4D data)."""
    dim = image.load_img(img).shape
    if len(dim) == 4:
        img = image.index_img(img, dim[-1] // 2)
    return img, dim
