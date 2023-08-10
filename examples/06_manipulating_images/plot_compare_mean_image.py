"""
Comparing the means of 2 images
===============================

The goal of this example is to illustrate the use of the function
:func:`nilearn.image.math_img` with a list of images as input.
We compare the means of 2 resting state 4D images. The mean of the images
could have been computed with nilearn :func:`nilearn.image.mean_img` function.
"""


###############################################################################
# Fetching 2 subject movie watching brain development fmri datasets.
from nilearn import datasets

dataset = datasets.fetch_development_fmri(n_subjects=2)


###############################################################################
# Print basic information on the adhd subjects resting state datasets.
print(f"Subject 1 resting state dataset at: {dataset.func[0]}")
print(f"Subject 2 resting state dataset at: {dataset.func[1]}")


###############################################################################
# Comparing the means of the 2 movie watching datasets.
from nilearn import image, plotting

result_img = image.math_img(
    "np.mean(img1, axis=-1) - np.mean(img2, axis=-1)",
    img1=dataset.func[0],
    img2=dataset.func[1],
)

plotting.plot_stat_map(
    result_img, title="Comparing means of 2 resting state 4D images."
)
plotting.show()
