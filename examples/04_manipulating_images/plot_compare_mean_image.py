"""
Comparing the mean of 2 images
==============================

Here we compare the means of 2 images.
"""

from nilearn import datasets, plotting, image

dataset = datasets.fetch_adhd(n_subjects=2)

# Print basic information on the adhd subjects resting state datasets.
print('Subject 1 resting state dataset at: %s' % dataset.func[0])
print('Subject 2 resting state dataset at: %s' % dataset.func[1])

# Comparing the means of the 2 resting state datasets.
formula = "np.mean(img1, axis=-1) - np.mean(img2, axis=-1)"

result_img = image.math_img(formula,
                            img1=dataset.func[0],
                            img2=dataset.func[1])

plotting.plot_epi(result_img, title="Comparing means of 2 resting 4D images.")
plotting.show()
