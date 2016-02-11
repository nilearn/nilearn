"""
Comparing the means of 2 images
===============================

Here we compare the means of 2 resting state 4D images.
"""

from nilearn import datasets, plotting, image

################################################################################
# Fetching 2 subject resting state functionnal MRI from datasets.
dataset = datasets.fetch_adhd(n_subjects=2)

################################################################################
# Print basic information on the adhd subjects resting state datasets.
print('Subject 1 resting state dataset at: %s' % dataset.func[0])
print('Subject 2 resting state dataset at: %s' % dataset.func[1])

################################################################################
# Comparing the means of the 2 resting state datasets.
result_img = image.math_img("np.mean(img1, axis=-1) - np.mean(img2, axis=-1)",
                            img1=dataset.func[0],
                            img2=dataset.func[1])

plotting.plot_epi(result_img,
                  title="Comparing means of 2 resting state 4D images.")
plotting.show()
