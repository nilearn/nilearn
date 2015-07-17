"""
Smoothing an image
===================

Here we smooth a mean EPI image and plot the result

As we vary the smoothing FWHM, note how we decrease the amount of noise,
but also loose spatial details. In general, the best amount of smoothing
for a given analysis depends on the spatial extent of the effects that
are expected.

"""

from nilearn import datasets, plotting, image

data = datasets.fetch_adhd(n_subjects=1)

# Print basic information on the dataset
print('First subject functional nifti image (4D) are located at: %s' %
      data.func[0])

first_epi_file = data.func[0]

# First the compute the mean image, from the 4D series of image
mean_func = image.mean_img(first_epi_file)

# Then we smooth, with a varying amount of smoothing, from none to 20mm
# by increments of 5mm
for smoothing in range(0, 25, 5):
    smoothed_img = image.smooth_img(mean_func, smoothing)
    plotting.plot_epi(smoothed_img,
                      title="Smoothing %imm" % smoothing)


plotting.show()
