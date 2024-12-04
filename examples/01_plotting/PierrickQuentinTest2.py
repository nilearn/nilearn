"""
A test to see if i can reproduce what GillesSchneider did (see issue 2136)
and then if it is possible to use a colorbar on add_contours (from nilearn plotting displays _slicers.py)
=====================================================================

Based on plot_demo_glass_brain_extensive.py and the work of GillesSchneider


"""

# Load data
# ---------------------------

from nilearn import datasets

# import 
# ---------------------------

from nilearn import plotting

#1
stat_imgQ = datasets.load_sample_motor_activation_image()

#2
from nilearn.datasets.neurovault import fetch_neurovault_motor_task
import numpy as np
motor_images = fetch_neurovault_motor_task()
stat_img = motor_images.images[0]

#reproduce the work of GillesSchneider
# ---------------------------

display = plotting.plot_glass_brain(None, colorbar=False, title='plot_stat_map GillesSchneider')
#Same colormap 
cmap = 'YlOrRd'
display.add_contours(stat_img, filled=False, levels = [1, 2], cmap=cmap)
display.add_contours(stat_img, filled=True, levels = [2, 3], colors='r')
display.add_contours(stat_img, show_level=False, filled=True, levels = [3, 6], cmap=cmap)
display.add_contours(stat_img, show_level=True, filled=True, levels = [7, np.inf], cmap=cmap)


display2 = plotting.plot_glass_brain(None, colorbar=False, title='test-2.0 : add_overlay without colorbar')
display2.add_overlay(stat_img,cmap=cmap)

display3 = plotting.plot_glass_brain(None, colorbar=False, title='test-2.1 : add_overlay with colorbar')
display3.add_overlay(stat_img,colorbar=True,cmap=cmap)

display4 = plotting.plot_glass_brain(None, colorbar=False, title='test-3.0 : add_contours2 without colorbar and filled=True')
display4.add_contours(stat_img,filled=True,cmap=cmap)

display5 = plotting.plot_glass_brain(None, colorbar=False, title='test-3.1 : add_contours2 with colorbar and filled=True')
display5.add_contours(stat_img,colorbar=True,filled=True,cmap=cmap)

display6 = plotting.plot_glass_brain(None, colorbar=False, title='test-3.2 : add_contours2 with colorbar and filled=False')
display6.add_contours(stat_img,colorbar=True,cmap=cmap)

#based on plot_demo_glass_brain_extensive.py
# ---------------------------

#1
displayQ = plotting.plot_glass_brain(None, display_mode="lzry")
displayQ.add_contours(stat_imgQ, colorbar=True, filled=True)
displayQ.title("test-4.0")

displayQ0 = plotting.plot_glass_brain(None, plot_abs=False, display_mode="lzry")
displayQ0.add_contours(stat_imgQ, filled=True, levels=[-np.inf, -2.8], colors="b")
displayQ0.add_contours(stat_imgQ, filled=True, levels=[3.0], colors="r")
displayQ0.title("test-5.0 : Identical as plot_demo_glass_brain_extensive.py")

displayQ1 = plotting.plot_glass_brain(None, plot_abs=False, display_mode="lzry")
displayQ1.add_contours(stat_imgQ, colorbar=True, filled=True)
displayQ1.title("test-5.1")

displayQ2 = plotting.plot_glass_brain(None, plot_abs=False, display_mode="lzry")
displayQ2.add_contours(stat_imgQ, filled=True, levels=[-np.inf, -2.8], colors="b")
displayQ2.add_contours(stat_imgQ, filled=True, levels=[3.0], colors="r")
displayQ2.title("test-5.2 : plot the same as test-5.0")

displayQ2 = plotting.plot_glass_brain(None, plot_abs=False, display_mode="lzry")
displayQ2.add_contours(stat_imgQ, filled=True, levels=[-np.inf, -2.8], colors="b")
displayQ2.add_contours(stat_imgQ, filled=True, levels=[3.0], colors="r")
displayQ2.add_contours(stat_imgQ, colorbar=True, filled=True)
displayQ2.title("test-5.3 : something possible but not relevant")

displayQ2 = plotting.plot_glass_brain(None, plot_abs=False, display_mode="lzry")
displayQ2.add_contours(stat_imgQ, colorbar=True, filled=True)
displayQ2.add_contours(stat_imgQ, filled=True, levels=[-np.inf, -2.8], colors="b")
displayQ2.title("test-5.3 : also something possible but not relevant")

displayQ3 = plotting.plot_glass_brain(None, plot_abs=False, display_mode="lzry")
displayQ3.add_contours(stat_imgQ, filled=True, levels=[-np.inf, -2.8], colors="b")
displayQ3.add_contours(stat_imgQ, colorbar=True, filled=True, levels=[3.0], colors="r") #raised an error : ok
displayQ3.title("test-5.4 : you should never see this, an error should occur before")


plotting.show()