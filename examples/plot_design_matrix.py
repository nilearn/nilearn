"""
Examples of design matrices
===========================

Three examples of design matrices specification and computation
(event-related design, block design, FIR design)

Requires matplotlib

Author : Bertrand Thirion: 2009-2015
"""
import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nistats.design_matrix import make_design_matrix, plot_design_matrix
import pandas as pd


#########################################################################
# Define parameters
# ----------------------------------
# first we define parameters related to the images acquisition
tr = 1.0
n_scans = 128
frame_times = np.arange(n_scans) * tr

#########################################################################
# then we define parameters related to the experimental design

conditions = ['c0', 'c0', 'c0', 'c1', 'c1', 'c1', 'c3', 'c3', 'c3']
onsets = [30., 70., 100., 10., 30., 90., 30., 40., 60.]
motion = np.cumsum(np.random.randn(n_scans, 6), 0)  # simulate motion
add_reg_names = ['tx', 'ty', 'tz', 'rx', 'ry', 'rz']

#########################################################################
# Create design matrices
# -------------------------------------
# The same parameters allow us to obtain a variety of design matrices
# First we compute an event-related design matrix
paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onsets})

hrf_model = 'glover'
X1 = make_design_matrix(
    frame_times, paradigm, drift_model='polynomial', drift_order=3,
    add_regs=motion, add_reg_names=add_reg_names, hrf_model=hrf_model)

#########################################################################
# Now we compute a block design matrix. We add duration to create the blocks.
duration = 7. * np.ones(len(conditions))
paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onsets,
                         'duration': duration})

X2 = make_design_matrix(frame_times, paradigm, drift_model='polynomial',
                        drift_order=3, hrf_model=hrf_model)

#########################################################################
# Finally we compute a FIR model
paradigm = pd.DataFrame({'trial_type': conditions, 'onset': onsets})
hrf_model = 'FIR'
X3 = make_design_matrix(frame_times, paradigm, hrf_model='fir',
                        drift_model='polynomial', drift_order=3,
                        fir_delays=np.arange(1, 6))

#########################################################################
# Here the three designs side by side
fig, (ax1, ax2, ax3) = plt.subplots(figsize=(10, 6), nrows=1, ncols=3)
plot_design_matrix(X1, ax=ax1)
ax1.set_title('Event-related design matrix', fontsize=12)
plot_design_matrix(X2, ax=ax2)
ax2.set_title('Block design matrix', fontsize=12)
plot_design_matrix(X3, ax=ax3)
ax3.set_title('FIR design matrix', fontsize=12)
plt.subplots_adjust(left=0.08, top=0.9, bottom=0.21, right=0.96, wspace=0.3)
plt.show()
