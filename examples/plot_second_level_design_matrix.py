"""
Example of second level design matrix
=====================================

Requires matplotlib

Author : Martin Perez-Guevara: 2016

"""

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise RuntimeError("This script needs the matplotlib library")

from nistats.design_matrix import (create_second_level_design,
                                   plot_design_matrix)
import pandas as pd

#########################################################################
# Create a simple experimental paradigm
# --------------------------------------
# We want to get the group result of a contrast for 20 subjects
n_subjects = 20
subjects_label = ['sub-%02d' % i for i in range(1, n_subjects + 1)]

##############################################################################
# Specify extra information about the subjects to create confounders
# Without confounders the design matrix would correspond to a one sample test
extra_info_subjects = pd.DataFrame({'subject_label': subjects_label,
                                    'age': range(15, 15 + n_subjects),
                                    'sex': [0, 1] * int(n_subjects / 2)})


#########################################################################
# Create a second level design matrix
# -----------------------------------

design_matrix = create_second_level_design(subjects_label, extra_info_subjects)

# plot the results
ax = plot_design_matrix(design_matrix)
ax.set_title('Second level design matrix', fontsize=12)
ax.set_ylabel('maps')
plt.tight_layout()
plt.show()
