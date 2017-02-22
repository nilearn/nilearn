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

from nistats.design_matrix import (create_simple_second_level_design,
                                   plot_design_matrix)
import pandas as pd

#########################################################################
# Create a simple experimental paradigm
# --------------------------------------
# We want to get the group result of a contrast for 20 subjects
column_names = ['map_name', 'subject_label', 'map_path']
n_subjects = 20
maps_name = ['contrast'] * n_subjects
subject_list = ['sub-%02d' % i for i in range(1, n_subjects + 1)]
maps_model = subject_list
maps_path = [''] * n_subjects
maps_table = pd.DataFrame({'map_name': maps_name,
                           'subject_label': maps_model,
                           'effects_map_path': maps_path})
##############################################################################
# Specify extra information about the subjects to create confounders
# Withoug confounders the design matrix would be correspond to a simple t-test
extra_info_subjects = pd.DataFrame({'subject_label': subject_list,
                                    'age': range(15, 35),
                                    'sex': [0, 1] * 10})


#########################################################################
# Create a second level design matrix
# -----------------------------------

design_matrix = create_simple_second_level_design(maps_table,
                                                  extra_info_subjects)

# plot the results
ax = plot_design_matrix(design_matrix)
ax.set_title('Second level design matrix', fontsize=12)
ax.set_ylabel('maps')
plt.tight_layout()
plt.show()
