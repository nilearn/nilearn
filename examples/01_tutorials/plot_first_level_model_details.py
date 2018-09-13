"""Studying firts-level-model details in a trials-and-error fashion
================================================================

In this tutorial, we study the parametrization of the first-level
model used for fMRI data analysis and clarify their impact on the
results of the analysis.

We use an exploratory approach, in which we incrementally include some
new features in the analysis and look at the outcome, i.e. the
resulting brain maps.

Readers without prior experience in fMRI data analysis should first
run the plot_sing_subject_single_run tutorial to get a bit more
familiar with the base concepts, and only then run thi script.

To run this example, you must launch IPython via ``ipython
--matplotlib`` in a terminal, or use ``jupyter-notebook``.

.. contents:: **Contents**
    :local:
    :depth: 1

"""

###############################################################################
# Retrieving the data
# -------------------
#
# We use a so-called localizer dataset, which consists in a 5-minutes
# acquisition of a fast event-related dataset. 


###############################################################################
# Running a base model
# -------------------
