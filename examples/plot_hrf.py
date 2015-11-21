""" 
Example of hemodynamic reponse functions.
We consider the hrf model in SPM together with the hrf shape proposed
by G.Glover, as well as their time and dispersion derivatives.

Author: Bertrand Thirion, 2015.
"""
print __doc__

import numpy as np
import matplotlib.pyplot as plt
from nistats import hemodynamic_models


# parameters
frame_times = np.linspace(0, 30, 61)
onset, amplitude, duration = 0., 1., 1.
stim = np.zeros_like(frame_times)
stim[(frame_times > onset) * (frame_times <= onset + duration)] = amplitude
exp_condition = np.array((onset, duration, amplitude)).reshape(3, 1)
hrf_models = ['glover + derivative', 'spm + derivative + dispersion']


fig = plt.figure(figsize=(9, 4))
# sample the hrf
for i, hrf_model in enumerate(hrf_models):
    signal, name = hemodynamic_models.compute_regressor(
        exp_condition, hrf_model, frame_times, con_id='main',
        oversampling=16)

    plt.subplot(1, 2, i)
    plt.fill(frame_times, stim, 'k', alpha=.5, label='stimulus')
    for j in range(signal.shape[1]):
        plt.plot(frame_times, signal.T[j], label=name[j])
    plt.xlabel('time (s)')
    plt.legend(loc=1)
    plt.title(hrf_model)

plt.subplots_adjust(bottom=.12)
fig.show()
