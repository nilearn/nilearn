import numpy as np
import matplotlib.pyplot as plt
from nistats import hemodynamic_models



frame_times = np.linspace(0, 30, 61)

onset, amplitude, duration = 0., 1., 1. 
exp_condition = np.array((onset, amplitude, duration)).reshape(3, 1)
hrf_model = 'canonical with derivative'

signal, name = hemodynamic_models.compute_regressor(
    exp_condition, hrf_model, frame_times, con_id='cond',
    oversampling=2, fir_delays=None, min_onset=-24)

plt.plot(frame_times, signal)
plt.xlabel('time (s)')
plt.title(name)
plt.show()
