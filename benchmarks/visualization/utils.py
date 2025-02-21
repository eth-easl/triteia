import numpy as np
import scipy.stats as st

colors = [
    (0, 0, 0),
    (0.7, 0, 0),
    (0.9, 0.3, 0),
    (1, 0.8, 0),
    (0.6, 0.8, 0.2),
    (0.2, 0.6, 0.4),
    (0, 0.4, 0.6),
]

colors_multiple = [
    (0, 0, 0),
    (0.7, 0, 0),
    (0.9, 0.3, 0),
    (1, 0.8, 0),
    (0.6, 0.8, 0.2),
    (0.2, 0.6, 0.4),
    (0, 0.4, 0.6),
]



def compute_confidence_interval(measurements, confidence=0.95):
    mean = np.mean(measurements)
    (lower_ci, upper_ci) = st.t.interval(confidence, len(measurements)-1, loc=np.mean(measurements), scale=st.sem(measurements))
    return [mean - lower_ci, upper_ci - mean]