import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt

def SES(time_series, smoothing_coeficient ):
    s = np.zeros(len(time_series)+1)
    s[0] = time_series[0]
    for i in range(1, len(time_series)):
        s[i] = s[i-1] + smoothing_coeficient * (time_series[i] - s[i-1])
    s[len(time_series)] = s[len(time_series)-1] + smoothing_coeficient * (time_series[-1] - s[len(time_series)-1])
    return s   

sample = np.random.random_sample((100,))
axis_x = np.arange(1, 102, 1)

'''
TESTS FOR SIMPLE EXPONENTIAL SMOOTHING

smoothing09 = SES(sample, 0.9)
smoothing08 = SES(sample, 0.8)
smoothing07 = SES(sample, 0.7)
smoothing06 = SES(sample, 0.6)
smoothing05 = SES(sample, 0.5)
smoothing04 = SES(sample, 0.4)
smoothing03 = SES(sample, 0.3)
smoothing02 = SES(sample, 0.2)
smoothing01 = SES(sample, 0.1)

plt.plot(axis_x[0:100], sample, label = "original")
plt.plot(axis_x, smoothing09, label = "a = 0.9")
plt.plot(axis_x, smoothing08, label = "a = 0.8")
plt.plot(axis_x, smoothing07, label = "a = 0.7")
plt.plot(axis_x, smoothing06, label = "a = 0.6")
plt.plot(axis_x, smoothing05, label = "a = 0.5")
plt.plot(axis_x, smoothing04, label = "a = 0.4")
plt.plot(axis_x, smoothing03, label = "a = 0.3")
plt.plot(axis_x, smoothing02, label = "a = 0.2")
plt.plot(axis_x, smoothing01, label = "a = 0.1")
'''

plt.legend()
plt.show()


