import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import random

precalc_dir_name = "./precalc_distance_summation_midv500/"
precalc_dir = os.listdir(precalc_dir_name)
precalc_file = random.choice(precalc_dir)
#precalc_file = "./precalc_distance_summation_midv500/date_CA01_field05_summation_precalc_dists.json"
print(precalc_file)
precalc_f = precalc_dir_name + precalc_file

with open(precalc_f) as js:
    data = json.load(js)
    
    
def SES(time_series, smoothing_coeficient):
    s = np.zeros(len(time_series)+1)
    s[0] = time_series[0]
    for i in range(1, len(time_series)):
        s[i] = s[i-1] + smoothing_coeficient * (time_series[i] - s[i-1])
    s[len(time_series)] = s[len(time_series)-1] + smoothing_coeficient * (time_series[-1] - s[len(time_series)-1])
    return s

axis_x = np.arange(1, len(data)+1, 1)

SES
final_sample = []
for i in range(len(data)-1):
    test = SES(data[0:i+1], 0.9)
    final_sample.append(test[-1])

final_sample.append(SES(data, 0.9)[-1])

plt.scatter(axis_x[6:], data[6:], label="Data")
print(len(final_sample))
plt.scatter(axis_x[6:], final_sample[6:], label = "Predictions")
plt.legend()
plt.show()

    
def LSM_AR(time_series):
    x = np.arange(1, len(time_series)+1, 1)
    a_11 = 2*len(time_series)
    a_12 = 0
    a_21 = 0
    a_22 = 0
    b_1 = 0
    b_2 = 0    
    for i in range(len(x)):
        a_12 += x[i]
        a_21 += x[i]
        a_22 += x[i] * x[i]
        b_1 += time_series[i]
        b_2 += time_series[i] * x[i]        
    a_12 *= 2
    a_22 *= 2
    b_1 *= 2
    b_2 *= 2
    a = np.array([[a_11, a_12], [a_21, a_22]])
    b = np.array([b_1, b_2])
    x = np.linalg.solve(a, b)
    return x

LSM_AR
final_sample = []
for i in range(len(data)-1):
    test = LSM_AR(data[0:i+1])
    final_sample.append(test[0] + axis_x[i+1] * test[1])
fin_elem = LSM_AR(data)
final_sample.append(fin_elem[0] + (axis_x[-1]+1) * test[1])

plt.scatter(axis_x[5:], data[5:], label="Data")
print(len(final_sample))
plt.scatter(axis_x[5:], final_sample[5:], label = "Predictions")
plt.legend()
plt.show()


def LSM_SQR(time_series):
    x = np.arange(1, len(time_series)+1, 1)
    frac_top = 0
    frac_bottom = 0
    for i in range(len(x)):
        frac_top += time_series[i] * (x[i] ** 2)
        frac_bottom += x[i]**4
    b = frac_top/frac_bottom
    return b

LSM_SQR
final_sample = []
for i in range(len(data)-1):
    test = LSM_SQR(data[0:i+1])
    final_sample.append(test * axis_x[i+1] * axis_x[i+1])

final_sample.append(LSM_SQR(data) * (axis_x[-1]+1 * axis_x[-1]+1))

plt.scatter(axis_x[5:], data[5:], label="Data")
print(len(final_sample))
plt.scatter(axis_x[5:], final_sample[5:], label = "Predictions")
plt.legend()
plt.show()

#martingale

final_sample = [0]

for i in range(len(data)-1):
    final_sample.append(data[i])

plt.scatter(axis_x[2:], data[2:], label="Data")
print(len(final_sample))
plt.scatter(axis_x[2:], final_sample[2:], label = "Predictions")

plt.legend()
plt.show()
'''
TEST FOR LSM_SQR

sample = np.random.random_sample((10,))

axis_x = np.arange(1, len(data)+1, 1)

def f(x, b):
    return b * (x ** 2)

test = LSM_SQR(data)
y = f(axis_x, test)

plt.scatter(axis_x, data, label = "data")
plt.plot(axis_x, y, label = "fit")


'''

'''
TESTS FOR SIMPLE EXPONENTIAL SMOOTHING
sample = np.random.random_sample((30,))
axis_x = np.arange(1, 31, 1)

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

'''
TESTS FOR LINEAR AUTOREGRESSION
sample = np.random.random_sample((30,))
axis_x = np.arange(1, 31, 1)

plt.scatter(axis_x, sample, label = "data")
testing_solution = LSM_AR(sample)
def f(x, k, b):
    return k*x + b

y = f(axis_x, testing_solution[1], testing_solution[0])

plt.plot(axis_x, y, label = "fit")
'''


