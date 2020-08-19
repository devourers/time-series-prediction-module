import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import random
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


precalc_dir_name = "./precalc_distance_summation_midv2019/"
precalc_dir = os.listdir(precalc_dir_name)
precalc_file = random.choice(precalc_dir)
#precalc_file = "./precalc_distance_summation_midv500/date_CA01_field05_summation_precalc_dists.json"
print(precalc_file)
precalc_f = precalc_dir_name + precalc_file

with open(precalc_f) as js:
    data = json.load(js)

#data[0] = 1 
axis_x = np.arange(1, len(data)+1, 1)
axis_x_2 = np.arange(2, len(data)+2, 1)


    
def MA(time_series, window):
    res = 0
    if window > len(time_series):
        window = len(time_series)
        window_l = time_series[-1*window:]
    else:
        window_l = time_series[-1*window:]
    for i in range (len(window_l)):
        res += window_l[i]
    res /= window
    return res
    

#MA
final_sample = []
for i in range(len(data)-1):
    test = MA(data[0:i+1], 2)
    final_sample.append(test)
final_sample.append(MA(data, 2))
plt.scatter(axis_x, data, label="Data")
plt.scatter(axis_x_2, final_sample, label = "Predictions MA")
plt.legend()
plt.show()   

def SES(time_series, smoothing_coeficient):
    s = np.zeros(len(time_series)+1)
    s[0] = time_series[0]
    for i in range(1, len(time_series)):
        s[i] = s[i-1] + smoothing_coeficient * (time_series[i] - s[i-1])
    s[len(time_series)] = s[len(time_series)-1] + smoothing_coeficient * (time_series[-1] - s[len(time_series)-1])
    return s


#SES
final_sample = []
for i in range(len(data)-1):
    test = SES(data[0:i+1], 0.6)
    final_sample.append(test[-1])

final_sample.append(SES(data, 0.9)[-1])

plt.scatter(axis_x, data, label="Data")
plt.scatter(axis_x_2, final_sample, label = "Predictions SES")
plt.legend()
plt.show()

    
def LSM_AR(time_series, window):
    if window > len(time_series):
        window = len(time_series)
    time_series = time_series[-1 * window:]
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

#LSM_AR
final_sample = []
for i in range(len(data)-1):
    test = LSM_AR(data[0:i+1], 10)
    final_sample.append(test[0] + axis_x[i+1] * test[1])
fin_elem = LSM_AR(data, 10)
final_sample.append(fin_elem[0] + (axis_x[-1]+1) * test[1])

plt.scatter(axis_x, data, label="Data")
plt.scatter(axis_x_2, final_sample, label = "Predictions LSM_AR")
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

#LSM_SQR
final_sample = []
for i in range(len(data)-1):
    test = LSM_SQR(data[0:i+1])
    final_sample.append(test * axis_x[i+1] * axis_x[i+1])

final_sample.append(LSM_SQR(data) * (axis_x[-1]+1 * axis_x[-1]+1))

plt.scatter(axis_x, data, label="Data")
plt.scatter(axis_x_2, final_sample, label = "Predictions LSM_SQR")
plt.legend()
plt.show()

#martingale

final_sample = [0]

for i in range(len(data)-1):
    final_sample.append(data[i])

plt.scatter(axis_x, data, label="Data")
plt.scatter(axis_x, final_sample, label = "Predictions Martingale")

plt.legend()
plt.show()

#LSM exponential

def F(a, b, c, x, y):
    res = 0
    for i in range(len(x)):
        res += (a * np.exp(b*x[i]) + c - y[i])**2
    return res

def trenar_search_exp(f, lin_k, c, x, y, left, right, eps):
    while abs(right-left) > eps:
        a = (left * 2 + right)/3
        b = (left + right * 2)/3
        if f(lin_k, a, c, x, y) < f(lin_k, b, c, x, y):
            #calculate linear and free coeficients for F
            x_e = np.exp(a*x)
            a_11 = 2*len(y)
            a_12 = 0
            a_21 = 0
            a_22 = 0
            b_1 = 0
            b_2 = 0    
            for i in range(len(x)):
                a_12 += x_e[i]
                a_21 += x_e[i]
                a_22 += x_e[i] * x_e[i]
                b_1 += y[i]
                b_2 += y[i] * x_e[i]        
            a_12 *= 2
            a_22 *= 2
            b_1 *= 2
            b_2 *= 2
            A = np.array([[a_11, a_12], [a_21, a_22]])
            B = np.array([b_1, b_2])
            lin_sol = np.linalg.solve(A, B)
            lin_k = lin_sol[1]
            c = lin_sol[0]
            left = a
        else:
            #calculate linear and free coeficients for F
            x_e = np.exp(b*x)
            a_11 = 2*len(y)
            a_12 = 0
            a_21 = 0
            a_22 = 0
            b_1 = 0
            b_2 = 0    
            for i in range(len(x)):
                a_12 += x_e[i]
                a_21 += x_e[i]
                a_22 += x_e[i] * x_e[i]
                b_1 += y[i]
                b_2 += y[i] * x_e[i]        
            a_12 *= 2
            a_22 *= 2
            b_1 *= 2
            b_2 *= 2
            A = np.array([[a_11, a_12], [a_21, a_22]])
            B = np.array([b_1, b_2])
            lin_sol = np.linalg.solve(A, B)
            lin_k = lin_sol[1]
            c = lin_sol[0]
            right = b
    return [(left+right)/2, lin_k, c]
    
def LSM_exp(time_series, window):
    if window > len(time_series):
        window = len(time_series)    
    time_series = time_series[-1*window:]
    x = np.arange(1, len(time_series)+1, 1)
    fin_coefs = trenar_search_exp(F, 1, 1, x, time_series, -1, 1, 0.000000001)
    #return fin_coefs[1] * (np.exp(fin_coefs[0] * (x[-1]+1))) + fin_coefs[2]
    return fin_coefs
    
'''    
plt.scatter(axis_x, data, label="Data")

fin_sample = []
for i in range(len(data) - 1):
    test = LSM_exp(data[0:i+1], 5)
    fin_sample.append(test)
fin_sample.append(LSM_exp(data, 5))
print(fin_sample)
plt.scatter(axis_x_2, fin_sample, label = "Predction LSM Exponential")
plt.legend()
plt.show() 
'''
    
plt.scatter(axis_x, data, label="Data")


a = LSM_exp(data, 30)
print(a)
sample = a[1] * np.exp(a[0] * axis_x_2) + a[2]
plt.plot(axis_x, sample, label = "Fit")
plt.legend()
plt.show() 