import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import random
import math
from statsmodels.tsa.holtwinters import SimpleExpSmoothing


precalc_dir_name = "./precalc_distance_summation_midv2019/"
precalc_dir = os.listdir(precalc_dir_name)
precalc_file = random.choice(precalc_dir)
print(precalc_file)
precalc_f = precalc_dir_name + precalc_file

with open(precalc_f) as js:
    data = json.load(js)

axis_x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
axis_x_2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
#axis_x = np.arange(1, len(data)+1, 1)
#axis_x_2 = np.arange(2, len(data)+2, 1)


    
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

def lin_LSE(x, y, coef):
    x_e = []
    for i in range(len(x)):
        x_e.append(math.e**(x[i] * coef))
    sum_x = 0
    sum_x2 = 0
    sum_y = 0
    sum_xy = 0
    for i in range(len(x_e)):
        sum_x += x_e[i]
        sum_x2 += x_e[i]**2
        sum_y += y[i]
        sum_xy += x_e[i] * y[i]
    det = sum_x2 * len(x_e) - sum_x * sum_x
    det_1 =  -1 * (sum_x * sum_y - sum_xy * len(x_e))
    det_2 = sum_x2 * sum_y - sum_xy * sum_x
    return [det_1/det, det_2/det]


def F(a, b, c, x, y):
    res = 0
    for i in range(len(x)):
        res += (a * math.e**(b*x[i]) + c - y[i])**2
    return res

def trenar_search_exp(f, x, y, x_L, x_R):
    eps = 1e-6
    left_ = x_L
    right_ = x_R
    while right_ > left_+eps:
        t = (right_ - left_)/3
        a = left_ + t
        b = right_ - t
        a_coefs = lin_LSE(x, y, a)
        b_coefs = lin_LSE(x, y, b)
        if f(a_coefs[0], a, a_coefs[1], x, y) < f(b_coefs[0], b, b_coefs[1], x, y):
            right_ = b
        else:
            left_ = a
    fin_k, fin_c = lin_LSE(x, y, (left_+right_)/2) 
    return [(left_+right_)/2, fin_k, fin_c]
    
def LSM_exp(time_series, window):
    if window > len(time_series):
        window = len(time_series)    
    time_series = time_series[-1*window:]
    x = []
    for i in range(len(time_series)):
        x.append(i+1)
    #x = np.arange(1, len(time_series)+1, 1)
    fin_coefs = trenar_search_exp(F, x, time_series, -10, 10)
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
    


data_ = [1, 0.46153846153846156, 0.2413073884407883, 0.10770967367013787, 0.07700750592137354, 0.05094869585348801, 0.03649031598765177, 0.03305359048477479, 0.06817662151845572, 0.023501736923423982, 0.04493310985928079, 0.042189709937925673, 0.020371837617953647, 0.023918397709030238, 0.028416117428578636, 0.016137991859223867, 0.021189357351962045, 0.01569411116843049, 0.01220096278892372, 0.013755405534419927, 0.010860387690834406, 0.020114855373696915, 0.010125973612102485, 0.01151475415192085, 0.018721230522486632, 0.02291392071377881, 0.007359811323309845, 0.006655482039178563, 0.0054911625657606145, 0.008595925853987073]
sample = []
plt.scatter(axis_x, data_, label="Data")
#a = lin_LSE(axis_x, data_, 1)
a = LSM_exp(data_, 30)
print(a)
for i in range(len(axis_x)):
    sample.append(a[1] * math.e**(a[0] * axis_x[i]) + a[2])
plt.plot(axis_x, sample, label = "Fit")
plt.legend()
plt.show() 