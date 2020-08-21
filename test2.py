import os, sys, json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

ALL_DATASETS = ['midv500', 'midv2019', 'ic15', 'yvt']
ESTIMATION_PLOT_DATASETS = [['midv500', 'midv2019'], ['ic15', 'yvt']]
TIMING_PLOT_DATASETS = [['midv500', 'midv2019'], ['ic15', 'yvt']]

EPPS_DATASETS = ['midv500', 'midv2019']
#EPPS_DATASETS = ['midv500']
#EPPS_DATASETS = ['ic15', 'yvt']

EPPS_YLIMITS = {
    'midv500': [0.06, 0.125],
    'midv2019': [0.09, 0.25],
    'ic15': [0.15, 0.35],
    'yvt': [0.195, 0.24]
}

DATASET_LABELS = {
    'midv500': 'MIDV-500',
    'midv2019': 'MIDV-2019',
    'ic15': 'IC15-Train',
    'yvt': 'YVT'
}

METHODS = ['summation', 'treap', 'base']
#METHODS = ['summation']

PRECALC_DIRECTORIES = {}
for dataset in ALL_DATASETS:
    PRECALC_DIRECTORIES[dataset] = {
        'base': './precalc_base_%s' % dataset,
        'summation': './precalc_summation_%s' % dataset,
        'treap': './precalc_treap_%s' % dataset
    }
    
PRECALC_DISTANCE_DIRECTORIES = {}
for dataset in ALL_DATASETS:
    PRECALC_DISTANCE_DIRECTORIES[dataset] = {
        'base': './precalc_distance_base_%s' % dataset,
        'summation': './precalc_distance_summation_%s' % dataset,
        'treap': './precalc_distance_treap_%s' % dataset
    }

PLOT_COLOR = { 'base': '0.0', 'summation': '0.5', 'treap': '0.2' }
PLOT_COLOR_ROC = { 'base': 'b', 'summation': 'm', 'treap': 'r' }
PLOT_LINESTYLE = { 'base': '-', 'summation': '--', 'treap': ':' }
PLOT_MARKER = { 'base': None, 'summation': 'o', 'treap': None }
PLOT_MARKERSIZE = { 'base': None, 'summation': 6, 'treap': None }
PLOT_LINEWIDTH = { 'base': 1.5, 'summation': 1.5, 'treap': 2.0 }

PLOT_LABEL = {
    'base': 'Base method', 
    'summation': 'Method A',
    'treap': 'Method B'
}
PLOT_LABEL_ROC = {
    'base': 'Base method ROC', 
    'summation': 'Method A ROC',
    'treap': 'Method B ROC'
}    

def SES(time_series, smoothing_coeficient):
    s = np.zeros(len(time_series)+1)
    s[0] = time_series[0]
    for i in range(1, len(time_series)):
        s[i] = s[i-1] + smoothing_coeficient * (time_series[i] - s[i-1])
    s[len(time_series)] = s[len(time_series)-1] + smoothing_coeficient * (time_series[-1] - s[len(time_series)-1])
    return s

def LSM_AR(time_series):
    x = np.arange(1, len(time_series)+1, 1)
    #first row
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
    sltn = np.linalg.solve(a, b)
    return sltn[0] + sltn[1] * (x[-1] + 1) 

def LSM_SQR(time_series):
    x = np.arange(1, len(time_series)+1, 1)
    frac_top = 0
    frac_bottom = 0
    for i in range(len(x)):  
        frac_top += time_series[i] * (x[i] ** 2)
        frac_bottom += x[i]**4
    b = frac_top/frac_bottom
    return b*(x[-1] + 1)

def MA(time_series):
    res = 0
    for i in range (len(time_series)):
        res += time_series[i]
    res /= len(time_series)
    return res

def lin_LSE(x, y, coef):
    if len(x) == 1:
        return [0, y[0]]
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
    #window_ = window
    #if window > len(time_series):
    #    window_ = len(time_series)    
    #time_series_ = time_series[-1*window_:]
    time_series_ = time_series
    x = []
    for i in range(len(time_series_)):
        x.append(i+1)
    fin_coefs = trenar_search_exp(F, x, time_series_, -10, 10)
    return fin_coefs[1] * (math.e**(fin_coefs[0] * (x[-1]+1))) + fin_coefs[2]

def lin_LSE_h(x, y, coef):
    if len(x) == 1:
        return [0, y[0]]    
    x_h = []
    for i in range(len(x)):
        x_h.append(1/(x[i] + coef))
    sum_x = 0
    sum_x2 = 0
    sum_y = 0
    sum_xy = 0
    for i in range(len(x_h)):
        sum_x += x_h[i]
        sum_x2 += x_h[i]**2
        sum_y += y[i]
        sum_xy += x_h[i] * y[i]
    det = sum_x2 * len(x_h) - sum_x * sum_x
    det_1 =  -1 * (sum_x * sum_y - sum_xy * len(x_h))
    det_2 = sum_x2 * sum_y - sum_xy * sum_x
    return [det_1/det, det_2/det]


def G(a, b, c, x, y):
    res = 0
    for i in range(len(x)):
        res += (a/(x[i] + b) + c - y[i])**2
    return res

def trenar_search_hprbl(f, x, y, x_L, x_R):
    eps = 1e-6
    left_ = x_L
    right_ = x_R
    while right_ > left_+eps:
        t = (right_ - left_)/3
        a = left_ + t
        b = right_ - t
        a_coefs = lin_LSE_h(x, y, a)
        b_coefs = lin_LSE_h(x, y, b)
        if f(a_coefs[0], a, a_coefs[1], x, y) < f(b_coefs[0], b, b_coefs[1], x, y):
            right_ = b
        else:
            left_ = a
    fin_k, fin_c = lin_LSE_h(x, y, (left_+right_)/2) 
    return [(left_+right_)/2, fin_k, fin_c]
    
def LSM_hprbl(time_series, window):
    #window_ = window
    #if window > len(time_series):
    #    window_ = len(time_series)    
    #time_series_ = time_series[-1*window_:]
    time_series_ = time_series
    x = []
    for i in range(len(time_series_)):
        x.append(i+1)
    #x = np.arange(1, len(time_series)+1, 1)
    fin_coefs = trenar_search_hprbl(G, x, time_series_, -10, 10)
    return fin_coefs[1] * (math.e**(fin_coefs[0] * (x[-1]+1))) + fin_coefs[2]
    #return fin_coefs

def collect_estimation_datapoints(method, dataset):
    '''
    Collects precalculated values for estimation 
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1

    x = [1.0 * (i + 1) for i in range(30)]
    y = [0.0 for i in range(30)]
    
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        for i in range(30):
            y[i] += (SMALL_DELTA + precalc_data[i][1]) / (i + 2)
    
    for i in range(30):
        y[i] /= len(precalc_files)
    
    return x, y, len(precalc_files)

plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams.update({'font.size': 12})

plt.clf()
plt.cla()

for i_plot, plot in enumerate(ESTIMATION_PLOT_DATASETS):
    plt.subplot(100 + 10 * len(ESTIMATION_PLOT_DATASETS) + i_plot + 1)
    plt.title(('%s) ' % chr(ord('a') + i_plot)) + ' and '.join([DATASET_LABELS[dataset] for dataset in plot]))

    X = {'summation': None, 'treap': None, 'base': None}
    Y = {'summation': None, 'treap': None, 'base': None}
    D = {'summation': 0, 'treap': 0, 'base': 0}
    
    for i_dataset, dataset in enumerate(plot):
        for method in METHODS:
            x, y, d = collect_estimation_datapoints(method, dataset)
            X[method] = x
            D[method] += d
            for i in range(len(y)):
                y[i] *= d
            if Y[method] is None:
                Y[method] = y
            else:
                for i in range(len(y)):
                    Y[method][i] += y[i]
                
    for method in METHODS:
        for i in range(len(Y[method])):
            Y[method][i] /= D[method]
    
    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.gca().set_xticks([1] + [5 * (i + 1) for i in range(6)])
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.01))
    plt.grid(which = 'minor', alpha = 0.3)
    plt.grid(which = 'major', alpha = 0.6)

    for method in METHODS:
        plt.plot(X[method], Y[method], \
                 label = PLOT_LABEL[method], \
                 color = PLOT_COLOR[method], \
                 linestyle = PLOT_LINESTYLE[method], \
                 marker = PLOT_MARKER[method], \
                 markersize = PLOT_MARKERSIZE[method], \
                 linewidth = PLOT_LINEWIDTH[method])

    plt.xlim([1, 30])
    plt.ylim([0.005, 0.065])

    plt.legend(loc='upper right', prop={'size': 11})

    plt.xlabel(r'Number of processed frame results')
    plt.ylabel('Mean estimation value')

plt.tight_layout(w_pad=1, h_pad=0)
plt.savefig('estimations-composite.pdf', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.show()

def collect_timing_datapoints(method, dataset):
    '''
    Collects timing values for estimation 
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    
    x = [1.0 * (i + 1) for i in range(30)]
    y = [0.0 for i in range(30)]
    
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        for i in range(30):
            y[i] += precalc_data[i][2] + precalc_data[i][3]
    
    for i in range(30):
        y[i] /= len(precalc_files)
    
    return x, y, len(precalc_files)

plt.rcParams['figure.figsize'] = (8, 4)
plt.rcParams.update({'font.size': 12})

plt.clf()
plt.cla()

for i_plot, plot in enumerate(TIMING_PLOT_DATASETS):
    plt.subplot(100 + 10 * len(TIMING_PLOT_DATASETS) + i_plot + 1)
    plt.title(('%s) ' % chr(ord('a') + i_plot)) + ' and '.join([DATASET_LABELS[dataset] for dataset in plot]))

    X = {'summation': None, 'treap': None, 'base': None}
    Y = {'summation': None, 'treap': None, 'base': None}
    D = {'summation': 0, 'treap': 0, 'base': 0}
    
    for i_dataset, dataset in enumerate(plot):
        for method in METHODS:
            x, y, d = collect_timing_datapoints(method, dataset)
            X[method] = x
            D[method] += d
            for i in range(len(y)):
                y[i] *= d
            if Y[method] is None:
                Y[method] = y
            else:
                for i in range(len(y)):
                    Y[method][i] += y[i]
    
    for method in METHODS:
        for i in range(len(Y[method])):
            Y[method][i] /= D[method]

    plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
    plt.gca().set_xticks([1] + [5 * (i + 1) for i in range(6)])
    plt.grid(which = 'minor', alpha = 0.3)
    plt.grid(which = 'major', alpha = 0.6)

    for method in METHODS:
        plt.plot(X[method], Y[method], \
                 label = PLOT_LABEL[method], \
                 color = PLOT_COLOR[method], \
                 linestyle = PLOT_LINESTYLE[method], \
                 marker = PLOT_MARKER[method], \
                 markersize = PLOT_MARKERSIZE[method], \
                 linewidth = PLOT_LINEWIDTH[method])

    plt.xlim([1, 30])
    plt.ylim([-0.05, 0.65])

    plt.legend(loc='upper right', prop={'size': 11})

    plt.xlabel(r'Number of processed frame results')
    plt.ylabel('Mean time in sec. per decision')

plt.tight_layout(w_pad=1, h_pad=0)
plt.savefig('timing-composite.pdf', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.show()

def collect_counting_stopper_epp(dataset):
    '''
    Collects expected performance profile for a simple stopper which 
    stops after a fixed number of processed frames
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset]['base']
    x = [1.0 * (i + 1) for i in range(30)]
    y = [0.0 for i in range(30)]
    
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        for i in range(30):
            y[i] += precalc_data[i][0]
    
    for i in range(30):
        y[i] /= len(precalc_files)
    
    return x, y

def collect_modelling_stopper_epp(method, dataset):
    '''
    Collects expected performance profile for a next combination result 
    modelling stopping method
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1
    
    DATAPOINTS_COUNT = 300
    MIN_THRESHOLD = -0.001
    MAX_THRESHOLD = 0.15
    THRESHOLDS = [MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD) * i / (DATAPOINTS_COUNT - 1) \
                  for i in range(DATAPOINTS_COUNT)]
    
    x = []
    y = []
    
    precalc = []
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    
    for threshold in THRESHOLDS:
        sum_clip_length = 0.0
        sum_error_level = 0.0
        
        for precalc_data in precalc:
            stopped = False
            clip_start = 1 if threshold <= 1.0 else 0
            for i in range(clip_start, 30):
                delta = (SMALL_DELTA + precalc_data[i][1]) / (i + 2)
                if delta <= threshold:
                    sum_clip_length += (i + 1)
                    sum_error_level += precalc_data[i][0]
                    stopped = True
                    break
            if not stopped:
                sum_clip_length += 30
                sum_error_level += precalc_data[-1][0]
                
        x.append(sum_clip_length / len(precalc))
        y.append(sum_error_level / len(precalc))
    
    return x, y

def collect_dist_stopper_epp(method, dataset):
    '''
    Collects expected performance profile for a next combination result 
    modelling stopping method with distance between them as a margin
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1
    
    DATAPOINTS_COUNT = 300
    MIN_THRESHOLD = -0.001
    MAX_THRESHOLD = 0.15
    THRESHOLDS = [MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD) * i / (DATAPOINTS_COUNT - 1) \
                  for i in range(DATAPOINTS_COUNT)]
    
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
                
    for threshold in THRESHOLDS:
        sum_clip_length = 0.0
        sum_error_level = 0.0
        j = 0
        for precalc_data in precalc:
            stopped = False
            clip_start = 1 if threshold <= 1.0 else 0
            for i in range(clip_start, 30):
                delta = precalc_dist[j][i]
                if delta <= threshold:
                    sum_clip_length += (i + 1)
                    sum_error_level += precalc_data[i][0]
                    stopped = True
                    break
            if not stopped:
                sum_clip_length += 30
                sum_error_level += precalc_data[-1][0]
            j += 1
                
        x.append(sum_clip_length / len(precalc))
        y.append(sum_error_level / len(precalc))
    
    return x, y


def exp_smth_stopper_epp(method, dataset):
    '''
    stopping method with TSP exponential smoothing as 
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1
    
    SMOOTHING_COEFICIENT = 0.9
    DATAPOINTS_COUNT = 300
    MIN_THRESHOLD = -0.001
    MAX_THRESHOLD = 0.15
    THRESHOLDS = [MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD) * i / (DATAPOINTS_COUNT - 1) \
                  for i in range(DATAPOINTS_COUNT)]
    
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
                
    for threshold in THRESHOLDS:
        sum_clip_length = 0.0
        sum_error_level = 0.0
        j = 0
        for precalc_data in precalc:
            stopped = False
            clip_start = 1 if threshold <= 1.0 else 0
            for i in range(clip_start, 30):
                if i == clip_start:
                    delta = SES([precalc_dist[j][i]], SMOOTHING_COEFICIENT)[-1]
                else:    
                    delta = SES(precalc_dist[j][0:i], SMOOTHING_COEFICIENT)[-1]
                if delta <= threshold:
                    sum_clip_length += (i + 1)
                    sum_error_level += precalc_data[i][0]
                    stopped = True
                    break
            if not stopped:
                sum_clip_length += 30
                sum_error_level += precalc_data[-1][0]
            j += 1
                
        x.append(sum_clip_length / len(precalc))
        y.append(sum_error_level / len(precalc))
    
    return x, y

def AR_stopper_epp(method, dataset):
    '''
    stopping method with TSP exponential smoothing as 
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1
    
    SMOOTHING_COEFICIENT = 0.6
    DATAPOINTS_COUNT = 300
    MIN_THRESHOLD = -0.001
    MAX_THRESHOLD = 0.15
    THRESHOLDS = [MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD) * i / (DATAPOINTS_COUNT - 1) \
                  for i in range(DATAPOINTS_COUNT)]
    
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
                
    for threshold in THRESHOLDS:
        sum_clip_length = 0.0
        sum_error_level = 0.0
        j = 0
        for precalc_data in precalc:
            stopped = False
            clip_start = 1 if threshold <= 1.0 else 0
            for i in range(clip_start, 30):
                if i == clip_start:
                    delta =  LSM_AR([precalc_dist[j][i]])
                else:    
                    delta = LSM_AR(precalc_dist[j][clip_start:i])
                if delta <= threshold:
                    sum_clip_length += (i + 1)
                    sum_error_level += precalc_data[i][0]
                    stopped = True
                    break
            if not stopped:
                sum_clip_length += 30
                sum_error_level += precalc_data[-1][0]
            j += 1
                
        x.append(sum_clip_length / len(precalc))
        y.append(sum_error_level / len(precalc))
    
    return x, y

def SQR_stopper_epp(method, dataset):
    '''
    stopping method with TSP exponential smoothing as 
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1
    
    SMOOTHING_COEFICIENT = 0.6
    DATAPOINTS_COUNT = 300
    MIN_THRESHOLD = -0.001
    MAX_THRESHOLD = 0.15
    THRESHOLDS = [MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD) * i / (DATAPOINTS_COUNT - 1) \
                  for i in range(DATAPOINTS_COUNT)]
    
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
                
    for threshold in THRESHOLDS:
        sum_clip_length = 0.0
        sum_error_level = 0.0
        j = 0
        for precalc_data in precalc:
            stopped = False
            clip_start = 1 if threshold <= 1.0 else 0
            for i in range(clip_start, 30):
                if i == clip_start:
                    delta =  LSM_SQR([precalc_dist[j][i]])
                else:    
                    delta = LSM_SQR(precalc_dist[j][clip_start:i])
                if delta <= threshold:
                    sum_clip_length += (i + 1)
                    sum_error_level += precalc_data[i][0]
                    stopped = True
                    break
            if not stopped:
                sum_clip_length += 30
                sum_error_level += precalc_data[-1][0]
            j += 1
                
        x.append(sum_clip_length / len(precalc))
        y.append(sum_error_level / len(precalc))
    
    return x, y

def MA_stopper_epp(method, dataset):
    '''
    stopping method with TSP exponential smoothing as 
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1
    
    SMOOTHING_COEFICIENT = 0.6
    DATAPOINTS_COUNT = 300
    MIN_THRESHOLD = -0.001
    MAX_THRESHOLD = 0.15
    THRESHOLDS = [MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD) * i / (DATAPOINTS_COUNT - 1) \
                  for i in range(DATAPOINTS_COUNT)]
    
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
                
    for threshold in THRESHOLDS:
        sum_clip_length = 0.0
        sum_error_level = 0.0
        j = 0
        for precalc_data in precalc:
            stopped = False
            clip_start = 1 if threshold <= 1.0 else 0
            for i in range(clip_start, 30):
                if i == clip_start:
                    delta =  MA([precalc_dist[j][i]])
                else:    
                    delta = MA(precalc_dist[j][clip_start:i])
                if delta <= threshold:
                    sum_clip_length += (i + 1)
                    sum_error_level += precalc_data[i][0]
                    stopped = True
                    break
            if not stopped:
                sum_clip_length += 30
                sum_error_level += precalc_data[-1][0]
            j += 1
                
        x.append(sum_clip_length / len(precalc))
        y.append(sum_error_level / len(precalc))
    
    return x, y

def double_dist_stopper_epp(method, dataset):
    '''
    Collects expected performance profile for a next combination result 
    modelling stopping method with distance between them as a margin
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1
    
    DATAPOINTS_COUNT = 300
    MIN_THRESHOLD = -0.001
    MAX_THRESHOLD = 0.15
    THRESHOLDS = [MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD) * i / (DATAPOINTS_COUNT - 1) \
                  for i in range(DATAPOINTS_COUNT)]
    
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
    
    for threshold in THRESHOLDS:
        sum_clip_length = 0.0
        sum_error_level = 0.0
        j = 0
        for precalc_data in precalc:
            double_dist = 0
            stopped = False
            clip_start = 1 if threshold <= 1.0 else 0
            for i in range(clip_start, 30):
                delta = precalc_dist[j][i]
                if delta <= threshold:
                    double_dist += 1
                    if double_dist == 2:
                        sum_clip_length += (i + 1)
                        sum_error_level += precalc_data[i][0]
                        stopped = True
                        break
                elif double_dist == 1:
                    double_dist = 0
            if not stopped:
                sum_clip_length += 30
                sum_error_level += precalc_data[-1][0]
            j += 1
                
        x.append(sum_clip_length / len(precalc))
        y.append(sum_error_level / len(precalc))
    
    return x, y

def LSM_exp_stopper_epp(method, dataset):
    '''
    stopping method with LSM exponential smoothing as 
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1
    
    SMOOTHING_COEFICIENT = 0.6
    DATAPOINTS_COUNT = 300
    MIN_THRESHOLD = -0.001
    MAX_THRESHOLD = 0.15
    THRESHOLDS = [MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD) * i / (DATAPOINTS_COUNT - 1) \
                  for i in range(DATAPOINTS_COUNT)]
    
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
                
    for threshold in THRESHOLDS:
        print(threshold)
        sum_clip_length = 0.0
        sum_error_level = 0.0
        j = 0
        for precalc_data in precalc:
            stopped = False
            clip_start = 1 if threshold <= 1.0 else 0
            for i in range(clip_start, 30):
                if i == clip_start:
                    delta =  LSM_exp([precalc_dist[j][i]], 5)
                else:    
                    delta = LSM_exp(precalc_dist[j][clip_start:i], 5)                   
                if delta <= threshold:
                    sum_clip_length += (i + 1)
                    sum_error_level += precalc_data[i][0]
                    stopped = True
                    break
            if not stopped:
                sum_clip_length += 30
                sum_error_level += precalc_data[-1][0]
            j += 1
                
        x.append(sum_clip_length / len(precalc))
        y.append(sum_error_level / len(precalc))
    
    return x, y

def LSM_hprbl_stopper_epp(method, dataset):
    '''
    stopping method with LSM exponential smoothing as 
    '''
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1
    
    SMOOTHING_COEFICIENT = 0.6
    DATAPOINTS_COUNT = 300
    MIN_THRESHOLD = -0.001
    MAX_THRESHOLD = 0.15
    THRESHOLDS = [MIN_THRESHOLD + (MAX_THRESHOLD - MIN_THRESHOLD) * i / (DATAPOINTS_COUNT - 1) \
                  for i in range(DATAPOINTS_COUNT)]
    
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
                
    for threshold in THRESHOLDS:
        print(threshold)
        sum_clip_length = 0.0
        sum_error_level = 0.0
        j = 0
        for precalc_data in precalc:
            stopped = False
            clip_start = 1 if threshold <= 1.0 else 0
            for i in range(clip_start, 30):
                if i == clip_start:
                    delta =  LSM_hprbl([precalc_dist[j][i]], 5)
                else:    
                    delta = LSM_hprbl(precalc_dist[j][clip_start:i], 5)                   
                if delta <= threshold:
                    sum_clip_length += (i + 1)
                    sum_error_level += precalc_data[i][0]
                    stopped = True
                    break
            if not stopped:
                sum_clip_length += 30
                sum_error_level += precalc_data[-1][0]
            j += 1
                
        x.append(sum_clip_length / len(precalc))
        y.append(sum_error_level / len(precalc))
    
    return x, y

def roc_curve_stoppers(method, dataset):
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
              
    points_of_interest = []
    for i in range (len(precalc)):
        points_of_interest.append([i, 0, precalc_dist[i][1], precalc[i][0][0]])
        for j in range(1, len(precalc[i])-1):
            if precalc_dist[i][j+1] < points_of_interest[-1][2]:
                points_of_interest.append([i, j, precalc_dist[i][j+1], precalc[i][j][0]])
    
    
    points_of_interest = sorted(points_of_interest, key = lambda POI: POI[2], reverse = True)
    stopping_frames_id = [0 for i in range(len(precalc))]
    sum_of_frames = len(precalc)
    sum_of_errors = 0
    for i in range(len(precalc)):
        sum_of_errors += precalc[i][0][0]
    
    x.append(sum_of_frames/len(precalc))
    y.append(sum_of_errors/len(precalc))
    for i in range(len(points_of_interest)):
        sum_of_frames -= (stopping_frames_id[points_of_interest[i][0]]+1)
        sum_of_errors -= precalc[points_of_interest[i][0]][stopping_frames_id[points_of_interest[i][0]]][0]
        stopping_frames_id[points_of_interest[i][0]] = points_of_interest[i][1]
        sum_of_frames += (points_of_interest[i][1]+1)
        sum_of_errors += points_of_interest[i][3]
        x.append(sum_of_frames/len(precalc))
        y.append(sum_of_errors/len(precalc))   
    return x, y

def roc_curve_fixed_stoppers(method, dataset):
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
                
    points_of_interest = []
    for i in range (len(precalc)):
        points_of_interest.append([i, 0, 1, precalc[i][0][0]])
        for j in range(1, len(precalc[i])-1):
            if 1/(j+1) < points_of_interest[-1][2]:
                points_of_interest.append([i, j, 1/(j+1), precalc[i][j][0]])
    
    
    points_of_interest = sorted(points_of_interest, key = lambda POI: POI[2], reverse = True)
    stopping_frames_id = [0 for i in range(len(precalc))]
    sum_of_frames = len(precalc)
    sum_of_errors = 0
    for i in range(len(precalc)):
        sum_of_errors += precalc[i][0][0]
    
    x.append(sum_of_frames/len(precalc))
    y.append(sum_of_errors/len(precalc))
    for i in range(len(points_of_interest)):
        sum_of_frames -= (stopping_frames_id[points_of_interest[i][0]]+1)
        sum_of_errors -= precalc[points_of_interest[i][0]][stopping_frames_id[points_of_interest[i][0]]][0]
        stopping_frames_id[points_of_interest[i][0]] = points_of_interest[i][1]
        sum_of_frames += (points_of_interest[i][1]+1)
        sum_of_errors += points_of_interest[i][3]
        x.append(sum_of_frames/len(precalc))
        y.append(sum_of_errors/len(precalc))  
        
    return x, y

def roc_curve_SES_stoppers(method, dataset):
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    precalc_dist_dir = PRECALC_DISTANCE_DIRECTORIES[dataset][method]
    SMOOTHING_COEFICIENT = 0.9
    x = []
    y = []
    
    precalc = []
    precalc_dist = []
    precalc_dist_files = [os.path.join(precalc_dist_dir, x) for x in os.listdir(precalc_dist_dir)]
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
    for precalc_dist_file in precalc_dist_files:
        precalc_dist_data = None
        if precalc_dist_file.endswith('.json'):
            with open(precalc_dist_file) as js:
                precalc_dist_data = json.load(js)
                precalc_dist.append(precalc_dist_data)
              
    points_of_interest = []
    for i in range (len(precalc)):
        points_of_interest.append([i, 0, precalc_dist[i][0], precalc[i][0][0]])
        for j in range(1, len(precalc[i])):
            if SES(precalc_dist[i][0:j], SMOOTHING_COEFICIENT)[-1] < points_of_interest[-1][2]:
                points_of_interest.append([i, j, SES(precalc_dist[i][0:j], SMOOTHING_COEFICIENT)[-1], precalc[i][j][0]])
    
    
    points_of_interest = sorted(points_of_interest, key = lambda POI: POI[2], reverse = True)
    stopping_frames_id = [0 for i in range(len(precalc))]
    sum_of_frames = len(precalc)
    sum_of_errors = 0
    for i in range(len(precalc)):
        sum_of_errors += precalc[i][0][0]
    
    x.append(sum_of_frames/len(precalc))
    y.append(sum_of_errors/len(precalc))
    for i in range(len(points_of_interest)):
        sum_of_frames -= (stopping_frames_id[points_of_interest[i][0]]+1)
        sum_of_errors -= precalc[points_of_interest[i][0]][stopping_frames_id[points_of_interest[i][0]]][0]
        stopping_frames_id[points_of_interest[i][0]] = points_of_interest[i][1]
        sum_of_frames += (points_of_interest[i][1]+1)
        sum_of_errors += points_of_interest[i][3]
        x.append(sum_of_frames/len(precalc))
        y.append(sum_of_errors/len(precalc))      
    return x, y

def roc_curve_base_a_b(method, dataset):
    precalc_dir = PRECALC_DIRECTORIES[dataset][method]
    SMALL_DELTA = 0.1
    x = []
    y = []
    
    precalc = []
    precalc_files = [os.path.join(precalc_dir, x) for x in os.listdir(precalc_dir)]
    for precalc_file in precalc_files:
        precalc_data = None
        with open(precalc_file) as js:
            precalc_data = json.load(js)
        precalc.append(precalc_data)
        
    points_of_interest = []
    
    for i in range(len(precalc)):
        curr = [i, 1, (SMALL_DELTA + precalc[i][1][1]) / 3, precalc[i][1][0]]
        points_of_interest.append(curr)
        for j in range(2, len(precalc[i])):
            delta = (SMALL_DELTA + precalc[i][j][1]) / (j + 2)
            if delta < points_of_interest[-1][2]:
                points_of_interest.append([i, j, delta, precalc[i][j][0]])
        if points_of_interest[-1] == curr:
            points_of_interest.append(i, 29, (SMALL_DELTA + precalc[i][-1][1]) / 31, precalc[i][-1][0])
    
    points_of_interest = sorted(points_of_interest, key = lambda POI: POI[2], reverse = True)
    stopping_frames_id = [1 for i in range(len(precalc))]
    sum_of_frames = 2*len(precalc)
    sum_of_errors = 0
    for i in range(len(precalc)):
        sum_of_errors += precalc[i][1][0]
    
    x.append(sum_of_frames/len(precalc))
    y.append(sum_of_errors/len(precalc))
    
    for i in range(len(points_of_interest)):
        sum_of_frames -= stopping_frames_id[points_of_interest[i][0]] + 1
        sum_of_errors -= precalc[points_of_interest[i][0]][stopping_frames_id[points_of_interest[i][0]]][0]
        stopping_frames_id[points_of_interest[i][0]] = points_of_interest[i][1]
        sum_of_frames += (points_of_interest[i][1]) + 1
        sum_of_errors += points_of_interest[i][3]
        x.append(sum_of_frames/len(precalc))
        y.append(sum_of_errors/len(precalc))
        
    return x, y


plt.rcParams['figure.figsize'] = (14, 4)
plt.rcParams.update({'font.size': 12})

plt.clf()
plt.cla()

for i_dataset, dataset in enumerate(EPPS_DATASETS):
    plt.subplot(100 + 10 * len(EPPS_DATASETS) + i_dataset + 1)
    plt.title(('%s) ' % chr(ord('a') + i_dataset)) + DATASET_LABELS[dataset])

    plt.gca().xaxis.set_minor_locator(MultipleLocator(0.5))
    plt.gca().xaxis.set_major_locator(MultipleLocator(2))

    c_x, c_y = collect_counting_stopper_epp(dataset)
    plt.plot(c_x, c_y, label=r'Stopping at fixed stage', color='0.5', linestyle='--', linewidth = 1)

    PLOT_LINEWIDTH = { 'base': 2.0, 'summation': 2.0, 'treap': 2.5 }

    for method in METHODS:
        a, b = roc_curve_fixed_stoppers('summation', dataset)
        plt.plot(a, b, label = "fixed stage", c = 'r')         
        x, y = collect_modelling_stopper_epp(method, dataset)
        if method == 'summation':
            plt.scatter(x, y, \
                 label = PLOT_LABEL[method], \
                 color = PLOT_COLOR[method])
        else:
            plt.plot(x, y, \
                     label = PLOT_LABEL[method], \
                     color = PLOT_COLOR[method], \
                     linestyle = PLOT_LINESTYLE[method], \
                     linewidth = PLOT_LINEWIDTH[method])
        a, b = roc_curve_base_a_b(method, dataset)
        plt.plot(a, b, \
                 label = PLOT_LABEL_ROC[method], \
                 color = PLOT_COLOR_ROC[method], \
                 linestyle = PLOT_LINESTYLE[method], \
                 linewidth = PLOT_LINEWIDTH[method])
        plt.xlabel(r'Mean number of frames')
        plt.ylabel('Mean error level')
        plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.xlim([1, 20])
        plt.ylim(EPPS_YLIMITS[dataset])
        plt.grid()
        plt.legend()        
        plt.show()

    #a, b = double_dist_stopper_epp('summation', dataset)
    #plt.plot(a, b, label = "Double Distance Stopper", color = 'r')
    #c, d = LSM_exp_stopper_epp('summation', dataset)
    #plt.plot(c, d, label = "LSM exponential stopper", color = 'b')
    #e, f = roc_curve_stoppers('summation', dataset)
    #plt.plot(e, f, label = "roc curve test", color = 'b')
    a, b = roc_curve_fixed_stoppers('summation', dataset)
    plt.plot(a, b, label = "fixed stage", c = 'r')    
    #c, d = collect_dist_stopper_epp('summation', dataset)
    #plt.plot(c, d, label = 'Distance Stoppers', color = 'g')
    #c, d = roc_curve_SES_stoppers('summation', dataset)
    #plt.plot(c, d, label = "SES", c = 'g')    
    #e, f = exp_smth_stopper_epp('summation', dataset)
    #plt.plot(e, f, label = 'SEP Stoppers', color = 'm')
    #e, f = AR_stopper_epp('summation', dataset)
    #plt.plot(e, f, label = 'Autoregression', color = 'b')
    #h, i = SQR_stopper_epp('summation', dataset)
    #plt.plot(h, i, label = 'Square regression', color = 'm')
    #plt.plot(j, k, label = "Moving Average", color = "c")
    
    plt.xlim([1, 20])
    plt.ylim(EPPS_YLIMITS[dataset])
    plt.grid()
    plt.legend()
    plt.xlabel(r'Mean number of frames')
    plt.ylabel('Mean error level')
    plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.2f'))

plt.savefig('epps_%s.pdf' % '_'.join(EPPS_DATASETS), dpi=1200, bbox_inches='tight', pad_inches=0)
#plt.show()