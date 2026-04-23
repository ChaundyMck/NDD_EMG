import numpy as np
from scipy.signal import find_peaks, convolve
from scipy.stats import norm
import pandas as pd

def dominant_direction(delsys_acc):
    '''
    Determine the direction of the acceleration that changes the most
    '''

    Acc = (2 * (np.mean(delsys_acc[['X', 'Y', 'Z']], axis = 0) > 0).astype(int)) - 1 #identify sign of mean
    Acc = delsys_acc[['X', 'Y', 'Z']] * np.tile(Acc, (delsys_acc.shape[0], 1)) #multiply by those signs so that all means are positive

    mid_start = int(1/3 * len(Acc))
    mid_end = int(2/3 * len(Acc))
    stds = np.std(Acc.iloc[mid_start:mid_end], axis=0) #Find std of the middle third of each column
    dom_dir_idx = np.argmax(stds)
    dom_Acc = Acc.iloc[:,dom_dir_idx]

    return dom_Acc


def derivative(value, smoothing_window, freq):
    '''
    Determine derivative of value with a gaussian smoothening function applied
    '''
    ##Smoothening
    smooth = []
    s_value = []
    s_value['Time'] = value['Time']
    for i in range(1,value.shape[1]):
        tau = int(np.ceil(smoothing_window))
        t = np.arange(-(tau-1), tau)
        gauss_funct = norm.pdf(t, loc = 0, scale = tau/2)
        smooth[value.columns[i]] = convolve(value.iloc[i], gauss_funct, mode='same')
        pd.DataFrame(smooth)
        pd.concat([s_value, smooth], axis = 1, ignore_index= True)

    ##Take derivative
    raw_derivative = np.zeros(value.shape)
    smooth_derivative = np.zeros(s_value.shape)
    raw_derivative.iloc[:, 0] = value['Time']
    smooth_derivative.iloc[:, 0] = s_value['Time']
    raw_derivative.loc[1:-1, 'X'] = (value['X'].diff())*freq
    raw_derivative.loc[1:-1, 'Y'] = (value['Y'].diff())*freq
    raw_derivative.loc[1:-1, 'Z'] = (value['Z'].diff())*freq
    smooth_derivative.loc[1:-1, 'X'] = (s_value['X'].diff())*freq
    smooth_derivative.loc[1:-1, 'Y'] = (s_value['Y'].diff())*freq
    smooth_derivative.loc[1:-1, 'Z'] = (s_value['Z'].diff())*freq
    mag_raw_dt = np.sqrt(np.sum(raw_derivative**2, axis = 1))
    mag_smooth_dt = np.sqrt(np.sum(smooth_derivative**2, axis = 1))

    return raw_derivative, smooth_derivative, mag_raw_dt, mag_smooth_dt



def get_trial(Acc, Acc_freq, s_Jrk, trial_duration_estimate):
    '''
    From the direction of dominant change determine the start, stop, and touch points for each trial. 
    Need to choose an estimate for length of one trial and use the freq of the variable to determine how many samples need to be considered.

    Start
    '''

    dom_Acc = dominant_direction(s_Acc)
##Do I need to do this?? just take the csv file and run it through her matlab code??

    #Identify touch points
    mid_start = int(1/3 * len(dom_Acc))
    mid_end = int(2/3 * len(dom_Acc))
    peak_threshold = np.mean(-dom_Acc[mid_start:mid_end]) + np.std(-dom_Acc[mid_start:mid_end])
    min_dist = int(trial_duration_estimate * Acc_freq)
    touch_idx, height = find_peaks(dom_Acc, height=peak_threshold, distance = min_dist)
    

    idx = find_peaks(dom_Acc)
    temp_threshold = min_dist/5
    for i in range(len(touch_idx)):
        Acc_btwn_touch = dom_Acc[touch_idx[i]:touch_idx[i+1]]
        Jrk_btwn_touch = s_Jrk[touch_idx[i]:touch_idx[i+1]]


