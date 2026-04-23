import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import re

def delsys_csv_breakdown(delsys_csv):
    """
    The Delsys csv needs read in in different parts and reworked to 
    make columns easily accessible. 
    """
    time = pd.read_csv(delsys_csv, nrows = 2)

    config = pd.read_csv(delsys_csv, skiprows = 5, nrows = 1)
    config.replace(" ", float("NaN"), inplace = True)
    config.dropna(axis = 1, inplace = True) # Just retain the columns that have units
    config = config
    cf_new_cols = []
    cf_col_counts = {}
    for cf_col in config.columns:
        if cf_col in cf_col_counts:
            cf_col_counts[col] += 1
            cf_new_cols.append(f"{cf_col} {cf_col_counts[cf_col]}")
        else:
            cf_col_counts[cf_col] = 1
            cf_new_cols.append(cf_col)
    config.columns = cf_new_cols

    data = pd.read_csv(delsys_csv,header=5,skiprows=[6,7])
    data.columns = data.columns.str.lstrip()
    data.columns = data.columns.str.replace(r"\.\d+$", "", regex=True)

    #make duplicate column names unique
    new_cols = []
    col_counts = {}
    for col in data.columns:
        if col in col_counts:
            col_counts[col] += 1
            new_cols.append(f"{col} {col_counts[col]}")
        else:
            col_counts[col] = 1
            new_cols.append(col)
    data.columns = new_cols

    for col in data.columns:
        data[col] = data[col].astype(str).str.replace(' ', '', regex=False)
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna(how="all")

    return (time, config, data)

def butter_bandpass_filter(signal, lowcut, highcut, fs, order = 4):
    """""
    Apply a 4th order butterworth bandpass filter

    signal: array-like
        input EMG signal
    lowcut: float
        low cutoff frequency Hz
    highcut: float
        high cutoff frequency Hz
    fs: float
        sampling frequency Hz
    order: int
        filter order (default = 4)

    Returns
    filtered_signal : ndarray
    """
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low,high], btype = 'bandpass')
    filtered_signal = filtfilt(b,a,signal)

    return filtered_signal
    

def rms_smoothing(signal, win_size):
    """
    Compute RMS smoothing using a sliding window

    signal : array-like
        input EMG signal
    win_size : int
        window size in samples

    Returns
    rms : ndarray
        rms smoothed signal
    """

    sq = signal**2
    win = np.ones(win_size)/win_size

    m_s = np.convolve(sq, win, mode='same')
    rms = np.sqrt(m_s)

    return rms

def peak_normalization(signal):
    """
    Normalize each respective muscle by the peak EMG amplitude
    
    signal: array-like
        pre-filtered and rectified EMG signal

    Returns
    norm_signal
    """

    peaks = np.max(signal)

