import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt

def butter_bandpass_filter(signal, lowcut, highcut, fs, order = 4):
    """
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
        filtered EMG signal
    """
    nyq = 0.5*fs
    low = lowcut/nyq
    high = highcut/nyq
    b,a = butter(order, [low,high], btype = 'bandpass')
    filtered_signal = filtfilt(b,a,signal)

    return filtered_signal
    

def resamp(signal):
    """
    Resample the data to 1000 Hz, nothing significant above 1000 Hz

    signal: array-like 
        input EMG signal

    Returns
    EMG_resamp: ndarray
        EMG signal resampled to 1000 Hz
    """
    
    EMG_resamp = signal.copy()
    EMG_resamp['Time'] = pd.to_datetime(EMG_resamp['Time'], unit='s')
    EMG_resamp = EMG_resamp.set_index('Time')
    EMG_resamp = EMG_resamp.resample('1ms').mean().interpolate(method='linear')
    EMG_resamp = EMG_resamp.reset_index()

    return EMG_resamp



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



def sync(analog_data, target_data, a_event_cutoff=0.01):
    """
    Sync the analog and target data based on event markers. 
    The function identifies the first event in each dataset that exceeds the specified cutoff values, calculates the time difference between these events, and then adjusts the timestamps of the target data to align with the analog data.

    Parameters:
    analog_data (pd.DataFrame): The DataFrame containing the analog data from delsys ['Time','Analog'].
    target_data (pd.DataFrame): The DataFrame containing the target data from delsys ['hitTime'].
    a_event_cutoff (float): The cutoff value for identifying events in the analog data.

    Returns:
    pd.DataFrame: The adjusted target_data DataFrame with synchronized timestamps.
    """
    
    # Identify the first event in the analog data that exceeds the cutoff, filtering does not change the time of the events.
    peaks, _ = find_peaks(analog_data['Analog'], height=a_event_cutoff, distance = 100)
    a_event_time = analog_data.loc[peaks]['Time'] 
    t_event_time = target_data['hitTime']/1000
    # Calculate the time difference between the two events, this gives the time difference for all delsys times to local computer time
    time_diff = a_event_time - t_event_time
    

    return time_diff


def peak_normalization(EMG_signal):
    """
    Normalize the EMG signal based on the maximum value within the trial.

    Parameters:
    signal (pd.DataFrame): The DataFrame containing the EMG signal with columns ['Time', 'EMG1', 'EMG2'].

    Returns:
    pd.DataFrame: The normalized EMG signal.
    """
    
    normalized_signal = EMG_signal.copy()
    normalized_signal['EMG1'] = normalized_signal['EMG1'] / normalized_signal['EMG1'].max()
    normalized_signal['EMG2'] = normalized_signal['EMG2'] / normalized_signal['EMG2'].max()

    return normalized_signal