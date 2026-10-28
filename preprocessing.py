import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.stats import norm

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
    

def resamp(signal, fs_new=1000):
    """
    Resample the data to 1000 Hz

    signal: array-like 
        input signal
    fs_new: int
        new sampling frequency (default = 1000 Hz)

    Returns
    rs_signal: ndarray
        signal resampled to 1000 Hz
    """
    
    
    newtime = np.arange(signal['Time'].iloc[0], signal['Time'].iloc[-1], 1/fs_new)

    rs_signal = pd.DataFrame(columns=signal.columns)
    rs_signal['Time'] = newtime
    for col in signal.columns[1:]:
        rs_signal[col] = np.interp(newtime, signal['Time'], signal[col])

    return rs_signal



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
    time_diff (float): The time difference between the synchronized events.
    """
    
    # Identify the first event in the analog data that exceeds the cutoff, filtering does not change the time of the events.
    analog = analog_data['Analog'].to_numpy(dtype=float)
    peaks, _ = find_peaks(np.nan_to_num(analog), height=a_event_cutoff, distance = 100)
    if len(peaks) == 0:
        raise ValueError("Incorrect analog data: no valid peaks found above the cutoff value; skipping trial.")
    a_event_time = analog_data['Time'].iloc[peaks].dropna()
    t_event_time = (target_data['hitTime']/1000).dropna()  # Convert milliseconds to seconds and drop NaN values
    if a_event_time.empty or t_event_time.empty:
        raise ValueError("No valid event times found in the analog or target data.")
    
    # Calculate the time difference between the two events, this gives the time difference for all delsys times to local computer time
    time_diff = float(a_event_time.iloc[-1] - t_event_time.iloc[-1])
    

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

def IMU_smooth(raw_accel, win):
    """
    Gaussian filter applied to IMU data
    """

    tau = int(np.ceil(win/2))
    t = np.arrange(-(tau-1),tau)
    gauss = norm.pdf(t, 0, tau/2)
    smooth = np.convolve(raw_accel, gauss, mode = 'same')
    
    return smooth
