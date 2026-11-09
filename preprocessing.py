import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt, iirnotch
from scipy.stats import norm

def butter_bandpass_filter(signal, lowcut, highcut, fs, order = 4):
    """
    Apply a Butterworth bandpass filter followed by a 60 Hz notch filter.

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
    notch_b, notch_a = iirnotch(60, Q=30, fs=fs)
    filtered_signal = filtfilt(notch_b, notch_a, filtered_signal)

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
    t = np.arange(-(tau-1),tau)
    gauss = norm.pdf(t, 0, tau/2)
    smooth = np.convolve(raw_accel, gauss, mode = 'same')
    
    return smooth


## IMU preprocessing
# Follows the workflow in Fibion, "Preprocess and align IMU coordinates" (Series 2, Part 3, Article 2):
#   1. resample to a uniform timeline, interpolating only short gaps (<0.1 s)        (Sec. 3.4)
#   2. remove gyroscope bias using the mean/median output during static periods     (Sec. 3.2)
#   3. 4th-order zero-phase Butterworth low-pass, 5 Hz cutoff                       (Sec. 3.1)
#   4. estimate gravity with a gyro/accel complementary filter (beta ~ 0.01) and
#      subtract it to get linear acceleration; keep both total and linear           (Sec. 3.2, 4.2, 4.4)
# Z-scoring (Sec. 5) and anatomical axis alignment (Sec. 6) are not applied: the analysis uses the
# orientation-invariant magnitude (Sec. 5.5) and Fano factor is undefined for zero-mean signals.

def _contiguous_runs(valid):
    """Yield (start, stop) slices of consecutive True values in a boolean array."""
    edges = np.diff(np.concatenate(([0], valid.astype(int), [0])))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def imu_resample(imu, fs, t_start=None, t_stop=None, max_gap=0.1):
    """
    Interpolate IMU data onto a uniform 1/fs timeline. Gaps longer than max_gap seconds are left as NaN.

    imu: pd.DataFrame with columns ['Time', 'X', 'Y', 'Z']
    """
    t = imu['Time'].to_numpy(dtype=float)
    t_start = t[0] if t_start is None else t_start
    t_stop = t[-1] if t_stop is None else t_stop
    new_t = np.arange(t_start, t_stop, 1 / fs)

    out = pd.DataFrame({'Time': new_t})
    for c in ['X', 'Y', 'Z']:
        out[c] = np.interp(new_t, t, imu[c].to_numpy(dtype=float), left=np.nan, right=np.nan)

    # Mask samples that fall inside a gap in the original timestamps longer than max_gap
    idx = np.searchsorted(t, new_t)
    idx = np.clip(idx, 1, len(t) - 1)
    in_long_gap = (t[idx] - t[idx - 1]) > max_gap
    out.loc[in_long_gap, ['X', 'Y', 'Z']] = np.nan
    return out


def imu_lowpass(x, fs, cutoff=5, order=4):
    """Zero-phase Butterworth low-pass (filtfilt). NaN gaps are skipped and each valid run is filtered separately."""
    x = np.asarray(x, dtype=float)
    b, a = butter(order, cutoff, btype='lowpass', fs=fs)
    padlen = 3 * max(len(a), len(b))
    y = np.full_like(x, np.nan)
    for s, e in _contiguous_runs(~np.isnan(x)):
        if e - s > padlen:
            y[s:e] = filtfilt(b, a, x[s:e])
    return y


def gyro_bias(gyro, static_intervals):
    """
    Estimate gyroscope bias as the median output (per axis) over known static periods.

    gyro: pd.DataFrame ['Time', 'X', 'Y', 'Z']
    static_intervals: list of (t_start, t_stop) tuples during which the sensor is still
    """
    mask = np.zeros(len(gyro), dtype=bool)
    for t0, t1 in static_intervals:
        mask |= (gyro['Time'] >= t0).to_numpy() & (gyro['Time'] <= t1).to_numpy()
    if not mask.any():
        raise ValueError("No gyroscope samples fall inside the static intervals.")
    return gyro.loc[mask, ['X', 'Y', 'Z']].median().to_numpy()


def gravity_complementary(acc, gyro_rad, fs, beta=0.01):
    """
    Track the gravity vector in the sensor frame with a complementary filter.

    Each step rotates the previous gravity estimate by the gyroscope rotation (a world-fixed vector
    rotates by -omega*dt in the sensor frame), then blends it toward the accelerometer reading by beta.

    acc: (N, 3) accelerometer, any units (the gravity estimate is returned in the same units)
    gyro_rad: (N, 3) angular velocity in rad/s, same timeline as acc
    """
    dt = 1 / fs
    g = np.full_like(acc, np.nan)
    prev = None
    for k in range(len(acc)):
        a, w = acc[k], gyro_rad[k]
        if np.isnan(a).any() or np.isnan(w).any():
            prev = None  # restart after a gap
            continue
        if prev is None:
            g[k] = prev = a
            continue
        theta = np.linalg.norm(w) * dt
        if theta > 0:
            u = -w / np.linalg.norm(w)  # rotate by -omega*dt (Rodrigues)
            prev = (prev * np.cos(theta) + np.cross(u, prev) * np.sin(theta)
                    + u * np.dot(u, prev) * (1 - np.cos(theta)))
        prev = (1 - beta) * prev + beta * a
        g[k] = prev
    return g


def preprocess_IMU(raw_acc, raw_gyro, static_intervals, fs=None, cutoff=5, order=4, beta=0.01, max_gap=0.1):
    """
    Full IMU preprocessing pipeline (see module notes above).

    raw_acc: pd.DataFrame ['Time', 'X', 'Y', 'Z'] in G
    raw_gyro: pd.DataFrame ['Time', 'X', 'Y', 'Z'] in deg/s
    static_intervals: list of (t_start, t_stop) where the sensor is still, used for gyro bias removal
    fs: target uniform sampling rate; defaults to the accelerometer's native rate

    Returns
    acc: pd.DataFrame ['Time', 'X', 'Y', 'Z', 'Mag', 'X_lin', 'Y_lin', 'Z_lin', 'Mag_lin']
        low-passed total acceleration and gravity-removed linear acceleration (G)
    gyro: pd.DataFrame ['Time', 'X', 'Y', 'Z'] bias-corrected, low-passed angular velocity (deg/s)
    """
    if fs is None:
        fs = 1 / np.median(np.diff(raw_acc['Time'].to_numpy(dtype=float)))
    t0 = max(raw_acc['Time'].iloc[0], raw_gyro['Time'].iloc[0])
    t1 = min(raw_acc['Time'].iloc[-1], raw_gyro['Time'].iloc[-1])
    acc = imu_resample(raw_acc, fs, t0, t1, max_gap)
    gyro = imu_resample(raw_gyro, fs, t0, t1, max_gap)

    gyro[['X', 'Y', 'Z']] = gyro[['X', 'Y', 'Z']].to_numpy() - gyro_bias(gyro, static_intervals)

    for c in ['X', 'Y', 'Z']:
        acc[c] = imu_lowpass(acc[c], fs, cutoff, order)
        gyro[c] = imu_lowpass(gyro[c], fs, cutoff, order)

    a = acc[['X', 'Y', 'Z']].to_numpy()
    g = gravity_complementary(a, np.deg2rad(gyro[['X', 'Y', 'Z']].to_numpy()), fs, beta)
    lin = a - g
    acc['Mag'] = np.linalg.norm(a, axis=1)
    acc[['X_lin', 'Y_lin', 'Z_lin']] = lin
    acc['Mag_lin'] = np.linalg.norm(lin, axis=1)
    return acc, gyro
