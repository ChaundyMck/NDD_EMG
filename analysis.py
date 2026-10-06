from scipy import signal
from scipy.signal import find_peaks

def peak_analysis(start_times, stop_times, signal):
    """
    Perform peak analysis on the cleaned EMG signal to identify the maximum value and its corresponding time.

    Parameters:
    start_times (list): List of start times for each trial.
    stop_times (list): List of stop times for each trial.
    signal (pd.DataFrame): The DataFrame containing the signal with columns ['Time', signal_column].

    Returns:
    tuple: DataFrames containing the maxima and minima for each EMG channel.
    """
    amplitude = signal.columns[1]  # Assuming the second column contains the signal data
    total_amp_diff = []
    total_time_diff = []
    for start, stop in zip(start_times, stop_times):
        trial_data = signal[(signal['Time'] >= start) & (signal['Time'] <= stop)]

        # Perform peak analysis on within trial data
        maxdices, _ = find_peaks(trial_data[amplitude], distance = 2)
        mindices, _ = find_peaks(-trial_data[amplitude], distance = 2)

        indicies = mindices.tolist() + maxdices.tolist()
        indicies.sort()

        trial_amp_diff = trial_data[indicies][amplitude].diff().abs()
        trial_time_diff = trial_data[indicies]['Time'].diff()

        total_amp_diff.append(trial_amp_diff)
        total_time_diff.append(trial_time_diff)



    return total_amp_diff, total_time_diff