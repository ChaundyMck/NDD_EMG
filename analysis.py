from scipy import signal
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def peak_analysis(start_times, stop_times, signal, save_path):
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
    extrema_diff = pd.DataFrame(columns=["amp_diff", "time_diff"])
    save_path = os.makedirs(save_path, exist_ok=True)
    
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

    extrema_diff["amp_diff"] = total_amp_diff
    extrema_diff["time_diff"] = total_time_diff

    plt.figure(figsize=(7, 6))
    sns.histplot(data=extrema_diff["amp_diff"].dropna(), kde=True, stat = "density", alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Amplitude Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Amplitude Differences')
    plt.savefig(save_path + '/Amplitude_Difference_Histogram.png', dpi=300, bbox_inches='tight')

    plt.figure(figsize=(7, 6))
    sns.histplot(data=extrema_diff["time_diff"].dropna(), kde=True, stat = "density", alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Time Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time Differences')
    plt.savefig(save_path + '/Time_Difference_Histogram.png', dpi=300, bbox_inches='tight')         

    plt.figure(figsize=(7, 7))
    plt.scatter(extrema_diff["time_diff"], extrema_diff["amp_diff"], alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Time Difference')
    plt.ylabel('Amplitude Difference')
    plt.title('Scatter Plot of Time vs Amplitude Differences')
    plt.savefig(save_path + '/Time_vs_Amplitude_Differences_Scatter.png', dpi=300, bbox_inches='tight')                  

    plt.figure(figsize=(7,6))
    sns.heatmap(extrema_diff, annot=True, cmap='coolwarm')
    plt.title('Heatmap of Amplitude and Time Differences')
    plt.savefig(save_path + '/Amplitude_and_Time_Differences_Heatmap.png', dpi=300, bbox_inches='tight')

    return extrema_diff

def onset_detection(EMG, threshold, min_duration=0.05, fs=1000):
    """
    Detect the onset of muscle activation in an EMG signal based on a threshold and minimum duration.

    Parameters:
    EMG (pd.DataFrame): The DataFrame containing the EMG signal with columns ['Time', 'EMG1', 'EMG2'].
    threshold (float): The threshold value for detecting onset.
    min_duration (float): The minimum duration (in seconds) that the signal must exceed the threshold to be considered an onset (default is 0.05).
    fs (int): The sampling frequency of the EMG signal (default is 1000 Hz).

    Returns:
    pd.DataFrame: A DataFrame containing the detected onset times for each EMG channel.
    """
    
    onset_times = {'EMG1': [], 'EMG2': []}
    
    for channel in ['EMG1', 'EMG2']:
        above_threshold = EMG[channel] > threshold
        onset_indices = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
        
        for idx in onset_indices:
            if idx + int(min_duration * fs) < len(EMG):
                if np.all(above_threshold[idx:idx + int(min_duration * fs)]):
                    onset_times[channel].append(EMG['Time'].iloc[idx])
    
    return pd.DataFrame(onset_times)

