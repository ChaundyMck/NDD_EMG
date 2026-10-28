from scipy import signal
from scipy.signal import find_peaks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def peak_analysis(start_times, stop_times, signal, save_path, plot_show=True):
    """
    Perform peak analysis on the signal to identify the maximum value and its corresponding time.

    Parameters:
    start_times (list): List of start times for each trial.
    stop_times (list): List of stop times for each trial.
    signal (pd.DataFrame): The DataFrame containing the signal with columns ['Time', signal_column].
    save_path (str): The path where the analysis results will be saved.
    plot_show (bool): Whether to display the plots (default is True).

    Returns:
    tuple: DataFrames containing the maxima and minima for each EMG channel.
    """
    amplitude = signal.columns[1]  # Assuming the second column contains the signal data
    total_amp_diff = []
    total_time_diff = []
    extrema_diff = pd.DataFrame(columns=["amp_diff", "time_diff"])
    os.makedirs(save_path, exist_ok=True)
    
    for start, stop in zip(start_times, stop_times):
        trial_data = signal[(signal['Time'] >= start) & (signal['Time'] <= stop)]

        # Perform peak analysis on within trial data
        maxdices, _ = find_peaks(trial_data[amplitude], distance = 2)
        mindices, _ = find_peaks(-trial_data[amplitude], distance = 2)

        indicies = mindices.tolist() + maxdices.tolist()
        indicies.sort()

        trial_amp_diff = trial_data.iloc[indicies][amplitude].diff()
        trial_time_diff = trial_data.iloc[indicies]['Time'].diff()

        total_amp_diff.append(trial_amp_diff)
        total_time_diff.append(trial_time_diff)

    extrema_diff["amp_diff"] = pd.concat(total_amp_diff)
    extrema_diff["time_diff"] = pd.concat(total_time_diff)


    plt.figure(figsize=(7, 6))
    sns.histplot(data=extrema_diff["amp_diff"].dropna(), kde=True, stat = "density", alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel('Amplitude Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Amplitude Differences')
    plt.savefig(save_path + '_Amplitude_Difference_Histogram.png', dpi=300, bbox_inches='tight')
    if plot_show:
        plt.show()
    plt.close()
  

    plt.figure(figsize=(7, 6))
    sns.histplot(data=extrema_diff["time_diff"].dropna(), kde=True, stat = "density", alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Time Difference')
    plt.ylabel('Frequency')
    plt.title('Histogram of Time Differences')
    plt.savefig(save_path + '_Time_Difference_Histogram.png', dpi=300, bbox_inches='tight')         
    if plot_show:
        plt.show()
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.scatter(extrema_diff["time_diff"], extrema_diff["amp_diff"], alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Time Difference')
    plt.ylabel('Amplitude Difference')
    plt.title('Scatter Plot of Time vs Amplitude Differences')
    plt.savefig(save_path + '_Time_vs_Amplitude_Differences_Scatter.png', dpi=300, bbox_inches='tight')
    if plot_show:
        plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.histplot(data=extrema_diff, x = 'time_diff', y = 'amp_diff', bins=50, stat='count', cbar=True, cbar_kws={'label': 'Co-occurrence count'}, cmap='magma')
    plt.xlabel('Time Difference (s)')
    plt.ylabel('Amplitude Difference')
    plt.title('Co-occurrence Frequency of Amplitude and Time Differences')
    plt.savefig(save_path + '_Amplitude_and_Time_Differences_Heatmap.png', dpi=300,  bbox_inches='tight')
    if plot_show:
        plt.show()
    plt.close()

    return extrema_diff


def emg_sanity_check(raw_signal, final_signal, start_times, stop_times, hit_times, save_path):
    """Plot raw and processed EMG1 over the same session interval and time axis."""

    session_start = min(start_times)
    session_stop = max(stop_times)
    session_midpoint = (session_start + session_stop) / 2
    plot_start = max(session_start, session_midpoint - 20)
    plot_stop = min(session_stop, session_midpoint + 20)
    raw_session = raw_signal[(raw_signal['Time'] >= plot_start) & (raw_signal['Time'] <= plot_stop)]
    final_session = final_signal[(final_signal['Time'] >= plot_start) & (final_signal['Time'] <= plot_stop)]

    os.makedirs(save_path, exist_ok=True)
    fig, (raw_ax_1, raw_ax_2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), sharex=True)
   
    for raw_ax in (raw_ax_1, raw_ax_2):
        j = 1
        final_ax = raw_ax.twinx()
        raw_line, = raw_ax.plot(raw_session['Time'], raw_session.iloc[:,j], color = 'blue', alpha = 0.5, label = 'Raw EMG')
        final_line, = final_ax.plot(final_session['Time'], final_session.iloc[:,j], color='orange', linewidth=0.7, label='Final EMG')

        raw_limit = np.max(np.abs(raw_session.iloc[:,j]))
        final_limit = np.max(np.abs(final_session.iloc[:,j]))
        raw_ax.set_ylim(-raw_limit, raw_limit)
        final_ax.set_ylim(-final_limit, final_limit)

        visible_starts = [start for start in start_times if plot_start <= start <= plot_stop]
        for i, start in enumerate(visible_starts):
            raw_ax.axvline(start, color='green', linestyle='--', label='Trial start' if i == 0 else None)
        visible_stops = [stop for stop in stop_times if plot_start <= stop <= plot_stop]
        for i, stop in enumerate(visible_stops):
            raw_ax.axvline(stop, color='red', linestyle='--',label='Trial stop' if i == 0 else None)
        visible_hits = [hit for hit in hit_times if plot_start <= hit <= plot_stop]
        for i, hit in enumerate(visible_hits):
            raw_ax.axvline(hit, color='yellow', label='Target hit' if i == 0 else None)

        raw_ax.set_xlim(plot_start, plot_stop)
        raw_ax.set_xlabel('Time (s)')
        raw_ax.set_ylabel('Raw EMG (mV)', color='blue')
        final_ax.set_ylabel('Final EMG', color='orange')
        raw_ax.set_title( f"{raw_session.columns[j]} Raw vs Final — Middle 40 Seconds of Session")
        raw_ax.grid(True, alpha=0.2)
        handles = [raw_line, final_line]
        labels = [line.get_label() for line in handles]
        event_handles, event_labels = raw_ax.get_legend_handles_labels()
        event_items = [
            (handle, label)
            for handle, label in zip(event_handles, event_labels)
            if label in {'Trial start', 'Trial stop', 'Target hit'}
        ]
        raw_ax.legend(
            handles + [handle for handle, _ in event_items],
            labels + [label for _, label in event_items],
            loc='upper right',
        )
        j = j+1

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, 'EMG_1_sanity_check.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)

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

