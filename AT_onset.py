"""
AT_onset.py - Adaptive Threshold (AT) EMG onset detection

Method from Carvalho et al. (2023), "Review of electromyography onset detection methods
for real-time control of robotic exoskeletons", J NeuroEng Rehabil 20:141. Of the methods
compared (single, double and adaptive threshold), AT had the best timing precision.

AT as described in the paper:
  * applied to the EMG envelope (the paper's advice for noisy surface EMG; raw
    rectified sEMG crosses the threshold too often at rest)
  * the recording is split into N equal windows, N = number of movement cycles
  * each window gets its own threshold from the SD of its baseline (1-3 SD in the paper)

Runs on rms_EMG_1.csv / rms_EMG_2.csv (30 ms RMS envelope) in each Final folder and saves onsets.csv there.
"""
import os
import numpy as np
import pandas as pd

final_data_path = '/Users/clmckeev/Library/CloudStorage/OneDrive-SharedLibraries-IndianaUniversity/[Sec-E] BL-Phys-NDD_EMG - Documents/NDD_EMG/Data/Final'
K = 3         # threshold = baseline mean + K*SD (paper used 1-3, chosen per subject by visual check)
BASE_S = 1.0  # quietest stretch of each window used as its baseline (s)
REST_S = 0.1  # EMG must stay below threshold this long before a crossing counts as an onset (s)


def at_onsets(time, emg, n_windows, k=K):
    """Return onset times (s) of the EMG using a per-window adaptive threshold."""
    time = np.asarray(time)
    fs = 1 / np.median(np.diff(time))
    base_n, rest_n = int(BASE_S*fs), int(REST_S*fs)
    x = np.asarray(emg)

    onsets = []
    for w in np.array_split(np.arange(len(x)), n_windows):
        seg = x[w]
        b = np.argmin(np.convolve(seg, np.ones(base_n)/base_n, mode='valid'))
        base = seg[b:b+base_n]
        above = seg > base.mean() + k*base.std()

        # onset = sample above threshold with no samples above it in the previous REST_S
        c = np.concatenate(([0], np.cumsum(above)))
        idx = np.flatnonzero(above[rest_n:] & (c[rest_n:-1] - c[:-rest_n-1] == 0)) + rest_n
        onsets.extend(time[w[idx]])

    return np.array(onsets)


if __name__ == '__main__':
    for subject_id in sorted(os.listdir(final_data_path)):
        subject_dir = os.path.join(final_data_path, subject_id)
        if not os.path.exists(os.path.join(subject_dir, 'EMG_1.csv')):
            continue
        # one window per movement cycle (trial)
        n_windows = pd.read_csv(os.path.join(subject_dir, 'accel_trial_data.csv'), usecols=['Trial'])['Trial'].nunique()

        out = []
        for sensor in ['1', '2']:
            emg = pd.read_csv(os.path.join(subject_dir, f'rms_EMG_{sensor}.csv'))
            for ch, col in [('1', 'EMG1'), ('2', 'EMG2')]:
                on = at_onsets(emg['Time'], emg[col], n_windows)
                out.append(pd.DataFrame({'Channel': f'EMG_{sensor}_{ch}', 'Onset': on}))

        pd.concat(out).to_csv(os.path.join(subject_dir, 'onsets.csv'), index=False)
        print(subject_id, 'done')
