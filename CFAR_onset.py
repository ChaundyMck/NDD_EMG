"""
CFAR_onset.py - EMG onset detection with a CFAR threshold on TKEO (Kreipe et al. 2024,
Bioengineering 11:119), tuned on a few onsets and a rest period you select by eye.

    python CFAR_onset.py 014 EMG_1_1

or from main.py, after EMG_1 and EMG_2 are normalized:

    from CFAR_onset import cfar_onsets
    onsets_11 = cfar_onsets(EMG_1, 'EMG1', end_subject_dir, 'EMG_1_1', start_times, stop_times, subject_id)

    left click       select an onset
    right click      remove the nearest onset
    r                the next two clicks set the rest period
    left/right keys  move through the recording
    up/down keys     zoom the amplitude
    Enter            optimize and draw the detected onsets and offsets (orange)
    close the window to return the result

Trial start (solid) and stop (dotted) times are drawn in purple when they are passed in.

If the orange lines are wrong, select onsets where they failed and press Enter again. Files are
saved in the subject's Final folder: marks_<channel>.csv, rest_<channel>.csv,
cfar_params_<channel>.json and onsets_<channel>.csv.

Detection, on the filtered EMG from main.py:
  feature    TKEO, x[n]^2 - x[n-1]*x[n+1], averaged over 50 ms windows every 5 ms
  threshold  T[i] = max(K * median(reference window), FLOOR * median feature in the rest period)
             The reference window is R_s long and ends GUARD_S before i.
  onset      first i where more than C of the D feature values from i on exceed T[i]

The paper doesn't publish its K, R or C values and tuned them on its own data, so they are
searched here. D, GUARD_S and the window lengths are fixed. Three things are added because the
data is analyzed offline: T is held constant during a burst, a burst ends after GAP_S below T,
and each onset is moved to the first sustained crossing at 1 ms resolution.
"""
import itertools
import json
import os
import sys
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

final_data_path = '/Users/clmckeev/Desktop/Raw/Final'

WIN_S = 0.05        # feature window (s)
STEP_S = 0.005      # feature step (s)
D = 10              # detection window (feature values)
GUARD_S = 0.1       # gap between the reference window and the sample being tested (s)
GAP_S = 0.2         # a burst ends after this long below its threshold (s)
MIN_BURST_S = 0.1   # shorter bursts are dropped (s)
TOL_S = 0.1         # a detection this close to a selected onset counts as finding it (s)
VIEW_S = 6          # seconds shown at once

GRID = dict(K=[2, 3, 4, 6, 8, 12], C=[5, 7, 9], R_s=[0.5, 1.0, 2.0], FLOOR=[1.5, 2, 3, 5])


def tkeo(x):
    """Teager-Kaiser energy of each sample, with negative values set to 0."""
    tk = np.zeros_like(x, dtype=float)
    tk[1:-1] = x[1:-1]**2 - x[:-2]*x[2:]
    tk[0], tk[-1] = tk[1], tk[-2]
    return np.maximum(tk, 0)


def feature(tk, fs):
    """Mean TKEO over sliding windows. Returns (values, first sample of each window)."""
    win, step = int(round(WIN_S*fs)), int(round(STEP_S*fs))
    starts = np.arange(0, len(tk) - win + 1, step)
    c = np.concatenate(([0.0], np.cumsum(tk)))
    return (c[starts + win] - c[starts])/win, starts


def reference_median(feat, fs, R_s):
    """Median of the reference window before each feature value."""
    step = STEP_S
    med = pd.Series(feat).rolling(int(round(R_s/step))).median().shift(int(round(GUARD_S/step)))
    return med.bfill().to_numpy()  # start of the recording uses the first full reference window


def find_bursts(feat, thr, C):
    """Return (start, stop, threshold) of each burst, in feature indices."""
    n = len(feat) - D + 1
    counts = (sliding_window_view(feat, D) > thr[:n, None]).sum(axis=1)
    candidates = np.flatnonzero((counts > C) & (feat[:n] > thr[:n]))
    gap, min_burst = int(round(GAP_S/STEP_S)), int(round(MIN_BURST_S/STEP_S))
    values = feat.tolist()

    bursts = []
    pos = 0
    while True:
        k = np.searchsorted(candidates, pos)
        if k == len(candidates):
            return bursts
        start = int(candidates[k])
        T = thr[start]
        j, below = start, 0
        while j < len(values) and below < gap:
            below = below + 1 if values[j] <= T else 0
            j += 1
        stop = j - below
        if stop - start >= min_burst:
            bursts.append((start, stop, T))
        pos = j


def refine(tk, fs, sample, T):
    """Move an onset to the first crossing of T by the 10 ms TKEO that lasts for most of the next 25 ms."""
    win, smooth, hold = int(round(WIN_S*fs)), int(round(0.01*fs)), int(round(0.025*fs))
    a, b = sample, min(sample + win + hold, len(tk))
    if b - a <= hold:
        return sample
    above = np.convolve(tk[a:b], np.ones(smooth)/smooth, mode='same') > T
    held = np.convolve(above, np.ones(hold)/hold, mode='valid') > 0.5
    hit = np.flatnonzero(above[:len(held)] & held)
    return a + hit[0] if hit.size else sample


def bursts_to_times(bursts, starts, tk, time, fs):
    """Return (onset times, offset times) in the recording's time base."""
    win = int(round(WIN_S*fs))
    on = [time[refine(tk, fs, starts[b[0]], b[2])] for b in bursts]
    off = [time[min(starts[min(b[1], len(starts) - 1)] + win//2, len(time) - 1)] for b in bursts]
    return np.array(on), np.array(off)


def rest_level(feat, starts, time, fs, rest):
    """Median feature value inside the rest period (start, stop)."""
    centre = time[np.minimum(starts + int(round(WIN_S*fs))//2, len(time) - 1)]
    inside = (centre >= rest[0]) & (centre <= rest[1])
    if not inside.any():
        raise ValueError('rest period is too short')
    return float(np.median(feat[inside]))


def detect(time, x, settings):
    """Detect onsets with settings from optimize(). Returns (onset times, offset times)."""
    fs = 1/np.median(np.diff(time))
    tk = tkeo(x)
    feat, starts = feature(tk, fs)
    thr = np.maximum(settings['K']*reference_median(feat, fs, settings['R_s']), settings['FLOOR']*settings['rest_level'])
    return bursts_to_times(find_bursts(feat, thr, settings['C']), starts, tk, time, fs)


def optimize(time, x, marks, rest, grid=GRID):
    """Try every setting against the selected onsets and rest period. Returns a table, best first.

    Ranked by selected onsets found minus bursts inside the rest period, then by timing error
    against the selections, then by the fewest detections.
    """
    marks = np.sort(np.asarray(marks, float))
    fs = 1/np.median(np.diff(time))
    tk = tkeo(x)
    feat, starts = feature(tk, fs)
    level = rest_level(feat, starts, time, fs, rest)
    medians = {R_s: reference_median(feat, fs, R_s) for R_s in grid['R_s']}

    rows = []
    for K, C, R_s, FLOOR in itertools.product(grid['K'], grid['C'], grid['R_s'], grid['FLOOR']):
        thr = np.maximum(K*medians[R_s], FLOOR*level)
        on, off = bursts_to_times(find_bursts(feat, thr, C), starts, tk, time, fs)
        errors = []
        used = np.zeros(len(on), bool)
        for m in marks:
            dist = np.abs(on - m)
            dist[used] = np.inf
            j = np.argmin(dist) if len(on) else -1
            if j >= 0 and dist[j] <= TOL_S:
                used[j] = True
                errors.append(on[j] - m)
        in_rest = int(np.sum((on <= rest[1]) & (off >= rest[0])))
        rows.append(dict(K=K, C=C, R_s=R_s, FLOOR=FLOOR, rest_level=level, found=len(errors), in_rest=in_rest,
                         score=len(errors) - in_rest, detected=len(on),
                         median_abs_err_ms=1000*np.median(np.abs(errors)) if errors else np.inf))
    return pd.DataFrame(rows).sort_values(['score', 'median_abs_err_ms', 'detected'],
                                          ascending=[False, True, True]).reset_index(drop=True)


def trial_of(times, start_times, stop_times):
    """Trial number (from 1) containing each time, or NaN if it is outside every trial."""
    trial = np.full(len(times), np.nan)
    for n, (start, stop) in enumerate(zip(start_times, stop_times), start=1):
        trial[(np.asarray(times) >= start) & (np.asarray(times) <= stop)] = n
    return trial


def selection_window(time, x, save_path, channel, start_times=None, stop_times=None, title=''):
    """Build the selection window. Returns (figure, state); state['result'] holds the latest detection."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    for key in ['keymap.back', 'keymap.forward', 'keymap.home']:
        plt.rcParams[key] = []

    time, x = np.asarray(time, float), np.asarray(x, float)
    trials = list(zip(start_times, stop_times)) if start_times is not None and stop_times is not None else []
    amplitude = np.sqrt(np.convolve(tkeo(x), np.ones(5)/5, mode='same'))
    os.makedirs(save_path, exist_ok=True)

    marks_file = os.path.join(save_path, f'marks_{channel}.csv')
    rest_file = os.path.join(save_path, f'rest_{channel}.csv')
    marks = pd.read_csv(marks_file)['Onset'].tolist() if os.path.exists(marks_file) else []
    rests = list(pd.read_csv(rest_file).itertuples(index=False, name=None)) if os.path.exists(rest_file) else []
    state = dict(t0=time[0], rest=rests[-1] if rests else None, clicks=None, zoom=1.0,
                 detected=[], stopped=[], message='', result=None)

    fig, (ax_emg, ax_tk) = plt.subplots(2, 1, figsize=(15, 7), sharex=True)
    top_emg, top_tk = np.percentile(np.abs(x), 99.5), np.percentile(amplitude, 99.5)
    fig.legend(handles=[Line2D([], [], color='tab:purple', lw=1.5, label='trial start'),
                        Line2D([], [], color='tab:purple', lw=1.5, ls=':', label='trial stop'),
                        Line2D([], [], color='tab:green', label='selected onset'),
                        Patch(color='tab:blue', alpha=.2, label='rest period'),
                        Line2D([], [], color='tab:orange', ls='--', label='detected onset'),
                        Line2D([], [], color='tab:orange', ls=':', label='detected offset')],
               loc='upper right', ncol=6, fontsize=8, frameon=False)

    def save():
        pd.DataFrame({'Onset': sorted(marks)}).to_csv(marks_file, index=False)
        pd.DataFrame([state['rest']] if state['rest'] else [], columns=['Start', 'Stop']).to_csv(rest_file, index=False)

    def draw():
        t0, t1 = state['t0'], state['t0'] + VIEW_S
        s = (time >= t0) & (time <= t1)
        ax_emg.clear()
        ax_tk.clear()
        ax_emg.plot(time[s], x[s], lw=.4, color='.3')
        ax_tk.plot(time[s], amplitude[s], lw=.5, color='k')
        for ax in (ax_emg, ax_tk):
            for start, stop in trials:
                if t0 <= start <= t1:
                    ax.axvline(start, color='tab:purple', lw=1.5)
                if t0 <= stop <= t1:
                    ax.axvline(stop, color='tab:purple', lw=1.5, ls=':')
            if state['rest']:
                ax.axvspan(*state['rest'], color='tab:blue', alpha=.2)
            for on in state['detected']:
                if t0 <= on <= t1:
                    ax.axvline(on, color='tab:orange', ls='--')
            for off in state['stopped']:
                if t0 <= off <= t1:
                    ax.axvline(off, color='tab:orange', ls=':')
            for on in marks:
                if t0 <= on <= t1:
                    ax.axvline(on, color='tab:green')
        ax_emg.set_xlim(t0, t1)
        ax_emg.set_ylim(-top_emg*state['zoom'], top_emg*state['zoom'])
        ax_tk.set_ylim(0, top_tk*state['zoom'])
        ax_emg.set_ylabel('Filtered EMG')
        ax_tk.set_ylabel('TKEO amplitude')
        ax_tk.set_xlabel('Time (s)')
        text = f'{title} {channel}: {len(marks)} onsets selected, rest {"set" if state["rest"] else "not set"}'
        if state['clicks'] is not None:
            text += ' | click the start and stop of the rest period'
        ax_emg.set_title(text.strip() + (f' | {state["message"]}' if state['message'] else ''), loc='left')
        fig.canvas.draw_idle()

    def run():
        if not marks or not state['rest']:
            state['message'] = 'select an onset and the rest period first'
            return
        best = optimize(time, x, marks, state['rest']).iloc[0]
        settings = {k: float(best[k]) for k in ['K', 'R_s', 'FLOOR', 'rest_level']}
        settings['C'] = int(best['C'])
        on, off = detect(time, x, settings)
        state['detected'], state['stopped'] = on.tolist(), off.tolist()
        result = pd.DataFrame({'Onset': on, 'Offset': off})
        if trials:
            result.insert(0, 'Trial', trial_of(on, *zip(*trials)))
        state['result'] = result
        with open(os.path.join(save_path, f'cfar_params_{channel}.json'), 'w') as f:
            json.dump(settings, f, indent=2)
        result.to_csv(os.path.join(save_path, f'onsets_{channel}.csv'), index=False)
        state['message'] = (f'found {int(best["found"])}/{len(marks)} selected, {int(best["in_rest"])} in rest, '
                            f'{len(on)} detected in total')
        print(title, channel, state['message'], settings)

    def on_click(event):
        toolbar = fig.canvas.manager.toolbar if fig.canvas.manager is not None else None
        if event.inaxes not in (ax_emg, ax_tk) or (toolbar is not None and toolbar.mode):
            return
        if state['clicks'] is not None:
            state['clicks'].append(event.xdata)
            if len(state['clicks']) == 2:
                state['rest'] = tuple(sorted(state['clicks']))
                state['clicks'] = None
        elif event.button == 1:
            marks.append(event.xdata)
        elif event.button == 3 and marks:
            marks.remove(min(marks, key=lambda on: abs(on - event.xdata)))
        save()
        draw()

    def on_key(event):
        if event.key == 'right':
            state['t0'] = min(state['t0'] + VIEW_S/2, time[-1] - VIEW_S)
        elif event.key == 'left':
            state['t0'] = max(state['t0'] - VIEW_S/2, time[0])
        elif event.key == 'up':
            state['zoom'] /= 1.5
        elif event.key == 'down':
            state['zoom'] *= 1.5
        elif event.key == 'r':
            state['clicks'] = []
        elif event.key == 'enter':
            run()
        draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    draw()
    return fig, state


def cfar_onsets(EMG, column, save_path, channel, start_times=None, stop_times=None, title=''):
    """Open the selection window for one EMG channel and return its detected onsets and offsets.

    EMG          DataFrame with a Time column, such as EMG_1 in main.py
    column       the EMG column to use, 'EMG1' or 'EMG2'
    save_path    folder for the marks, rest, settings and onsets files (the subject's Final folder)
    channel      name used in the file names, such as 'EMG_1_1'
    start_times, stop_times
                 trial start and stop times, drawn in purple; also adds a Trial column to the result
    title        shown in the window title, such as the subject ID

    The window blocks until you close it. Returns a DataFrame with Onset and Offset (and Trial)
    from the last time you pressed Enter, or None if you never did.
    """
    import matplotlib.pyplot as plt
    fig, state = selection_window(EMG['Time'], EMG[column], save_path, channel, start_times, stop_times, title)
    plt.show(block=True)
    plt.close(fig)
    return state['result']


if __name__ == '__main__':
    subject, channel = sys.argv[1], sys.argv[2]
    _, sensor, ch = channel.split('_')
    folder = os.path.join(final_data_path, subject)
    cfar_onsets(pd.read_csv(os.path.join(folder, f'EMG_{sensor}.csv')), f'EMG{ch}', folder, channel, title=subject)
