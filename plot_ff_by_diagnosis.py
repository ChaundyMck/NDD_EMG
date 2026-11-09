"""Plot Fano factor (FF) or entropy by diagnosis for every participant folder in Final.

Reads each participant's Measures.csv (FF or entropy values) and testing_info.csv (Diagnosis),
then saves to the Final folder, where <Measure> is FF or Entropy:
    <Measure>_by_diagnosis.csv  - one row per participant
    <Measure>_by_diagnosis.png  - one panel per channel, one column of dots per diagnosis

Usage:
    python plot_ff_by_diagnosis.py [final_folder] [--measure {ff,entropy}] [--show]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_FINAL_PATH = '/Users/clmckeev/Desktop/Raw/Final'

# Measures.csv column prefix -> panel title (sensor placement as documented in main.py)
CHANNELS = {
    'EMG_11': 'Posterior deltoid (EMG 1-1)',
    'EMG_12': 'Anterior deltoid (EMG 1-2)',
    'EMG_21': 'IDF (EMG 2-1)',
    'EMG_22': 'Brachioradialis (EMG 2-2)',
    'Accel': 'Hand acceleration',
}

# column_suffix: Measures.csv column ending; name: text in plots; axis: y-axis label; file: output file prefix
MEASURES = {
    'ff': {'column_suffix': 'FanoFactor', 'name': 'Fano factor', 'axis': 'Fano factor', 'file': 'FF'},
    'entropy': {'column_suffix': 'Entropy', 'name': 'Entropy', 'axis': 'Entropy (nats)', 'file': 'Entropy'},
}

DOT_COLOR = '#2a78d6'
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
SECONDARY_INK = '#52514e'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'


def find_csv(folder, name):
    """Return the path to a file in folder, ignoring filename case, or None."""
    for file in os.listdir(folder):
        if file.lower() == name.lower():
            return os.path.join(folder, file)
    return None


def measure_columns(measure):
    """Return {Measures.csv column: panel title} for a key of MEASURES."""
    suffix = MEASURES[measure]['column_suffix']
    return {f'{channel}_{suffix}': title for channel, title in CHANNELS.items()}


def collect_measure(final_path, measure):
    """Return (one row per participant, {folder: reason it was skipped})."""
    columns = measure_columns(measure)
    rows = []
    skipped = {}
    for subject_id in sorted(os.listdir(final_path)):
        folder = os.path.join(final_path, subject_id)
        if not os.path.isdir(folder):
            continue

        measures_path = find_csv(folder, 'Measures.csv')
        info_path = find_csv(folder, 'testing_info.csv')
        if measures_path is None or info_path is None:
            missing = [n for n, p in [('Measures.csv', measures_path), ('testing_info.csv', info_path)] if p is None]
            skipped[subject_id] = 'missing ' + ' and '.join(missing)
            continue

        # keep_default_na=False so a diagnosis written as "None" or "NA" stays a label
        # rather than being read as a missing value
        info = pd.read_csv(info_path, dtype=str, keep_default_na=False)
        
        # testing_info.csv repeats the participant's details on every row
        diagnosis = info['Diagnosis'].iloc[0].strip()
        if not diagnosis:
            skipped[subject_id] = 'blank Diagnosis in testing_info.csv'
            continue

        measures = pd.read_csv(measures_path)
        absent = [c for c in columns if c not in measures.columns]
        if absent or measures.empty:
            skipped[subject_id] = f"Measures.csv missing {', '.join(absent) or 'data'}"
            continue

        row = {'ID': subject_id, 'Diagnosis': diagnosis}
        row.update({c: pd.to_numeric(measures[c], errors='coerce').iloc[0] for c in columns})
        rows.append(row)

    return pd.DataFrame(rows), skipped


def plot_measure(data, measure, save_path):
    """Draw one strip plot per channel, with a median line for each diagnosis."""
    diagnoses = sorted(data['Diagnosis'].unique())
    rotate = len(diagnoses) > 4 or max(len(d) for d in diagnoses) > 8
    rng = np.random.default_rng(0)  # fixed seed so the jitter is the same on every run

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), facecolor=SURFACE)
    axes = axes.ravel()
    axes[-1].set_visible(False)  # five channels in a 2x3 grid

    for ax, (column, title) in zip(axes, measure_columns(measure).items()):
        tick_labels = []
        for x, diagnosis in enumerate(diagnoses):
            values = data.loc[data['Diagnosis'] == diagnosis, column]
            values = values[np.isfinite(values)]
            tick_labels.append(f'{diagnosis}\nn={len(values)}')
            if values.empty:
                continue
            jitter = rng.uniform(-0.12, 0.12, len(values))
            ax.scatter(x + jitter, values, s=48, color=DOT_COLOR, edgecolors=SURFACE, linewidths=1, zorder=3)
            ax.hlines(values.median(), x - 0.28, x + 0.28, color=INK, linewidth=2, zorder=4)

        ax.set_facecolor(SURFACE)
        ax.set_title(title, loc='left', fontsize=12, color=INK)
        ax.set_ylabel(MEASURES[measure]['axis'], color=SECONDARY_INK)
        ax.set_xlim(-0.6, len(diagnoses) - 0.4)
        ax.set_xticks(range(len(diagnoses)))
        ax.set_xticklabels(tick_labels, rotation=35 if rotate else 0, ha='right' if rotate else 'center')
        ax.tick_params(colors=SECONDARY_INK, length=0)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ['top', 'right', 'left']:
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color(AXIS)

    fig.suptitle(
        f"{MEASURES[measure]['name']} by diagnosis\nEach dot is one participant; black line is the group median. Y-axes differ between panels.",
        x=0.01, ha='left', fontsize=13, color=INK,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, facecolor=SURFACE)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('final_path', nargs='?', default=DEFAULT_FINAL_PATH, help='folder holding the participant folders')
    parser.add_argument('--measure', choices=MEASURES, default='ff', help='what to plot (default: ff)')
    parser.add_argument('--show', action='store_true', help='also open the plot in a window')
    args = parser.parse_args()

    data, skipped = collect_measure(args.final_path, args.measure)

    print(f'Collected {len(data)} participants.')
    for subject_id, reason in skipped.items():
        print(f'Skipped {subject_id}: {reason}')
    if data.empty:
        return
    print(data['Diagnosis'].value_counts().sort_index().to_string())
    unusable = data[list(measure_columns(args.measure))].isna().sum()
    for column, count in unusable[unusable > 0].items():
        print(f'{column}: {count} participant(s) have no usable value')

    file_prefix = MEASURES[args.measure]['file']
    data.to_csv(os.path.join(args.final_path, f'{file_prefix}_by_diagnosis.csv'), index=False)
    save_path = os.path.join(args.final_path, f'{file_prefix}_by_diagnosis.png')
    plot_measure(data, args.measure, save_path)
    print(f'Saved {save_path}')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
