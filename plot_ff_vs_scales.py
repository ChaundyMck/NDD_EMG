"""Plot Fano factor (FF) or entropy against the ASD and ADHD scales, colored by official diagnosis.

Reads each participant's Measures.csv (FF or entropy values) and testing_info.csv (Diagnosis,
Age and scale scores), then saves to the Final folder, where <Measure> is FF or Entropy:
    <Measure>_vs_scales.csv         - one row per participant
    <Measure>_vs_scales_under12.png - participants younger than 12
    <Measure>_vs_scales_12plus.png  - participants aged 12 or older
Each plot has one panel per channel and scale.

Scales are recorded in several formats. Fractions are computed into values ("18/50" becomes
0.36) and bare scores stay as recorded ("15" stays 15), so the two are not on the same scale.
Marker shape shows each score's recorded format.

Usage:
    python plot_ff_vs_scales.py [final_folder] [--measure {ff,entropy}] [--show]
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_ff_by_diagnosis import (AXIS, DEFAULT_FINAL_PATH, GRID, INK, MEASURES, SECONDARY_INK, SURFACE,
                                  collect_measure, find_csv, measure_columns)

# testing_info.csv column -> x-axis label
SCALES = {
    'ASD scale': 'ASD scale score',
    'ADHD scale total': 'ADHD scale total score',
}

AGE_CUTOFF = 12  # participants younger than this are plotted separately

# Colors stay fixed to a diagnosis, whichever groups are present in the data
DIAGNOSIS_COLORS = {'NT': '#2a78d6', 'ADHD': '#eb6834', 'AuDHD': '#1baf7a'}
EXTRA_COLORS = ['#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']

NO_MAX = 'max not recorded'
SHAPES = ['o', 's', '^', 'D', 'v', 'P', 'X']


def parse_score(text):
    """Turn '18/50' into (0.36, 'out of 50'); a bare '15' gives (15.0, NO_MAX)."""
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*(?:/\s*(\d+))?', str(text).strip())
    if match is None:
        return np.nan, None
    score, maximum = match.groups()
    if maximum is None:
        return float(score), NO_MAX
    if int(maximum) == 0:
        return np.nan, None
    return float(score) / int(maximum), f'out of {maximum}'


def add_info(data, final_path):
    """Add Age, plus a recorded text, score and format column for each scale in SCALES."""
    data = data.copy()
    infos = [pd.read_csv(find_csv(os.path.join(final_path, subject_id), 'testing_info.csv'), dtype=str, keep_default_na=False)
             for subject_id in data['ID']]
    ages = [info['Age'].iloc[0] if 'Age' in info.columns else '' for info in infos]
    data['Age'] = pd.to_numeric(pd.Series(ages, index=data.index), errors='coerce')
    for scale in SCALES:
        recorded = [info[scale].iloc[0] if scale in info.columns else '' for info in infos]
        parsed = [parse_score(text) for text in recorded]
        data[f'{scale} recorded'] = recorded
        data[f'{scale} score'] = [score for score, _ in parsed]
        data[f'{scale} format'] = [score_format for _, score_format in parsed]
    return data


def diagnosis_colors(diagnoses):
    """Return {diagnosis: color}; labels outside DIAGNOSIS_COLORS take the extra colors in order."""
    extras = [d for d in diagnoses if d not in DIAGNOSIS_COLORS]
    if len(extras) > len(EXTRA_COLORS):
        raise ValueError(f'Too many diagnosis labels to color distinctly: {extras}')
    colors = dict(zip(extras, EXTRA_COLORS))
    return {d: DIAGNOSIS_COLORS.get(d, colors.get(d)) for d in diagnoses}


def format_order(score_format):
    """Sort NO_MAX first, then by increasing maximum."""
    return (0, 0) if score_format == NO_MAX else (1, int(score_format.split()[-1]))


def score_formats(data):
    """Return the score formats present in data, in legend order."""
    formats = pd.concat([data[f'{scale} format'] for scale in SCALES]).dropna().unique()
    return sorted(formats, key=format_order)


def plot_measure_vs_scales(data, measure, colors, shapes, group_label, save_path):
    """Draw a grid of scatter plots: one row per scale, one column per channel."""
    diagnoses = sorted(data['Diagnosis'].unique())
    formats = score_formats(data)
    columns = measure_columns(measure)

    fig, axes = plt.subplots(len(SCALES), len(columns), figsize=(18, 8.5), sharex='row', sharey='col',
                             facecolor=SURFACE, squeeze=False)

    for row, (scale, x_label) in enumerate(SCALES.items()):
        for col, (column, title) in enumerate(columns.items()):
            ax = axes[row, col]
            usable = data[np.isfinite(data[f'{scale} score']) & np.isfinite(data[column])]
            for (diagnosis, score_format), group in usable.groupby(['Diagnosis', f'{scale} format']):
                ax.scatter(group[f'{scale} score'], group[column], s=56, color=colors[diagnosis],
                           marker=shapes[score_format], edgecolors=SURFACE, linewidths=1, zorder=3)

            ax.set_facecolor(SURFACE)
            if row == 0:
                ax.set_title(title, loc='left', fontsize=12, color=INK)
            if col == 0:
                ax.set_ylabel(MEASURES[measure]['axis'], color=SECONDARY_INK)
            ax.set_xlabel(f'{x_label} (n={len(usable)})', color=SECONDARY_INK)
            ax.tick_params(colors=SECONDARY_INK, length=0, labelbottom=True, labelleft=True)
            ax.grid(color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ['top', 'right', 'left']:
                ax.spines[side].set_visible(False)
            ax.spines['bottom'].set_color(AXIS)

    def handle(label, color, marker):
        return Line2D([], [], linestyle='', marker=marker, markersize=8, color=color,
                      markeredgecolor=SURFACE, label=label)

    fig.suptitle(
        f"{MEASURES[measure]['name']} vs. ASD and ADHD scale scores: {group_label} (N={len(data)})\n"
        'Each point is one participant. Fractions are computed into values and bare scores are as recorded, '
        'so compare across marker shapes with care.',
        x=0.01, ha='left', fontsize=13, color=INK,
    )
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    diagnosis_legend = fig.legend(handles=[handle(d, colors[d], 'o') for d in diagnoses], title='Diagnosis (color)',
                                  loc='upper left', bbox_to_anchor=(0.9, 0.9), frameon=False, labelcolor=INK)
    fig.add_artist(diagnosis_legend)
    fig.legend(handles=[handle(f, SECONDARY_INK, shapes[f]) for f in formats], title='Score format (shape)',
               loc='upper left', bbox_to_anchor=(0.9, 0.62), frameon=False, labelcolor=INK)
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

    data = add_info(data, args.final_path)
    print(data['Diagnosis'].value_counts().sort_index().to_string())
    no_age = data.loc[data['Age'].isna(), 'ID'].tolist()
    if no_age:
        print(f'No usable Age, left out of both plots: {", ".join(no_age)}')
    for scale in SCALES:
        print(f'\n{scale} formats:')
        print(data[f'{scale} format'].value_counts().to_string())
        unusable = data.loc[data[f'{scale} score'].isna(), 'ID'].tolist()
        if unusable:
            print(f'No usable {scale} score, left out of those panels: {", ".join(unusable)}')

    file_prefix = MEASURES[args.measure]['file']
    data.to_csv(os.path.join(args.final_path, f'{file_prefix}_vs_scales.csv'), index=False)

    # Colors and shapes come from everyone so they mean the same thing in both plots
    colors = diagnosis_colors(sorted(data['Diagnosis'].unique()))
    shapes = dict(zip(score_formats(data), SHAPES))
    age_groups = [
        (f'younger than {AGE_CUTOFF}', f'{file_prefix}_vs_scales_under{AGE_CUTOFF}.png', data['Age'] < AGE_CUTOFF),
        (f'age {AGE_CUTOFF} and older', f'{file_prefix}_vs_scales_{AGE_CUTOFF}plus.png', data['Age'] >= AGE_CUTOFF),
    ]
    print()
    for group_label, file_name, in_group in age_groups:
        if not in_group.any():
            print(f'No participants {group_label}, so no plot was made')
            continue
        save_path = os.path.join(args.final_path, file_name)
        plot_measure_vs_scales(data[in_group], args.measure, colors, shapes, group_label, save_path)
        print(f'Saved {save_path} ({in_group.sum()} participants)')
    if args.show:
        plt.show()


if __name__ == '__main__':
    main()
