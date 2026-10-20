import pandas as pd
import numpy as np
import re
from scipy.signal import find_peaks
import glob
import os
import warnings

class MissingFileError(Exception):
    """Custom exception for missing files."""
    pass

def find_file(subject_dir, pattern):
    search_pattern = os.path.join(glob.escape(subject_dir), pattern)
    matches = sorted(glob.glob(search_pattern))
    if not matches:
        raise MissingFileError(f"no file matching '{pattern}' in {subject_dir}")
    if len(matches) > 1:
        warnings.warn(f"{len(matches)} files match '{pattern}' in {subject_dir}; using {matches[0]}")
    return matches[0]

def delsys_csv_breakdown(delsys_csv):
    """
    The Delsys csv needs read in in different parts and reworked to 
    make columns easily accessible. 
    delsys_csv: path
        the path to the csv file containing the delsys data

    Returns
    time: pd.DataFrame
        the first two rows of the emg.csv containing the time and date of the recording
    config: pd.DataFrame
        the configuration information for the recording (units, sampling rate, etc.). Column names are made unique.
    data: pd.DataFrame
        the recorded data from delsys with cleaned column names and converted to numeric values.
    """
    time = pd.read_csv(delsys_csv, nrows = 2)

    config = pd.read_csv(delsys_csv, skiprows = 5, nrows = 1)
    config.replace(" ", float("NaN"), inplace = True)
    config.dropna(axis = 1, inplace = True) # Just retain the columns that have units
    config = config

    #make duplicate column names unique
    cf_new_cols = []
    cf_col_counts = {}
    for cf_col in config.columns:
        if cf_col in cf_col_counts:
            cf_col_counts[col] += 1
            cf_new_cols.append(f"{cf_col} {cf_col_counts[cf_col]}")
        else:
            cf_col_counts[cf_col] = 1
            cf_new_cols.append(cf_col)
    config.columns = cf_new_cols

    data = pd.read_csv(delsys_csv,header=5,skiprows=[6,7],index_col=False, skipinitialspace=True)
    data.columns = (
        data.columns
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"\.\d+$", "", regex=True))
    
    #make duplicate column names unique
    new_cols = []
    col_counts = {}
    for col in data.columns:
        if col in col_counts:
            col_counts[col] += 1
            new_cols.append(f"{col} {col_counts[col]}")
        else:
            col_counts[col] = 1
            new_cols.append(col)
    data.columns = new_cols

    for col in data.columns:
        data[col] = data[col].astype(str).str.replace(' ', '', regex=False)
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    data = data.dropna(how="all")

    return (time, config, data)


def trial_determination(mat_data, target_data):
    """
    Determine the start and stop times for each trial based on the target data.

    Parameters:
    mat_data (pd.DataFrame): The DataFrame containing the mat data with columns.
    target_data (pd.DataFrame): The DataFrame containing the target data with columns ['hitTime'].

    Returns:
    tuple: Two lists containing the start times and stop times for each trial.
    """

    start_times = []
    stop_times = []
    hit_times = []

    mat_data['derivative'] = mat_data['Total'].diff()

    start_ind, _ = find_peaks(-mat_data['derivative'], distance = 20, height = 300)
    stop_ind, _ = find_peaks(mat_data['derivative'], distance = 20, height = 300)

    starts = mat_data['Time'].iloc[start_ind]
    stops = mat_data['Time'].iloc[stop_ind]

    for i in range(len(target_data)):
        hit_time = target_data['hitTime'].iloc[i]
        nearest_start = starts[starts <= hit_time].max()
        nearest_stop = stops[stops >= hit_time].min()
    
        if 0<i<len(target_data)-1:      # only consider trials that have their own start and stop times, select only trials that occur after the EMG has started, and consider the starts and stops closest to the hit of the trial.
            next_hit_time = target_data['hitTime'].iloc[i+1]
            previous_hit_time = target_data['hitTime'].iloc[i-1]
            if (
                bool(nearest_start > 0) and bool(nearest_start >= previous_hit_time)and bool(nearest_stop <= next_hit_time)
            ):
                start_times.append(nearest_start)
                stop_times.append(nearest_stop)
                hit_times.append(hit_time)
          

    return start_times, stop_times, hit_times