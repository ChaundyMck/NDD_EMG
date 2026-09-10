import pandas as pd
import numpy as np
import re

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

    data = pd.read_csv(delsys_csv,header=5,skiprows=[6,7],index_col=False)
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
