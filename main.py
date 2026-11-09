#Will take input folders and output analyzed data
import pandas as pd
import glob
import os
from analysis import peak_analysis, emg_sanity_check
from extract import delsys_csv_breakdown, trial_determination, MissingFileError, find_file
from preprocessing import butter_bandpass_filter, resamp, rms_smoothing, IMU_smooth, preprocess_IMU
import numpy as np
from scipy.signal import find_peaks
from CFAR_onset import cfar_onsets



initial_data_path = '/Users/clmckeev/Desktop/Raw/All'
final_data_path = '/Users/clmckeev/Desktop/Raw/Final'
folders = [ '019']
switched = ['034', '048'] #Folders that have switched sensor placements
required_files = {'EMG': '*EMG.csv', 'Mat': 'mat*.csv', 'Target': 'target*.csv'}
misses = {}
cohort_info = pd.read_csv('/Users/clmckeev/Desktop/Raw/cohort_data.csv', dtype=str, index_col=False)

for subject_id in folders: #Collect recorded data of each participant
    subject_dir = os.path.join(initial_data_path, subject_id)
    end_subject_dir = os.path.join(final_data_path, subject_id)
    os.makedirs(end_subject_dir, exist_ok=True)
    try: paths = {name:find_file(subject_dir, pattern) for name, pattern in required_files.items()}
    except MissingFileError as e:
        print(f"Skipping {subject_id} due to missing file: {e}")
        misses[subject_id] = str(e)
        continue
    info = {}
    (DT, sensor_config, delsys_data) = delsys_csv_breakdown(paths['EMG'])
    p_data = pd.DataFrame(columns = cohort_info.columns)
    p_data = cohort_info[cohort_info['ID #'] == subject_id]
    m_data = pd.read_csv(paths['Mat'])
    mat_data = pd.DataFrame()
    mat_data['Time'] = m_data.iloc[:, 0]
    mat_data['Total'] = m_data.iloc[:, 1:].sum(axis=1) #Only concerned with time of lift and not location of the hand at this time, so we can consider the total of all cells in the pressure mat 
    target_data = pd.read_csv(paths['Target'])

                  
    #Collect all test info in one place
    info = p_data.iloc[0].to_dict()
    info['Test Length'] = DT.iloc[1,1]
    for measurement in sensor_config.columns:
        info[measurement] = sensor_config[measurement]
    info = pd.DataFrame(info)
    info.to_csv(os.path.join(end_subject_dir,'testing_info.csv'), index=False, header=True) #Save info file in individuals folder

    #Seperate all data into desired chunks
    #Standard is EMG_1_1 is Posterior Deltoid EMG_1_2 is Anterior Deltoid and the IMU_1 is on the acromion
    #EMG_2_1 is the IDF EMG_2_2 is Brachioradialus and IMU_2 is on the back of the hand
    if subject_id in switched:
        delsys_data = delsys_data.rename(columns={'EMG 1 (mV)': 'EMG 1 (mV) 2', 'EMG 1 (mV) 2': 'EMG 1 (mV)',
                                                  'EMG 2 (mV)': 'EMG 2 (mV) 2', 'EMG 2 (mV) 2': 'EMG 2 (mV)',
                                                  'ACC X (G)': 'ACC X (G) 2', 'ACC Y (G)': 'ACC Y (G) 2', 'ACC Z (G)': 'ACC Z (G) 2',
                                                  'ACC X (G) 2': 'ACC X (G)', 'ACC Y (G) 2': 'ACC Y (G)', 'ACC Z (G) 2': 'ACC Z (G)',
                                                  'GYRO X (deg/s)': 'GYRO X (deg/s) 2', 'GYRO Y (deg/s)': 'GYRO Y (deg/s) 2', 'GYRO Z (deg/s)': 'GYRO Z (deg/s) 2',
                                                  'GYRO X (deg/s) 2': 'GYRO X (deg/s)', 'GYRO Y (deg/s) 2': 'GYRO Y (deg/s)', 'GYRO Z (deg/s) 2': 'GYRO Z (deg/s)'})
        delsys_data = delsys_data.rename(columns={'EMG 1 Time Series (s)': 'EMG 1 Time Series (s) 2', 'EMG 1 Time Series (s) 2': 'EMG 1 Time Series (s)',
                                                  'ACC X Time Series (s)': 'ACC X Time Series (s) 2', 'ACC X Time Series (s) 2': 'ACC X Time Series (s)',
                                                  'GYRO X Time Series (s)': 'GYRO X Time Series (s) 2', 'GYRO X Time Series (s) 2': 'GYRO X Time Series (s)'})

    rawEMG_1 = delsys_data[['EMG 1 Time Series (s)', 'EMG 1 (mV)', 'EMG 2 (mV)']].dropna()
    rawEMG_1.columns = ['Time', 'EMG1', 'EMG2']
    rawEMG_1.to_csv(os.path.join(end_subject_dir,'EMG_1_raw.csv'), index=False, header=True) 
    rawAcc_1 = delsys_data[['ACC X Time Series (s)', 'ACC X (G)', 'ACC Y (G)', 'ACC Z (G)']].dropna()
    rawAcc_1.columns = ['Time', 'X', 'Y', 'Z']
    rawAcc_1.to_csv(os.path.join(end_subject_dir,'Acc_1_raw.csv'), index=False, header=True) 
    rawGyro_1 = delsys_data[['GYRO X Time Series (s)', 'GYRO X (deg/s)', 'GYRO Y (deg/s)', 'GYRO Z (deg/s)']].dropna()
    rawGyro_1.columns = ['Time', 'X', 'Y', 'Z']
    rawGyro_1.to_csv(os.path.join(end_subject_dir,'Gyro_1_raw.csv'), index=False, header=True) 

    rawEMG_2 = delsys_data[['EMG 1 Time Series (s) 2', 'EMG 1 (mV) 2', 'EMG 2 (mV) 2']].dropna()
    rawEMG_2.columns = ['Time', 'EMG1', 'EMG2']
    rawEMG_2.to_csv(os.path.join(end_subject_dir,'EMG_2_raw.csv'), index=False, header=True) 
    rawAcc_2 = delsys_data[['ACC X Time Series (s) 2', 'ACC X (G) 2', 'ACC Y (G) 2', 'ACC Z (G) 2']].dropna()
    rawAcc_2.columns = ['Time', 'X', 'Y', 'Z']
    rawAcc_2.to_csv(os.path.join(end_subject_dir,'Acc_2_raw.csv'), index=False, header=True) 
    rawGyro_2 = delsys_data[['GYRO X Time Series (s) 2', 'GYRO X (deg/s) 2', 'GYRO Y (deg/s) 2', 'GYRO Z (deg/s) 2']].dropna()
    rawGyro_2.columns = ['Time', 'X', 'Y', 'Z']
    rawGyro_2.to_csv(os.path.join(end_subject_dir,'Gyro_2_raw.csv'), index=False, header=True)

    analog_candidates = [col for col in delsys_data.columns if 'analog' in str(col).lower()]
    if len(analog_candidates) < 2:
        print(f"Skipping {subject_id} due to missing Analog data.")
        misses[subject_id] = "Missing Analog data"
        continue
    analog_cols = analog_candidates[:2]
    Analog = delsys_data[analog_cols].dropna()
    Analog.columns = ['Time', 'Analog']
    Analog['Analog'] = Analog['Analog'].to_numpy(dtype=float)
    Analog.to_csv(os.path.join(end_subject_dir,'Analog.csv'), index=False, header=True)


    ## Preprocessing data

     # Sync analog and digital data 
    analog_peaks, _ = find_peaks(np.nan_to_num(Analog['Analog']), height=0.003, distance = 100)
    if len(analog_peaks) < 20:
        print(f"Skipping {subject_id} due to incorrectly recorded Analog data")
        misses[subject_id] = "Incorrect Analog data"
        continue

    a_event_time = Analog['Time'].iloc[analog_peaks].dropna()
    target_data['hitTime'] = (target_data['hitTime']/1000) # Convert milliseconds to seconds and drop NaN values
       
    t_delay = float(a_event_time.iloc[-1] - target_data['hitTime'].iloc[-1])  # Calculate the time difference between the two events, this gives the time difference for all delsys times to local computer time
    
    target_data['hitTime'] = target_data['hitTime'] + t_delay 
    mat_data['Time'] = (mat_data['Time']/1000) + t_delay # Convert to seconds and add delay

     # Determine trial start and stop times and remove unsuccesful trials, clean_EMG only includes data within the start and stop times of each trial
    start_times, stop_times, hit_times = trial_determination(mat_data, target_data)
    if len(start_times) < 10 or len(stop_times) < 10:
        print(f"Skipping {subject_id} due to incorrectly recorded Mat data")
        misses[subject_id] = "Incorrect Mat data"
        continue
    trial_info = pd.DataFrame({'Start Time': start_times, 'Stop Time': stop_times, 'Hit Time': hit_times})
    trial_info.to_csv(os.path.join(end_subject_dir,'trial_info.csv'), index=False, header=True)

     ## EMG data
    samp_freqEMG = float(sensor_config[' EMG 1 (mV).1'].iloc[0].split()[0])
    print(samp_freqEMG)

     # Apply a 4th order Butterworth bandpass (20-450Hz) using the sampling frequency from Delsys
    filtEMG_1 = pd.DataFrame(columns=['Time', 'EMG1', 'EMG2'])
    filtEMG_2 = pd.DataFrame(columns=['Time', 'EMG1', 'EMG2'])
    filtEMG_1['Time'] = rawEMG_1['Time']
    filtEMG_2['Time'] = rawEMG_2['Time']
    filtEMG_1['EMG1'] = butter_bandpass_filter(rawEMG_1['EMG1'],20, 450, fs=samp_freqEMG)
    filtEMG_1['EMG2'] = butter_bandpass_filter(rawEMG_1['EMG2'],20, 450, fs=samp_freqEMG)
    filtEMG_2['EMG1'] = butter_bandpass_filter(rawEMG_2['EMG1'],20, 450, fs=samp_freqEMG)
    filtEMG_2['EMG2'] = butter_bandpass_filter(rawEMG_2['EMG2'],20, 450, fs=samp_freqEMG)

     # Resample to 1000 Hz
    rsEMG_1 = resamp(filtEMG_1)
    rsEMG_2 = resamp(filtEMG_2)

     # Shortened EMG data to only include the active testing time (between the first start time and last stop time
    EMG_1 = rsEMG_1[(rsEMG_1['Time'] >= start_times[0]) & (rsEMG_1['Time'] <= stop_times[-1])].copy()
    EMG_2 = rsEMG_2[(rsEMG_2['Time'] >= start_times[0]) & (rsEMG_2['Time'] <= stop_times[-1])].copy()

     # Normalize data using the maximum within task value for each individual muscle
    EMG_1['EMG1'] = EMG_1['EMG1']/(EMG_1['EMG1'].abs().max())
    EMG_1['EMG2'] = EMG_1['EMG2']/(EMG_1['EMG2'].abs().max())
    EMG_2['EMG1'] = EMG_2['EMG1']/(EMG_2['EMG1'].abs().max())
    EMG_2['EMG2'] = EMG_2['EMG2']/(EMG_2['EMG2'].abs().max())
    EMG_1.to_csv(os.path.join(end_subject_dir,'EMG_1.csv'), index=False, header=True)
    EMG_2.to_csv(os.path.join(end_subject_dir,'EMG_2.csv'), index=False, header=True)


    smooth_EMG_1 = EMG_1.copy()
    smooth_EMG_2 = EMG_2.copy()
    smooth_EMG_1['EMG1'] = rms_smoothing(EMG_1['EMG1'], 300) # window size of 30 samples (30 ms)
    smooth_EMG_1['EMG2'] = rms_smoothing(EMG_1['EMG2'], 300)
    smooth_EMG_2['EMG1'] = rms_smoothing(EMG_2['EMG1'], 300)
    smooth_EMG_2['EMG2'] = rms_smoothing(EMG_2['EMG2'], 300)
    smooth_EMG_1.to_csv(os.path.join(end_subject_dir,'rms_EMG_1.csv'), index=False, header=True)
    smooth_EMG_2.to_csv(os.path.join(end_subject_dir,'rms_EMG_2.csv'), index=False, header=True)

    emg_sanity_check(rawEMG_1, smooth_EMG_1, start_times, stop_times, target_data['hitTime'], save_path=end_subject_dir)

     #Onset detection using CFAR
    onsets_EMG_1_1 = cfar_onsets(EMG_1, 'EMG1', channel = 'EMG_1_1', start_times = start_times, stop_times = stop_times, save_path = os.path.join(end_subject_dir,'Graphs','EMG_1_1_onsets'), title = 'EMG 1.1 Onset Detection')
    onsets_EMG_1_2 = cfar_onsets(EMG_1, 'EMG2', channel = 'EMG_1_2', start_times = start_times, stop_times = stop_times, save_path = os.path.join(end_subject_dir,'Graphs','EMG_1_2_onsets'), title = 'EMG 1.2 Onset Detection')
    onsets_EMG_2_1 = cfar_onsets(EMG_2, 'EMG1', channel = 'EMG_2_1', start_times = start_times, stop_times = stop_times, save_path = os.path.join(end_subject_dir,'Graphs','EMG_2_1_onsets'), title = 'EMG 2.1 Onset Detection')
    onsets_EMG_2_2 = cfar_onsets(EMG_2, 'EMG2', channel = 'EMG_2_2', start_times = start_times, stop_times = stop_times, save_path = os.path.join(end_subject_dir,'Graphs','EMG_2_2_onsets'), title = 'EMG 2.2 Onset Detection')

    
    ##IMU

     # Resample, remove gyro bias, 5 Hz zero-phase low-pass, and separate gravity (see preprocess_imu)
     # The hand rests on the mat between one trial's stop and the next trial's start, so those periods are used as static intervals for gyro bias
    static_intervals = list(zip(stop_times[:-1], start_times[1:]))
     # beta=0.001 (~7 s time constant at ~148 Hz) instead of the paper's thigh/walking value of 0.01 (~0.7 s), which absorbs part of a ~2 s reach into the gravity estimate
    proc_acc_2, proc_gyro_2 = preprocess_IMU(rawAcc_2, rawGyro_2, static_intervals, cutoff=5, order=4, beta=0.001)

     #Only within trial data
     # start/stop are times (s), so select rows by the Time column and stack trials in long format with a Trial column
    accel_trials = []
    gyro_trials = []
    for trial, (start, stop) in enumerate(zip(start_times, stop_times), start=1):
        accel_trials.append(proc_acc_2[(proc_acc_2['Time'] >= start) & (proc_acc_2['Time'] <= stop)].assign(Trial=trial))
        gyro_trials.append(proc_gyro_2[(proc_gyro_2['Time'] >= start) & (proc_gyro_2['Time'] <= stop)].assign(Trial=trial))
    accel_tdata_2 = pd.concat(accel_trials, ignore_index=True)[['Trial', 'Time', 'X', 'Y', 'Z', 'Mag', 'X_lin', 'Y_lin', 'Z_lin', 'Mag_lin']]
    gyro_tdata_2 = pd.concat(gyro_trials, ignore_index=True)[['Trial', 'Time', 'X', 'Y', 'Z']]
    accel_tdata_2['Mag'] = np.sqrt(accel_tdata_2[['X', 'Y', 'Z']].pow(2).sum(axis = 1))
        
    accel_tdata_2.to_csv(os.path.join(end_subject_dir,'accel_trial_data.csv'), index = False, header =True)
    gyro_tdata_2.to_csv(os.path.join(end_subject_dir,'gyro_trial_data.csv'), index = False, header =True)

     #Smooth IMU data with Gaussian filter
    accel = accel_tdata_2.copy()
    for c in ['X', 'Y', 'Z', 'Mag']:
        accel[c] = IMU_smooth(accel_tdata_2[c],96)

   
    ## Analysis
    Measures = pd.DataFrame(columns=['EMG_11_FanoFactor', 'EMG_11_Entropy', 'EMG_12_FanoFactor', 'EMG_12_Entropy', 'EMG_21_FanoFactor', 'EMG_21_Entropy', 'EMG_22_FanoFactor', 'EMG_22_Entropy', 'Accel_FanoFactor', 'Accel_Entropy'])
     # change between subsequent maxima and minima within trials
    EMG_11_diff, EMG_11_FF, EMG_11_Entropy = peak_analysis(start_times, stop_times, smooth_EMG_1[['Time', 'EMG1']], save_path = os.path.join(end_subject_dir,'Graphs','EMG_1_1_extrema'), plot_show=False, decimals=2)
    EMG_12_diff, EMG_12_FF, EMG_12_Entropy = peak_analysis(start_times, stop_times, smooth_EMG_1[['Time', 'EMG2']], save_path = os.path.join(end_subject_dir,'Graphs','EMG_1_2_extrema'), plot_show=False, decimals=2)
    EMG_21_diff, EMG_21_FF, EMG_21_Entropy = peak_analysis(start_times, stop_times, smooth_EMG_2[['Time', 'EMG1']], save_path = os.path.join(end_subject_dir,'Graphs','EMG_2_1_extrema'), plot_show=False, decimals=2)
    EMG_22_diff, EMG_22_FF, EMG_22_Entropy = peak_analysis(start_times, stop_times, smooth_EMG_2[['Time', 'EMG2']], save_path = os.path.join(end_subject_dir,'Graphs','EMG_2_2_extrema'), plot_show=False, decimals=2)

    EMG_11_diff.to_csv(os.path.join(end_subject_dir,'EMG_11_extrema.csv'), index=False, header=True)
    EMG_12_diff.to_csv(os.path.join(end_subject_dir,'EMG_12_extrema.csv'), index=False, header=True)
    EMG_21_diff.to_csv(os.path.join(end_subject_dir,'EMG_21_extrema.csv'), index=False, header=True)
    EMG_22_diff.to_csv(os.path.join(end_subject_dir,'EMG_22_extrema.csv'), index=False, header=True)

    accel_diff, FF, Entropy = peak_analysis(start_times, stop_times, accel[['Time', 'Mag']], save_path= os.path.join(end_subject_dir,'Graphs','accel_extrema'), plot_show = False, decimals = 2, units = "G")
    accel_diff.to_csv(os.path.join(end_subject_dir,'accel_extrema_preprocessed.csv'), index=False, header=True)
 
    Measures.loc[0] = [EMG_11_FF, EMG_11_Entropy, EMG_12_FF, EMG_12_Entropy, EMG_21_FF, EMG_21_Entropy, EMG_22_FF, EMG_22_Entropy, FF, Entropy]
    Measures.to_csv(os.path.join(end_subject_dir,'Measures.csv'), index=False, header=True)

print("Processing complete. The following subjects were skipped due to missing files or data:")
for subject_id, reason in misses.items():
    print(f"Subject {subject_id}: {reason}")