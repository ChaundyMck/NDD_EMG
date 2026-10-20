#Will take input folders and output analyzed data
from more_itertools import only
import pandas as pd
import glob
import os
from analysis import peak_analysis, emg_sanity_check
from extract import delsys_csv_breakdown, trial_determination, MissingFileError, find_file
from preprocessing import butter_bandpass_filter, resamp, sync, rms_smoothing

initial_data_path = '/Users/clmckeev/Library/CloudStorage/OneDrive-SharedLibraries-IndianaUniversity/[Sec-E] BL-Phys-NDD_EMG - Documents/NDD_EMG/Data/Raw/All'
final_data_path = '/Users/clmckeev/Library/CloudStorage/OneDrive-SharedLibraries-IndianaUniversity/[Sec-E] BL-Phys-NDD_EMG - Documents/NDD_EMG/Data/Final'
folders = ['001', '002', '003','004', '005', '006', '007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '031','032','033','034','035','036','037','038','039'] #This line should be altered as needed
required_files = {'EMG': '*EMG.csv', 'Info': '*info.csv', 'Mat': 'mat*.csv', 'Target': 'target*.csv'}
misses = {}
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
    p_data = pd.read_csv(paths['Info'])
    m_data = pd.read_csv(paths['Mat'])
    mat_data = pd.DataFrame()
    mat_data['Time'] = m_data.iloc[:, 0]
    mat_data['Total'] = m_data.iloc[:, 1:].sum(axis=1) #Only concerned with time of lift and not location of the hand at this time, so we can consider the total of all cells in the pressure mat 
    target_data = pd.read_csv(paths['Target'])

                  
    #Collect all test info in one place
    info['Id'] = subject_id
    info['Date'] = DT.iloc[0,1]
    info['Age'] = p_data['Age']
    info['Race'] = p_data['Race']
    info['Diagnosis'] = p_data['Diagnosis']
    info['Severity'] = p_data['Severity']
    info['Medicated'] = p_data['Medicated']
    info['Family History'] = p_data['Family History']
    info['Brain Injury/Epilepsy'] = p_data['Brain Injury/Epilepsy']
    info['Dominant Hand'] = p_data['Dom Hand']
    info['Hand-eye Coordination'] = p_data['Hand-eye']
    info['Clumsy'] = p_data['Clumsy']
    info['Inhibition of arm movement'] = p_data['Inhibition']
    info['ASD scale'] = p_data['ASD scale']
    info['ADHD scale A'] = p_data['ADHD scale A']
    info['ADHD scale total'] = p_data['ADHD scale total']
    info['Active Medications'] = p_data['Meds']
    info['Relatives'] = p_data['Relatives']
    info['Notes'] = p_data['Notes']
    
    info['Test Length'] = DT.iloc[1,1]
    for measurement in sensor_config.columns:
        info[measurement] = sensor_config.loc[0, measurement]
    info = pd.DataFrame([info])
    info.to_csv(os.path.join(end_subject_dir,'testing_info.csv'), index=False, header=True) #Save info file in individuals folder

    #Seperate all data into desired chunks
    #Standard is EMG_1_1 is Posterior Deltoid EMG_1_2 is Anterior Deltoid and the IMU_1 is on the acromion
    #EMG_2_1 is the IDF EMG_2_2 is Brachioradialus and IMU_2 is on the back of the hand
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

    if not delsys_data.columns.str.contains('Analog_In_1 Time Series \(s\)').any() and not delsys_data.columns.str.contains('Analog_In_1 \(V\)').any():
        print(f"Skipping {subject_id} due to missing Analog data.")
        misses[subject_id] = "Missing Analog data"
        continue 
    Analog = delsys_data[['Analog_In_1 Time Series (s)', 'Analog_In_1 (V)']].dropna()
    Analog.columns = ['Time', 'Analog']
    Analog.to_csv(os.path.join(end_subject_dir,'Analog.csv'), index=False, header=True)


    ## Preprocessing data

     # Sync analog and digital data
    t_delay = sync(Analog, target_data, a_event_cutoff=0.01)
    target_data['hitTime'] = (target_data['hitTime'] / 1000)  + t_delay # Convert to seconds and add delay
    mat_data['Time'] = (mat_data['Time']/1000) + t_delay

     # Determine trial start and stop times and remove unsuccesful trials, clean_EMG only includes data within the start and stop times of each trial
    start_times, stop_times, hit_times = trial_determination(mat_data, target_data)

     ## EMG data
    samp_freqEMG = float(sensor_config.iloc[0, 0].split()[0])

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
    EMG_1 = rsEMG_1[(rsEMG_1['Time'] >= start_times[0]) & (rsEMG_1['Time'] <= stop_times[-1])]
    EMG_2 = rsEMG_2[(rsEMG_2['Time'] >= start_times[0]) & (rsEMG_2['Time'] <= stop_times[-1])]

     # Normalize data using the maximum within task value for each individual muscle
    EMG_1['EMG1'] = EMG_1['EMG1']/(EMG_1['EMG1'].abs().max())
    EMG_1['EMG2'] = EMG_1['EMG2']/(EMG_1['EMG2'].abs().max())
    EMG_2['EMG1'] = EMG_2['EMG1']/(EMG_2['EMG1'].abs().max())
    EMG_2['EMG2'] = EMG_2['EMG2']/(EMG_2['EMG2'].abs().max())

    emg_sanity_check(
        raw_signal=rawEMG_1,
        final_signal=EMG_1,
        start_times=start_times,
        stop_times=stop_times,
        hit_times=target_data['hitTime'],
        save_path=end_subject_dir,
    )

    smooth_EMG_1 = EMG_1.copy()
    smooth_EMG_2 = EMG_2.copy()
    smooth_EMG_1['EMG1'] = rms_smoothing(EMG_1['EMG1'], 30) # window size of 30 samples (30 ms)
    smooth_EMG_1['EMG2'] = rms_smoothing(EMG_1['EMG2'], 30)
    smooth_EMG_2['EMG1'] = rms_smoothing(EMG_2['EMG1'], 30)
    smooth_EMG_2['EMG2'] = rms_smoothing(EMG_2['EMG2'], 30)
   

    ## Analysis

     # change between subsequent maxima and minima within trials
    EMG_11_diff = peak_analysis(start_times, stop_times, smooth_EMG_1[['Time', 'EMG1']], save_path = os.path.join(end_subject_dir,'Graphs','EMG_1_1_extrema'), plot_show=False)
    EMG_12_diff = peak_analysis(start_times, stop_times, smooth_EMG_1[['Time', 'EMG2']], save_path = os.path.join(end_subject_dir,'Graphs','EMG_1_2_extrema'), plot_show=False)
    EMG_21_diff = peak_analysis(start_times, stop_times, smooth_EMG_2[['Time', 'EMG1']], save_path = os.path.join(end_subject_dir,'Graphs','EMG_2_1_extrema'), plot_show=False)
    EMG_22_diff = peak_analysis(start_times, stop_times, smooth_EMG_2[['Time', 'EMG2']], save_path = os.path.join(end_subject_dir,'Graphs','EMG_2_2_extrema'), plot_show=False)

    EMG_11_diff.to_csv(os.path.join(end_subject_dir,'EMG_11_extrema.csv'), index=False, header=True)
    EMG_12_diff.to_csv(os.path.join(end_subject_dir,'EMG_12_extrema.csv'), index=False, header=True)
    EMG_21_diff.to_csv(os.path.join(end_subject_dir,'EMG_21_extrema.csv'), index=False, header=True)
    EMG_22_diff.to_csv(os.path.join(end_subject_dir,'EMG_22_extrema.csv'), index=False, header=True)




print("Processing complete. The following subjects were skipped due to missing files or data:")
for subject_id, reason in misses.items():
    print(f"Subject {subject_id}: {reason}")