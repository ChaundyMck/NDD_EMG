#Will take input folders and output analyzed data
import pandas as pd
import glob
import os
from extract import delsys_csv_breakdown
from preprocessing import butter_bandpass_filter, resamp

initial_data_path = '/Users/clmckeev/Library/CloudStorage/OneDrive-IndianaUniversity/Documents/NDD_EMG/Data/Raw/All'
final_data_path = '/Users/clmckeev/Library/CloudStorage/OneDrive-IndianaUniversity/Documents/NDD_EMG/Data/Final'
folders = ['001','002'] #This line should be altered as needed


for subject_id in folders: #Collect recorded data of each participant
    info = []
    (DT, sensor_config, delsys_data) = delsys_csv_breakdown(glob.glob(os.path.join(initial_data_path,subject_id,'*emg.csv'))[0])
    delsys_data.dropna()
    p_data = pd.read_csv(glob.glob(os.path.join(initial_data_path,subject_id,'*info.csv'))[0])
    mat_data = pd.read_csv(glob.glob(os.path.join(initial_data_path,subject_id,'mat*.csv'))[0])
    target_data = pd.read_csv(glob.glob(os.path.join(initial_data_path,subject_id,'target*.csv'))[0])

                  
    #Collect all test info in one place
    info['Id'] = subject_id
    info['Date'] = DT.iloc[0,1]
    info['Age'] = p_data.loc['Age', 0]
    info['Race'] = p_data.loc['Race', 0]
    info['Diagnosis'] = p_data.loc['Diagnosis', 0]
    info['Severity'] = p_data.loc['Severity', 0]
    info['Medicated'] = p_data.loc['Medicated', 0]
    info['Family History'] = p_data.loc['Family History', 0]
    info['Brain Injury/Epilepsy'] = p_data.loc['Brain Injury/Epilepsy', 0]
    info['Dominant Hand'] = p_data.loc['Dom Hand', 0]
    info['Hand-eye Coordination'] = p_data.loc['Hand-eye', 0]
    info['History of hobbies requiring hand eye corrdination'] = p_data.loc['History Hand-eye', 0]
    info['Clumsy'] = p_data.loc['Clumsy', 0]
    info['Inhibition of arm movement'] = p_data.loc['Inhibition', 0]
    info['ASD scale'] = p_data.loc['ASD scale', 0]
    info['ADHD scale A'] = p_data.loc['ADHD scale A', 0]
    info['ADHD scale total'] = p_data.loc['ADHD scale total', 0]
    info['Notes'] = p_data.loc['Notes', 0]
    
    info['Test Length'] = DT.iloc[1,1]
    for measurement in sensor_config.columns:
        info[measurement] = sensor_config.loc[0, measurement]
    pd.DataFrame(info)
    info.to_csv(os.path.join(final_data_path,subject_id,'testing_info.csv'), index=False, header=True) #Save info file in individuals folder

    #Seperate all data into desired chunks
    #Standard is EMG_1_1 is Posterior Deltoid EMG_1_2 is Anterior Deltoid and the IMU_1 is on the acromion
    #EMG_2_1 is the Pronator Teres EMG_2_2 is Brachioradialus and IMU_2 is on the back of the hand
    rawEMG_1 = delsys_data[['EMG 1 Time Series (s)', 'EMG 1 (mV)', 'EMG 2 (mV)']].dropna()
    rawEMG_1.columns = ['Time', 'EMG1', 'EMG2']
    rawEMG_1.to_csv(os.path.join(final_data_path,subject_id,'EMG_1_raw.csv'), index=False, header=True) 
    rawAcc_1 = delsys_data[['ACC X Time Series (s)', 'ACC X (G)', 'ACC Y (G)', 'ACC Z (G)']].dropna()
    rawAcc_1.columns = ['Time', 'X', 'Y', 'Z']
    rawAcc_1.to_csv(os.path.join(final_data_path,subject_id,'Acc_1_raw.csv'), index=False, header=True) 
    rawGyro_1 = delsys_data[['GYRO X Time Series (s)', 'GYRO X (deg/s)', 'GYRO Y (deg/s)', 'GYRO Z (deg/s)']].dropna()
    rawGyro_1.columns = ['Time', 'X', 'Y', 'Z']
    rawGyro_1.to_csv(os.path.join(final_data_path,subject_id,'Gyro_1_raw.csv'), index=False, header=True) 

    rawEMG_2 = delsys_data[['EMG 1 Time Series (s) 2', 'EMG 1 (mV) 2', 'EMG 2 (mV) 2']].dropna()
    rawEMG_2.columns = ['Time', 'EMG1', 'EMG2']
    rawEMG_2.to_csv(os.path.join(final_data_path,subject_id,'EMG_2_raw.csv'), index=False, header=True) 
    rawAcc_2 = delsys_data[['ACC X Time Series (s) 2', 'ACC X (G) 2', 'ACC Y (G) 2', 'ACC Z (G) 2']].dropna()
    rawAcc_2.columns = ['Time', 'X', 'Y', 'Z']
    rawAcc_2.to_csv(os.path.join(final_data_path,subject_id,'Acc_2_raw.csv'), index=False, header=True) 
    rawGyro_2 = delsys_data[['GYRO X Time Series (s) 2', 'GYRO X (deg/s) 2', 'GYRO Y (deg/s) 2', 'GYRO Z (deg/s) 2']].dropna()
    rawGyro_2.columns = ['Time', 'X', 'Y', 'Z']
    rawGyro_2.to_csv(os.path.join(final_data_path,subject_id,'Gyro_2_raw.csv'), index=False, header=True)

    Analog = delsys_data[ 'Analog_In_1 Time Series (s)',' Analog_In_1 (V)']
    Analog.columns = ['Time', 'Analog']
    Analog.to_csv(os.path.join(final_data_path,subject_id,'Analog.csv'), index=False, header=True)


    ## Preprocessing data
     # EMG data
    samp_freqEMG = float(sensor_config.iloc[0, 0].split()[0])

     # Resample to 1000 Hz
    rsEMG_1 = resamp(rawEMG_1)
    rsEMG_2 = resamp(rawEMG_2)

     # Apply a 4th order Butterworth bandpass (20-450Hz) using the sampling frequency from Delsys
    filtEMG_1 = []
    filtEMG_2 = []
    filtEMG_1['Time'] = rsEMG_1['Time']
    filtEMG_2['Time'] = rsEMG_2['Time']
    filtEMG_1['EMG1'] = butter_bandpass_filter(rsEMG_1['EMG1'],20, 450, samp_freqEMG, order = 4)
    filtEMG_1['EMG2'] = butter_bandpass_filter(rsEMG_1['EMG2'],20, 450, samp_freqEMG, order = 4)
    filtEMG_2['EMG1'] = butter_bandpass_filter(rsEMG_2['EMG1'],20, 450, samp_freqEMG, order = 4)
    filtEMG_2['EMG2'] = butter_bandpass_filter(rsEMG_2['EMG2'],20, 450, samp_freqEMG, order = 4)
    
     # Determine trial start and stop times and remove unsuccesful trials



     # Normalize data using the maximum within trial measurement
    EMG_1 = cleanEMG_1
    EMG_2 = cleanEMG_2
    EMG_1['EMG1'] = (EMG_1['EMG1']/EMG_1['EMG1'].max())
    EMG_1['EMG2'] = (EMG_1['EMG2']/EMG_1['EMG2'].max())
    EMG_2['EMG1'] = (EMG_2['EMG1']/EMG_2['EMG1'].max())
    EMG_2['EMG2'] = (EMG_2['EMG2']/EMG_2['EMG2'].max())
    
    


