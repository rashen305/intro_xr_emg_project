import numpy as np
import pandas as pd
from scipy import signal
from constants import FS, N_CHANNELS, HP_CUTOFF_FREQ, HP_ORDER, WINDOW_SIZE, STRIDE

def preprocess_rms(data_path: str) -> np.ndarray:
    """
    Loads raw EMG data, high-pass filters it, segments it into windows,
    and extracts the Root Mean Square (RMS) feature for each channel.
    
    Args:
        data_path (str): Path to the raw EMG CSV file.
        
    Returns: A NumPy array of shape (N_windows, N_channels)
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return np.array([])
        
    # EMG columns look like this:
    # timestamp,sample_number,emg1,emg2,emg3,emg4,emg5,emg6,emg7,emg8
    emg_data = df.iloc[:, 2:2+N_CHANNELS].values 

    # 1. Design the 4th order 10Hz High-Pass filter
    # Note: HP_ORDER, HP_CUTOFF_FREQ, FS, and N_CHANNELS are still global constants
    b, a = signal.butter(HP_ORDER, HP_CUTOFF_FREQ, btype='highpass', fs=FS)
    
    # 2. Apply the filter using filtfilt (zero-phase lag)
    emg_filtered = signal.filtfilt(b, a, emg_data, axis=0)

    # 3. Segment and Extract RMS Feature
    N_samples = emg_filtered.shape[0]
    windows = []
    
    # Loop over the data, using the provided window_size and stride
    # Note: We use window_size in the loop condition and stride for the step
    for i in range(0, N_samples - WINDOW_SIZE + 1, STRIDE):
        window = emg_filtered[i:i + WINDOW_SIZE, :]
        
        # Calculate RMS for each of the 8 channels in the window
        rms_features = np.sqrt(np.mean(window**2, axis=0)) 
        windows.append(rms_features)

    return np.array(windows)