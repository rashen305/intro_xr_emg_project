import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, stft, detrend
import torch
from torch.utils.data import Dataset

# ===========================
# 1. Configuration Constants
# ===========================
FS = 200                      # Sampling rate
WINDOW_SIZE = 256             # Window size for the sliding window
STRIDE = 50                   # Stride for windowing (step between windows)
NPERSEG = 128                 # STFT NPERSEG
NOVERLAP = 64                 # STFT NOVERLAP
N_CHANNELS = 8                # Number of EMG channels

# ===========================
# 2. Filter Coefficients
# ===========================
B_NOTCH, A_NOTCH = iirnotch(60.0 / (FS / 2), 30)
B_BAND, A_BAND = butter(4, [20/(FS/2), 90/(FS/2)], btype='band')

# ===========================
# 3. Preprocessing Function
# ===========================

def preprocess(path):
    """
    Reads raw EMG data from a CSV path, applies filtering and STFT, 
    and returns a stack of log-spectrogram windows.
    """
    try:
        # Load raw data. Set header=0 to treat the first row as column names.
        # This allows us to skip it when reading the values.
        df = pd.read_csv(path, header=0) 
        
        # Select ONLY the 8 EMG channel columns (columns 1 through 8, 
        # skipping the Timestamp_ms column at index 0)
        # and then convert to a NumPy array.
        # This is the FIX for the ValueError.
        raw_data = df.iloc[:, 1:].values.astype(np.float64) 
        
    except FileNotFoundError:
        print(f"File not found: {path}")
        return np.array([])
    except Exception as e:
        print(f"Error processing file {path}: {e}")
        return np.array([])

    # 1. Detrend and DC removal
    data = detrend(raw_data, axis=0, type='constant')
    data = data - np.mean(data, axis=0, keepdims=True)

    # 2. Notch 60 Hz
    data = filtfilt(B_NOTCH, A_NOTCH, data, axis=0)

    # 3. Bandpass 20–90 Hz
    data = filtfilt(B_BAND, A_BAND, data, axis=0)

    all_specs = []
    N_samples = data.shape[0]
    
    # 4. Sliding Window and STFT
    for i in range(0, N_samples - WINDOW_SIZE + 1, STRIDE):
        window = data[i:i + WINDOW_SIZE, :]
        
        specs = []
        for ch in range(N_CHANNELS):
            f, t, Zxx = stft(
                window[:, ch], fs=FS,
                nperseg=NPERSEG,
                noverlap=NOVERLAP,
                boundary=None
            )
            specs.append(np.abs(Zxx))
        
        all_specs.append(np.array(specs).astype(np.float32))

    if not all_specs:
        return np.array([])
        
    # Stack and convert to Log-Spectrogram
    X = np.stack(all_specs) 
    X = np.log1p(X) 
    
    return X

# ===========================
# 4. PyTorch Dataset Class
# ===========================

class EMGDataset(Dataset):
    """Custom PyTorch Dataset for EMG Spectrograms."""
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).long() 

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]