import numpy as np
import pandas as pd
from scipy import signal
from constants import FS, N_CHANNELS, HP_CUTOFF_FREQ, HP_ORDER, WINDOW_SIZE, STRIDE

def preprocess_rms(data_path: str):
    """
    Loads EMG CSV, splits by label, concatenates all samples of each class,
    filters, window-slices each class separately, and extracts RMS features.

    Returns:
        X: shape (N_windows, N_channels)
        Y: shape (N_windows,)
    """
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"File not found: {data_path}")
        return np.array([]), np.array([])
        
    # Extract EMG (8 ch) + labels
    emg = df.iloc[:, 1:1+N_CHANNELS].values 
    labels = df.iloc[:, -1].values

    #TODO
    # Currently hardcode for 2 classes

    # ---- 1) SPLIT BY LABEL ----
    class0 = emg[labels == 0]
    class1 = emg[labels == 1]

    # ---- 2) CONCATENATE EACH CLASS INTO ONE CONTINUOUS SIGNAL ----
    emg_by_class = {
        0: class0,  # rest
        1: class1,  # clinched
    }

    # ---- 3) High-pass filter design ----
    b, a = signal.butter(HP_ORDER, HP_CUTOFF_FREQ, btype='highpass', fs=FS)

    X_all = []
    Y_all = []

    # ---- 4) Process each class independently ----
    for label_value, emg_signal in emg_by_class.items():

        if len(emg_signal) < WINDOW_SIZE:
            continue  # too short to window

        # Filter each concatenated class signal
        emg_filtered = signal.filtfilt(b, a, emg_signal, axis=0)
        n_samples = emg_filtered.shape[0]

        # ---- 5) Sliding window ----
        for i in range(0, n_samples - WINDOW_SIZE + 1, STRIDE):
            window = emg_filtered[i : i + WINDOW_SIZE, :]

            # RMS per channel
            rms_feat = np.sqrt(np.mean(window ** 2, axis=0))

            X_all.append(rms_feat)
            Y_all.append(label_value)

    return np.array(X_all), np.array(Y_all)
