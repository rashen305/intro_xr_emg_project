import numpy as np
import pandas as pd
from scipy import signal
from constants import FS, N_CHANNELS, HP_CUTOFF_FREQ, HP_ORDER, WINDOW_SIZE, STRIDE

def preprocess_rms(data_path: str) -> np.ndarray | np.ndarray:
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

    # TODO: Currently hardcoded for 2 classes

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
        print(f"Processing label {label_value} with {emg_signal.shape[0]} samples.")

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


def preprocess_rms_realtime(data_window: np.ndarray) -> np.ndarray:
    """
    Applies high-pass filtering and calculates the Root Mean Square (RMS) feature.
    
    The scaling step is intentionally omitted here as the loaded SVC_MODEL 
    is a Scikit-learn pipeline that includes a StandardScaler.
    
    Input: data_window (np.ndarray) of shape (WINDOW_SIZE, 8)
    Output: feature_vector (np.ndarray) of shape (1, 8)
    """
    # b and a are the numerator and denominator coefficients of the filter
    b_hp, a_hp = signal.butter(HP_ORDER, HP_CUTOFF_FREQ, btype='highpass', fs=FS)

    # 1. Apply High-Pass Filter (Zero-Phase Lag)
    # This must match the filtering used during training.
    emg_filtered = signal.filtfilt(b_hp, a_hp, data_window, axis=0)

    # 2. Calculate RMS: sqrt(mean(x^2)) for each of the 8 channels (axis=0)
    rms_features = np.sqrt(np.mean(np.square(emg_filtered), axis=0))
    
    # 3. Reshape to (1, 8) for scikit-learn's prediction input (1 sample, 8 features)
    feature_vector = rms_features.reshape(1, -1)
    
    # 4. NOTE: No external scaling is performed here. The loaded SVC_MODEL pipeline 
    # will apply the required Standard Scaling using the parameters learned during training.

    return feature_vector