import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple
# import your constants or define them here:
# from constants import FS, N_CHANNELS, HP_CUTOFF_FREQ, HP_ORDER, WINDOW_SIZE, STRIDE

# Example defaults (replace with your constants file)
FS = 200
N_CHANNELS = 8
HP_CUTOFF_FREQ = 10.0
HP_ORDER = 4
WINDOW_SIZE = 256
STRIDE = 50

def preprocess_force_rms(
    data_path: str,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    fs: int = FS,
    hp_cutoff: float = HP_CUTOFF_FREQ,
    hp_order: int = HP_ORDER,
    force_scale_to_percent: bool = True,
    force_max_value: float | None = None,
    target_reduce: str = "mean"   # options: "mean", "median", "center", "trimmed_mean"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess EMG+force CSV for regression:
    - Filters EMG channels
    - Sliding windows
    - Extract RMS feature per channel for each window
    - Compute window-level force target (mean/median/center/trimmed_mean)

    CSV expected layout:
      timestamp, emg1, emg2, ..., emg8, force
    (timestamp optional; we ignore it)
    Returns:
      X: (N_windows, N_CHANNELS)    -- RMS per window per channel
      y: (N_windows,)               -- continuous force target (float)
    """
    df = pd.read_csv(data_path)
    # keep only EMG columns (next N_CHANNELS cols) and force last column
    emg = df.iloc[:, 1:1+N_CHANNELS].values.astype(np.float64)
    force = df.iloc[:, -1].values.astype(np.float64)

    # design high-pass filter
    b_hp, a_hp = signal.butter(hp_order, hp_cutoff, btype='highpass', fs=fs)

    # (optionally) detrend or remove DC here if you already do that elsewhere
    # emg = signal.detrend(emg, axis=0, type='constant')
    # emg = emg - emg.mean(axis=0, keepdims=True)

    # filter EMG (zero-phase)
    emg_f = signal.filtfilt(b_hp, a_hp, emg, axis=0)

    X_list = []
    y_list = []

    n_samples = emg_f.shape[0]
    for start in range(0, n_samples - window_size + 1, stride):
        win_emg = emg_f[start : start + window_size, :]   # shape (window_size, N_CHANNELS)
        win_force = force[start : start + window_size]    # shape (window_size,)

        # feature: RMS per channel
        rms = np.sqrt(np.mean(win_emg**2, axis=0))        # shape (N_CHANNELS,)

        # choose force target
        if target_reduce == "mean":
            target = float(np.mean(win_force))
        elif target_reduce == "median":
            target = float(np.median(win_force))
        elif target_reduce == "center":
            center_idx = start + window_size // 2
            target = float(force[center_idx])
        elif target_reduce == "trimmed_mean":
            # drop top/bottom 10% then mean
            sorted_vals = np.sort(win_force)
            k = int(round(0.1 * len(sorted_vals)))
            if len(sorted_vals) - 2*k > 0:
                target = float(sorted_vals[k:-k].mean())
            else:
                target = float(sorted_vals.mean())
        else:
            raise ValueError("target_reduce must be one of ['mean','median','center','trimmed_mean']")

        X_list.append(rms)
        y_list.append(target)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    # optional: scale force to percentage (0-100) or 0-1 using provided max
    if force_scale_to_percent:
        if force_max_value is None:
            # infer from data (may clip later for real collection)
            force_max_value = max(force.max(), 1e-6)
        y = (y / force_max_value) * 100.0

    return X, y
