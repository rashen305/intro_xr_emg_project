import os

# These global constants are used across multiple modules in the ML pipeline.

# --- Hardware/Signal Acquisition Constants ---
FS = 200            # Sampling Frequency (Hz)
N_CHANNELS = 8      # Number of EMG channels

# --- Feature Extraction Parameters ---
WINDOW_MS = 200     # Window size in milliseconds
OVERLAP = 0.5       # 50% overlap

# Calculated Constants (Derived from the above)
WINDOW_SIZE = int(FS * WINDOW_MS / 1000) # 40 samples at 200 Hz
STRIDE = int(WINDOW_SIZE * (1 - OVERLAP)) # 20 samples

# --- Filter Parameters ---
HP_CUTOFF_FREQ = 10.0 # 10 Hz high-pass filter cutoff
HP_ORDER = 4        # 4th order high-pass filter

# --- Model/Training Constants ---
TEST_SIZE = 0.2     # 20% of data for testing
RANDOM_SEED = 42