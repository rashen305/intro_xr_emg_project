import numpy as np
import torch
import torch.nn as nn
from scipy.signal import butter, filtfilt, iirnotch, stft, detrend

# ===========================
# 1. Config (Must match train_200.py)
# ===========================
FS = 200                      # Sampling rate
WINDOW_SIZE = 256             # Window size for the live buffer
NPERSEG = 128                 # STFT NPERSEG
NOVERLAP = 64                 # STFT NOVERLAP
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["rest", "pinch"] # Class names for output

# Global variable to hold the loaded model and normalization parameters
_MODEL = None
_MEAN = None
_STD = None

# ===========================
# 2. CNN Model Architecture (Copied from train_200.py)
# ===========================
class CNNmodel(nn.Module):
    def __init__(self, in_channels=8, num_classes=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, (5,3), padding=(2,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d((2,1))

        self.conv2 = nn.Conv2d(32, 64, (3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d((2,1))

        self.conv3 = nn.Conv2d(64, 128, (3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(128, 64)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.global_pool(self.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

# ===========================
# 3. Preprocessing Steps (Adapted for a single window)
# ===========================

# Pre-calculate filter coefficients to avoid re-calculating on every call
# Notch 60 Hz
B_NOTCH, A_NOTCH = iirnotch(60.0 / (FS / 2), 30)
# Bandpass 20–90 Hz
B_BAND, A_BAND = butter(4, [20/(FS/2), 90/(FS/2)], btype='band')

def preprocess_window(window: np.ndarray) -> np.ndarray:
    """
    Applies the full preprocessing pipeline to a single (WINDOW_SIZE, 8) EMG window.
    
    Args:
        window: A NumPy array of shape (256, 8) containing the raw EMG samples.
        
    Returns:
        A NumPy array of shape (8, num_freq_bins, num_time_steps) representing 
        the STFT features, ready for normalization and model input.
    """
    
    # 1. Detrend + DC removal
    data = detrend(window, axis=0, type='constant')
    data = data - np.mean(data, axis=0, keepdims=True)

    # 2. Notch 60 Hz
    data = filtfilt(B_NOTCH, A_NOTCH, data, axis=0)

    # 3. Bandpass 20–90 Hz
    data = filtfilt(B_BAND, A_BAND, data, axis=0)

    # 4. STFT per channel
    specs = []
    for ch in range(data.shape[1]):
        # The input data is already a single window
        f, t, Zxx = stft(
            data[:, ch], fs=FS,
            nperseg=NPERSEG,
            noverlap=NOVERLAP,
            boundary=None
        )
        specs.append(np.abs(Zxx))
    
    # Shape is (channels, freq_bins, time_steps)
    return np.array(specs).astype(np.float32)


# ===========================
# 4. Model & Normalization Loading
# ===========================

def load_model_and_params(model_path: str, normalization_path: str):
    """
    Loads the model weights and normalization parameters (mean/std) once.
    
    NOTE: You MUST save the mean and std from your training script 
          (e.g., in a separate .npy file) for this to work correctly.
    """
    global _MODEL, _MEAN, _STD
    
    # Load Model Architecture and Weights
    if _MODEL is None:
        try:
            model = CNNmodel().to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
            _MODEL = model
            print(f"🧠 Model: Loaded weights from '{model_path}' and set to {DEVICE}.")
        except Exception as e:
            print(f"❌ Model: Failed to load model weights from {model_path}. Error: {e}")
            raise
    
    # Load Normalization Parameters (Mean and Std)
    if _MEAN is None or _STD is None:
        try:
            # Assume normalization_path points to a file containing a dictionary 
            # or array with mean and std saved during training.
            # !!! THIS PART REQUIRES YOU TO SAVE MEAN/STD IN TRAIN_200.PY !!!
            
            # --- Placeholder for loading normalization parameters ---
            # You must run train_200.py once, print the 'mean' and 'std' values,
            # and manually set them here, or save them to a file.
            
            # Example if you manually set them (replace with your actual values)
            # These were derived by running train_200.py once and printing them.
            _MEAN = np.array([[[[3.6599715, ..., 3.6599715]]]]).astype(np.float32) 
            _STD = np.array([[[[0.7258384, ..., 0.7258384]]]]).astype(np.float32)

            # NOTE: The shapes must match the saved mean/std from training
            
            # If you saved a numpy file:
            # params = np.load(normalization_path, allow_pickle=True).item()
            # _MEAN = params['mean']
            # _STD = params['std']
            # --- End Placeholder ---
            
            if _MEAN is None or _STD is None:
                 print("⚠️ Model: Normalization parameters (mean/std) are set to placeholder values. Update them from your training script!")
                 # Set a neutral placeholder if actual values aren't loaded/set
                 # This will result in poor performance unless corrected.
                 # We'll set these to None and handle normalization dynamically in run_inference
                 pass

        except Exception as e:
            print(f"❌ Model: Failed to load normalization parameters. Error: {e}")
            raise


# ===========================
# 5. Dedicated Inference Function
# ===========================

def run_inference(emg_window: np.ndarray) -> str:
    """
    Runs the full inference pipeline (preprocess -> normalize -> predict).
    
    This function will replace the 'placeholder_inference' in your receiver script.
    
    Args:
        emg_window: A NumPy array of shape (256, 8) containing the latest EMG samples.
        
    Returns:
        The predicted class name (e.g., "rest", "pinch").
    """
    global _MODEL, _MEAN, _STD
    
    if _MODEL is None:
        # Load model with placeholder paths if not already loaded
        # YOU MUST REPLACE THE PATHS BELOW with your saved files
        model_file = "train_single_subject_myo_model.pth"
        norm_file = "normalization_params.npy" # Placeholder
        load_model_and_params(model_file, norm_file)
        
    # Ensure the model is loaded after the first attempt
    if _MODEL is None:
        return "ERROR: Model not loaded."


    # 1. Preprocessing (Detrend, Filter, STFT)
    # Output shape: (8, F, T) -> (channels, freq, time)
    X_spec = preprocess_window(emg_window)
    
    # 2. Log + Normalization
    X = np.log1p(X_spec)
    # The normalization parameters must have been calculated over the entire
    # training set for ALL axes (0, 2, 3), but applied per channel.
    # Handle case where normalization params aren't loaded yet
    if _MEAN is None or _STD is None:
        # Fallback: use zero mean and unit std if normalization not loaded
        # This ensures the code doesn't crash, but performance will be poor
        # until proper normalization parameters are set
        _MEAN = np.zeros_like(X)
        _STD = np.ones_like(X)
        print("⚠️ Warning: Using fallback normalization (zero mean, unit std). Model performance may be poor.")
    
    # Ensure shapes are compatible for broadcasting
    # _MEAN and _STD should be shape (8, F, T) or broadcastable to it
    X = (X - _MEAN) / _STD
    
    # 3. Prepare for PyTorch model (Add batch dimension)
    # Input shape to model must be (1, C, F, T) -> (1, 8, F, T)
    X_tensor = torch.from_numpy(X[np.newaxis, ...]).to(DEVICE)
    
    # 4. Inference
    with torch.no_grad():
        output = _MODEL(X_tensor)
        
    # 5. Get Prediction
    # The output is a tensor like [logit_rest, logit_pinch]
    prediction_idx = output.argmax(1).item()
    
    return CLASS_NAMES[prediction_idx]


# ===========================
# 6. Example Usage (Self-Test)
# ===========================

if __name__ == '__main__':
    print("--- Running Self-Test ---")
    
    # 1. Create a dummy rest signal
    dummy_rest = np.zeros((WINDOW_SIZE, 8), dtype=np.float64)
    # 2. Create a dummy pinch signal (simulating activity on two channels)
    dummy_pinch = np.zeros((WINDOW_SIZE, 8), dtype=np.float64)
    dummy_pinch[:, 0] = 50 * np.sin(np.linspace(0, 10 * np.pi, WINDOW_SIZE))
    dummy_pinch[:, 1] = 40 * np.sin(np.linspace(0, 8 * np.pi, WINDOW_SIZE))
    
    # NOTE: You must first run train_200.py to generate 'train_single_subject_myo_model.pth'

    try:
        # Test 1: Rest
        prediction_rest = run_inference(dummy_rest)
        print(f"\n✅ Test 1 (Dummy Rest): Predicted class: {prediction_rest}")
        
        # Test 2: Pinch
        prediction_pinch = run_inference(dummy_pinch)
        print(f"\n✅ Test 2 (Dummy Pinch): Predicted class: {prediction_pinch}")

    except Exception as e:
        print(f"\n❌ Self-Test FAILED. Ensure model weights and paths are correct. Error: {e}")