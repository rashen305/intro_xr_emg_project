# This file implements the EMG listener and realtime ML inference portions of the pipeline
# MODIFIED for Support Vector Classification (SVC) using RMS features.
# It now includes the required high-pass filtering step.

import socket
import threading
import json
import time
import collections
import numpy as np
import pickle
# New required imports for filtering
from sklearn.preprocessing import scale 
from scipy import signal 
from constants import FS, N_CHANNELS, HP_CUTOFF_FREQ, HP_ORDER, N_CHANNELS, WINDOW_SIZE

# --- Configuration ---
HOST = '127.0.0.1'  # Must match the C++ sender's host
PORT = 9002         # Must match the C++ sender's port
BUFFER_SIZE = 1024  # Total number of samples to store
FEATURE_WINDOW = WINDOW_SIZE # The window size for RMS calculation

# The deque will hold tuples: (timestamp, [emg_channel_1, ..., emg_channel_8])
emg_buffer = collections.deque(maxlen=BUFFER_SIZE)

# Lock for safe access to the shared emg_buffer
buffer_lock = threading.Lock()

# Flag to control the main loops
stop_event = threading.Event()

# --------------------------------------------------------------------------
# --- Model Loading and Feature Extraction Setup ---
# --------------------------------------------------------------------------

# 1. Design the High-Pass filter coefficients once globally
b_hp, a_hp = signal.butter(HP_ORDER, HP_CUTOFF_FREQ, btype='highpass', fs=FS)

def load_svc_model(filepath="svc_model.pkl"):
    """Loads the trained SVC model."""
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"🧠 Model: Successfully loaded SVC model from {filepath}")
        return model
    except FileNotFoundError:
        print(f"❌ Model: ERROR: Model file not found at {filepath}. Please train and save the model first.")
        stop_event.set()
        return None
    except Exception as e:
        print(f"❌ Model: Error loading model: {e}")
        stop_event.set()
        return None

# Load the model once when the script starts
SVC_MODEL = load_svc_model()

def extract_rms_features(data_window: np.ndarray) -> np.ndarray:
    """
    Applies high-pass filtering, calculates the Root Mean Square (RMS) feature 
    for each channel, and scales the result.
    
    Input: data_window (np.ndarray) of shape (FEATURE_WINDOW, 8)
    Output: feature_vector (np.ndarray) of shape (1, 8)
    """
    
    # 1. Apply High-Pass Filter (Zero-Phase Lag)
    # NOTE: Using filtfilt on small streaming windows is an approximation of 
    # the training preprocessing, and is technically non-causal. A dedicated 
    # stateful IIR filter (signal.lfilter) is better for strict real-time, 
    # but this matches the filtfilt step in your training script.
    emg_filtered = signal.filtfilt(b_hp, a_hp, data_window, axis=0)

    # 2. Calculate RMS: sqrt(mean(x^2)) for each of the 8 channels (axis=0)
    # Use the filtered data for RMS calculation
    rms_features = np.sqrt(np.mean(np.square(emg_filtered), axis=0))
    
    # 3. Reshape to (1, 8) to match the expected scikit-learn input for prediction
    feature_vector = rms_features.reshape(1, -1)
    
    # 4. Apply scaling/normalization
    # This scaling must match the scaling used during training (e.g., StandardScaler).
    feature_vector_scaled = scale(feature_vector, axis=1) 

    return feature_vector_scaled

# --------------------------------------------------------------------------
# --- Actual ML Inference Function ---
# --------------------------------------------------------------------------

def actual_inference_caller(data_window: np.ndarray):
    """
    Performs RMS feature extraction and then SVC classification.
    
    The input `data_window` is a NumPy array of raw EMG samples 
    of shape (FEATURE_WINDOW, 8).
    """
    if SVC_MODEL is None:
        return -1, np.zeros(8) # Return a safe prediction if model failed to load
        
    # 1. Feature Extraction (RMS + Filtering)
    feature_vector = extract_rms_features(data_window)
    
    # 2. Run SVC Classification
    prediction = SVC_MODEL.predict(feature_vector)[0]
    
    # 3. Calculate details for logging (e.g., mean absolute value of the raw data)
    mean_abs_emg = np.mean(np.abs(data_window), axis=0)
    
    return prediction, mean_abs_emg

# --------------------------------------------------------------------------
# --- Thread 1: Data Listener and Buffer Manager (Unchanged) ---
# --------------------------------------------------------------------------

def data_listener_thread():
    """Listens for TCP connection and receives streaming EMG data."""
    print(f"📡 Listener: Starting TCP server on {HOST}:{PORT}")
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            
            print("📡 Listener: Waiting for connection...")
            conn, addr = s.accept()
            print(f"📡 Listener: Connection established with {addr}")
            
            with conn:
                json_buffer = "" 
                
                while not stop_event.is_set():
                    data = conn.recv(64).decode('utf-8') 
                    
                    if not data:
                        break 
                        
                    json_buffer += data 
                    
                    while '}\n' in json_buffer:
                        end_index = json_buffer.find('}\n') + 1 
                        
                        if end_index == 0:
                            break
                            
                        json_str = json_buffer[:end_index]
                        json_buffer = json_buffer[end_index:].lstrip()
                        
                        try:
                            sample_data = json.loads(json_str)
                            
                            timestamp = sample_data.get('t', time.time())
                            emg_values = sample_data.get('emg', None)
                            
                            if emg_values and len(emg_values) == N_CHANNELS:
                                with buffer_lock:
                                    emg_buffer.append((timestamp, emg_values))
                                    
                        except json.JSONDecodeError as e:
                            print(f"❌ Listener: JSON Decode Error: {e}. Buffer content: '{json_str[:50]}...'")
                            next_start = json_buffer.find('{')
                            if next_start != -1:
                                json_buffer = json_buffer[next_start:]
                            else:
                                json_buffer = ""
                            break
                            
    except ConnectionRefusedError:
        print("❌ Listener: Connection refused. Is the C++ sender running?")
    except Exception as e:
        if not stop_event.is_set():
             print(f"❌ Listener: An unexpected error occurred: {e}")
    finally:
        print("📡 Listener: Thread stopped.")
        stop_event.set() 

# --------------------------------------------------------------------------
# --- Thread 2: ML Inference Worker (Unchanged logic) ---
# --------------------------------------------------------------------------

def inference_worker_thread():
    """Continuously checks the buffer and performs ML inference."""
    print(f"🧠 Worker: Starting SVC inference thread. Window size: {FEATURE_WINDOW} samples.") 
    
    if SVC_MODEL is None:
        print("🧠 Worker: Model failed to load. Stopping worker thread.")
        return

    while not stop_event.is_set():
        current_buffer_size = 0
        with buffer_lock:
            current_buffer_size = len(emg_buffer)
            
        if current_buffer_size >= FEATURE_WINDOW:
            
            recent_data = None
            with buffer_lock:
                recent_data = list(emg_buffer)[-FEATURE_WINDOW:] 
            
            if recent_data is None:
                continue
                
            data_window = np.array([item[1] for item in recent_data], dtype=np.float64)
            latest_timestamp = recent_data[-1][0]
            
            try:
                prediction, details = actual_inference_caller(data_window)
                
                print(f"\r[t={latest_timestamp:.3f}s | "
                      f"Prediction: **{prediction}** | "
                      f"RMS: {np.round(details, 2)}]", end='', flush=True)
                
            except Exception as e:
                print(f"\n❌ Worker: Error during inference: {e}")
                
        else:
            time.sleep(0.01)
            
    print("\n🧠 Worker: Thread stopped.")

# --------------------------------------------------------------------------
# --- Main Execution (Unchanged) ---
# --------------------------------------------------------------------------

def main():
    """Starts the two threads and handles graceful shutdown."""
    
    if SVC_MODEL is None and stop_event.is_set():
        print("🛑 Main: Cannot run inference without a loaded SVC model. Exiting.")
        return

    listener_thread = threading.Thread(target=data_listener_thread)
    worker_thread = threading.Thread(target=inference_worker_thread)
    
    listener_thread.start()
    worker_thread.start()
    
    try:
        while not stop_event.is_set():
            time.sleep(1) 
            
    except KeyboardInterrupt:
        print("\n🛑 Main: Shutdown signal received (Ctrl+C).")
        
    finally:
        stop_event.set()
        print("🛑 Main: Waiting for threads to terminate...")
        
        listener_thread.join()
        worker_thread.join()
        
        print("🎉 Main: All threads terminated. Program finished.")

if __name__ == '__main__':
    main()