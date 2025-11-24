# This file implements the EMG listener and realtime ML inference portions of the pipeline
# It uses a trained SVC model, applying RMS features and required high-pass filtering.

import socket
import threading
import json
import time
import collections
import numpy as np
import pickle
from scipy import signal 
from constants import FS, N_CHANNELS, HP_CUTOFF_FREQ, HP_ORDER, N_CHANNELS, WINDOW_SIZE
from feature_extraction import rms

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
# b and a are the numerator and denominator coefficients of the filter
b_hp, a_hp = signal.butter(HP_ORDER, HP_CUTOFF_FREQ, btype='highpass', fs=FS)

def load_svc_model(filepath="svc_model.pkl"):
    """Loads the trained SVC model pipeline."""
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"🧠 Model: Successfully loaded SVC pipeline from {filepath}")
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

# --------------------------------------------------------------------------
# --- Actual ML Inference Function ---
# --------------------------------------------------------------------------

def actual_inference_caller(data_window: np.ndarray):
    """
    Performs RMS feature extraction and then SVC classification.
    """
    if SVC_MODEL is None:
        return -1, np.zeros(N_CHANNELS)
        
    # 1. Feature Extraction (Filtering + RMS)
    feature_vector = rms.preprocess_rms_realtime(data_window)
    
    # 2. Run SVC Classification (Pipeline handles internal scaling)
    prediction = SVC_MODEL.predict(feature_vector)[0]
    
    # 3. Calculate details for logging (e.g., mean absolute value of the raw data)
    mean_abs_emg = np.mean(np.abs(data_window), axis=0)
    
    return prediction, mean_abs_emg

# --------------------------------------------------------------------------
# --- Thread 1: Data Listener and Buffer Manager ---
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
                    # Receive small chunks of data
                    data = conn.recv(64).decode('utf-8') 
                    
                    if not data:
                        break 
                        
                    json_buffer += data 
                    
                    # Process completed JSON objects
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
                            # Attempt to recover by dropping up to the next potential JSON start
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
        stop_event.set() # Ensure the worker thread also stops

# --------------------------------------------------------------------------
# --- Thread 2: ML Inference Worker ---
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
            
        # Check if we have enough new data for an inference window
        if current_buffer_size >= FEATURE_WINDOW:
            
            # --- Safely extract the data window ---
            recent_data = None
            with buffer_lock:
                # Extract only the last FEATURE_WINDOW samples
                recent_data = list(emg_buffer)[-FEATURE_WINDOW:] 
            
            if recent_data is None:
                continue
                
            # Separate the raw EMG data from the timestamp
            # Convert to float64 for compatibility with filtering/NumPy math
            data_window = np.array([item[1] for item in recent_data], dtype=np.float64)
            latest_timestamp = recent_data[-1][0]
            
            # --- Run Inference ---
            try:
                prediction, details = actual_inference_caller(data_window)
                
                # Print results in a single line (using carriage return)
                print(f"\r[t={latest_timestamp:.3f}s | "
                      f"Prediction: **{prediction}** | "
                      f"RMS: {np.round(details, 2)}]", end='', flush=True)
                
            except Exception as e:
                print(f"\n❌ Worker: Error during inference: {e}")
                
        else:
            # Wait a short period if the buffer is not yet full enough
            time.sleep(0.01) # 10 ms sleep
            
    print("\n🧠 Worker: Thread stopped.")

# --------------------------------------------------------------------------
# --- Main Execution ---
# --------------------------------------------------------------------------

def main():
    """Starts the two threads and handles graceful shutdown."""
    
    if SVC_MODEL is None and stop_event.is_set():
        print("🛑 Main: Cannot run inference without a loaded SVC model. Exiting.")
        return

    # 1. Initialize and start the threads
    listener_thread = threading.Thread(target=data_listener_thread)
    worker_thread = threading.Thread(target=inference_worker_thread)
    
    listener_thread.start()
    worker_thread.start()
    
    try:
        # 2. Keep the main thread alive and responsive to Ctrl+C
        while not stop_event.is_set():
            time.sleep(1) 
            
    except KeyboardInterrupt:
        print("\n🛑 Main: Shutdown signal received (Ctrl+C).")
        
    finally:
        # 3. Signal threads to stop and wait for them to finish
        stop_event.set()
        print("🛑 Main: Waiting for threads to terminate...")
        
        listener_thread.join()
        worker_thread.join()
        
        print("🎉 Main: All threads terminated. Program finished.")

if __name__ == '__main__':
    main()