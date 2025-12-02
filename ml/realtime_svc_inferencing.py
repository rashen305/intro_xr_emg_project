import socket
import threading
import json
import time
import collections
import numpy as np
import pickle
from typing import Optional # New import for Optional type hint
from constants import N_CHANNELS, WINDOW_SIZE # Note: constants.N_CHANNELS was duplicated in original
from feature_extraction import rms

# --- Configuration for EMG Listener (Server side) ---
HOST = '127.0.0.1'  # Must match the C++ sender's host
PORT = 9002         # Must match the C++ sender's port
BUFFER_SIZE = 1024  # Total number of samples to store

# --- Configuration for Unity Sender (Client side) ---
VRHOST = '172.26.92.82'  # Headset IP Address on CMU-Secure laptop address 172.26.103.154/ headset address 172.26.92.82
VRPORT_TCP = 50000       # TCP Port for Unity Receiver

# --- Prediction Label Mapping ---
PREDICTION_LABELS = {
    0: "rest",
    1: "clench",
    2: "spread",
    3: "flexion",
    4: "extension"
}

# --- Prediction Smoothing Configuration ---
CONSECUTIVE_PREDICTIONS_REQUIRED = 5  # Default consecutive predictions for most gestures
CONSECUTIVE_PREDICTIONS_SPREAD = 10    # Require more consecutive predictions for "spread" gesture (prediction 2)

# --- Inference Rate Configuration ---
INFERENCE_RATE_HZ = 50  # Target inference rate in Hz
INFERENCE_INTERVAL = 1.0 / INFERENCE_RATE_HZ  # Time between inferences in seconds (0.01s = 10ms)

# The deque will hold tuples: (timestamp, [emg_channel_1, ..., emg_channel_8])
emg_buffer = collections.deque(maxlen=BUFFER_SIZE)

# Prediction smoothing state
# Use the maximum of the two thresholds for the deque size
prediction_history = collections.deque(maxlen=max(CONSECUTIVE_PREDICTIONS_REQUIRED, CONSECUTIVE_PREDICTIONS_SPREAD))
current_stable_prediction = 0  # Start with "rest"
prediction_lock = threading.Lock()

# Lock for safe access to the shared emg_buffer
buffer_lock = threading.Lock()

# Flag to control the main loops
stop_event = threading.Event()

# --- Global TCP Variables for Persistence ---
tcp_socket: Optional[socket.socket] = None
tcp_connected: bool = False

# --------------------------------------------------------------------------
# --- Model Loading and Feature Extraction Setup ---
# --------------------------------------------------------------------------

def load_svc_model(filepath="weights/svc_v9.pkl"):
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
# --- Prediction Smoothing Function ---
# --------------------------------------------------------------------------

def smooth_prediction(raw_prediction: int) -> int:
    """
    Apply temporal smoothing to predictions.
    Only changes the output prediction if the same prediction appears
    CONSECUTIVE_PREDICTIONS_REQUIRED times in a row.
    
    SPECIAL CASES:
    - "rest" (0): switches immediately without requiring consecutive predictions
    - "spread" (2): requires CONSECUTIVE_PREDICTIONS_SPREAD (30) consecutive predictions
    - Others: require CONSECUTIVE_PREDICTIONS_REQUIRED (15) consecutive predictions
    
    Args:
        raw_prediction: The raw prediction from the model
        
    Returns:
        The smoothed (stable) prediction
    """
    global current_stable_prediction, prediction_history
    
    with prediction_lock:
        # SPECIAL CASE: If prediction is "rest" (0), switch immediately
        if raw_prediction == 0:
            current_stable_prediction = 0
            prediction_history.clear()  # Clear history for fresh start
            return current_stable_prediction
        
        # Add the new prediction to history
        prediction_history.append(raw_prediction)
        
        # Determine required consecutive predictions based on gesture type
        if raw_prediction == 2:  # "spread" gesture
            required_count = CONSECUTIVE_PREDICTIONS_SPREAD
        else:
            required_count = CONSECUTIVE_PREDICTIONS_REQUIRED
        
        # Check if we have enough predictions
        if len(prediction_history) < required_count:
            # Not enough history yet, return current stable prediction
            return current_stable_prediction
        
        # Check if all recent predictions are identical
        if len(set(prediction_history)) == 1:
            # All predictions in history are the same
            new_prediction = prediction_history[0]
            if new_prediction != current_stable_prediction:
                # Update the stable prediction
                current_stable_prediction = new_prediction
        
        return current_stable_prediction

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
# --- TCP Connection and Sender Functions (COPIED/ADAPTED from dummy_sender.py) ---
# --------------------------------------------------------------------------

def connect_tcp_socket(host, port) -> Optional[socket.socket]:
    """Establishes a single, persistent TCP connection."""
    global tcp_socket, tcp_connected
    
    if tcp_connected and tcp_socket:
        return tcp_socket

    print(f"[UNITY-TCP] Attempting to establish persistent connection to {host}:{port}...")
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((host, port))
        
        tcp_socket = s
        tcp_connected = True
        print("[UNITY-TCP] *** PERSISTENT CONNECTION ESTABLISHED ***")
        return tcp_socket
        
    except ConnectionRefusedError:
        print(f"[UNITY-TCP] Connection refused. Ensure Unity receiver is running on {host}:{port}.")
        tcp_connected = False
        return None
    except Exception as e:
        print(f"[UNITY-TCP] Error during connection: {e}")
        tcp_connected = False
        return None

def send_tcp_data_persistent(data_string):
    """Sends data using the persistent TCP connection."""
    global tcp_socket, tcp_connected
    
    if not tcp_connected or not tcp_socket:
        # We don't print a message here to keep the terminal output clean; 
        # the connect_tcp_socket call handles the status.
        return False
        
    # Append a newline character for the Unity receiver to delimit packets.
    message = (data_string + '\n').encode('utf-8')

    try:
        tcp_socket.sendall(message)
        # Optional: Print confirmation if debugging, but removing for performance
        # print(f"[UNITY-TCP] Sent: '{message.decode().strip()}'")
        return True
    except ConnectionResetError:
        print("\n[UNITY-TCP] Connection reset by peer. Reconnecting...")
        tcp_socket.close()
        tcp_connected = False
        return False
    except BrokenPipeError:
        print("\n[UNITY-TCP] Broken pipe. Connection lost. Reconnecting...")
        tcp_socket.close()
        tcp_connected = False
        return False
    except Exception as e:
        print(f"\n[UNITY-TCP] Error during send: {e}. Reconnecting...")
        try:
            tcp_socket.close()
        except:
            pass
        tcp_connected = False
        return False
        
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
    """Continuously checks the buffer, performs ML inference, and sends results."""
    global tcp_connected
    print(f"🧠 Worker: Starting SVC inference thread. Window size: {WINDOW_SIZE} samples.") 
    print(f"🧠 Worker: Target inference rate: {INFERENCE_RATE_HZ} Hz")
    
    if SVC_MODEL is None:
        print("🧠 Worker: Model failed to load. Stopping worker thread.")
        return

    # Inference rate tracking
    inference_count = 0
    last_rate_update = time.time()
    current_inference_rate = 0.0
    
    # Inference timing control
    last_inference_time = 0.0

    while not stop_event.is_set():
        # --- NEW: Check and Maintain TCP Connection to Unity ---
        if not tcp_connected:
            connect_tcp_socket(VRHOST, VRPORT_TCP)

        current_buffer_size = 0
        with buffer_lock:
            current_buffer_size = len(emg_buffer)
            
        # Check if we have enough new data for an inference window
        if current_buffer_size >= WINDOW_SIZE:
            
            # --- Rate limiting: Check if enough time has passed since last inference ---
            current_time = time.time()
            time_since_last_inference = current_time - last_inference_time
            
            if time_since_last_inference < INFERENCE_INTERVAL:
                # Not enough time has passed, wait a bit
                sleep_time = INFERENCE_INTERVAL - time_since_last_inference
                time.sleep(sleep_time)
                continue
            
            # Update last inference time
            last_inference_time = current_time
            
            # --- Safely extract the data window ---
            recent_data = None
            with buffer_lock:
                # Extract only the last WINDOW_SIZE samples
                recent_data = list(emg_buffer)[-WINDOW_SIZE:] 
            
            if recent_data is None:
                continue
                
            # Separate the raw EMG data from the timestamp
            # Convert to float64 for compatibility with filtering/NumPy math
            data_window = np.array([item[1] for item in recent_data], dtype=np.float64)
            latest_timestamp = recent_data[-1][0]
            
            # --- Run Inference ---
            try:
                raw_prediction, details = actual_inference_caller(data_window)
                
                # Update inference rate counter
                inference_count += 1
                current_time = time.time()
                if current_time - last_rate_update >= 1.0:
                    current_inference_rate = inference_count / (current_time - last_rate_update)
                    inference_count = 0
                    last_rate_update = current_time
                
                # Apply temporal smoothing to reduce false positives
                smoothed_prediction = smooth_prediction(int(raw_prediction))
                
                # Convert predictions to string labels
                raw_label = PREDICTION_LABELS.get(int(raw_prediction), "unknown")
                smoothed_label = PREDICTION_LABELS.get(smoothed_prediction, "unknown")
                
                # Print results in a single line (using carriage return)
                print(f"\r[{current_inference_rate:.1f} Hz | "
                      f"Raw: {raw_label} → Smoothed: **{smoothed_label}** | "
                      f"RMS: {np.round(details, 2)}]", end='', flush=True)
                
                # --- NEW: Build and Send Prediction Packet ---
                # Build the DataPacket structure with smoothed label
                prediction_packet = {
                    "classification": smoothed_label,  # Send smoothed label to Unity
                }
                json_string = json.dumps(prediction_packet)
                
                # Send via Persistent TCP Connection
                send_tcp_data_persistent(json_string)

            except Exception as e:
                print(f"\n❌ Worker: Error during inference or send: {e}")
                
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
        
        # Clean up the global TCP socket on shutdown
        global tcp_socket
        if tcp_socket:
             tcp_socket.close()
             
        print("🎉 Main: All threads terminated. Program finished.")

if __name__ == '__main__':
    main()