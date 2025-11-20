# This file implements the EMG listener and realtime ML inference portions of the pipeline
# TODO: send inferencing results to Unity via TCP
import socket
import threading
import json
import time
import collections
import numpy as np

# --- IMPORT THE DEDICATED INFERENCE FUNCTION ---
from inference_function import run_inference 

# --- Configuration ---
HOST = '127.0.0.1'  # Must match the C++ sender's host
PORT = 9002         # Must match the C++ sender's port
BUFFER_SIZE = 1024  # Total number of samples to store
INFERENCE_WINDOW = 256 # Number of latest samples for inference

# The deque will hold tuples: (timestamp, [emg_channel_1, ..., emg_channel_8])
emg_buffer = collections.deque(maxlen=BUFFER_SIZE)

# Lock for safe access to the shared emg_buffer
buffer_lock = threading.Lock()

# Flag to control the main loops
stop_event = threading.Event()

# --------------------------------------------------------------------------
# --- Actual ML Inference Function (Now calls run_inference) ---
# --------------------------------------------------------------------------

def actual_inference_caller(data_window: np.ndarray):
    """
    Calls the run_inference function from inference_function.py.
    
    The input `data_window` is a NumPy array of shape (INFERENCE_WINDOW, 8).
    """
    
    # 1. Call the dedicated inference function
    prediction = run_inference(data_window)
    
    # 2. Calculate details (e.g., mean absolute value for logging/debugging)
    mean_abs_emg = np.mean(np.abs(data_window), axis=0)
    
    return prediction, mean_abs_emg

# --------------------------------------------------------------------------
# --- Thread 1: Data Listener and Buffer Manager ---
# --------------------------------------------------------------------------

def data_listener_thread():
    """Listens for TCP connection and receives streaming EMG data."""
    print(f"📡 Listener: Starting TCP server on {HOST}:{PORT}")
    
    try:
        # Create a socket (AF_INET for IPv4, SOCK_STREAM for TCP)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((HOST, PORT))
            s.listen(1)
            print("📡 Listener: Waiting for Myo sender (C++ program) to connect...")
            
            # Accept a connection
            conn, addr = s.accept()
            print(f"✅ Listener: Connection established from {addr}")

            with conn.makefile('r') as client_socket_file:
                while not stop_event.is_set():
                    try:
                        line = client_socket_file.readline()
                        if not line:
                            print("⚠️ Listener: Sender disconnected.")
                            break
                        
                        data = json.loads(line)
                        timestamp = data.get('timestamp')
                        emg_data = data.get('emg')
                        
                        if timestamp is not None and emg_data is not None:
                            sample = (timestamp, emg_data)
                            
                            with buffer_lock:
                                emg_buffer.append(sample)
                        
                    except json.JSONDecodeError:
                        print(f"❌ Listener: Failed to decode JSON: {line.strip()}")
                    except ConnectionResetError:
                        print("⚠️ Listener: Connection forcibly closed by the remote host.")
                        break
                    except Exception as e:
                        print(f"❌ Listener: An unexpected error occurred: {e}")
                        break
                        
    except socket.error as e:
        print(f"❌ Listener: Socket error: {e}")
    finally:
        stop_event.set()
        print("📡 Listener: Thread stopped.")


# --------------------------------------------------------------------------
# --- Thread 2: ML Inference Worker ---
# --------------------------------------------------------------------------

def inference_worker_thread():
    """Continuously checks the buffer and performs ML inference."""
    print(f"🧠 Worker: Starting inference thread. Window size: {INFERENCE_WINDOW} samples.")
    
    while not stop_event.is_set():
        
        current_buffer_size = 0
        with buffer_lock:
            current_buffer_size = len(emg_buffer)
        
        if current_buffer_size >= INFERENCE_WINDOW:
            
            with buffer_lock:
                # Copy the latest INFERENCE_WINDOW samples
                window_data = list(emg_buffer)[-INFERENCE_WINDOW:]
            
            # Extract only the EMG values and convert to a NumPy array (256, 8)
            emg_values = [sample[1] for sample in window_data]
            data_array = np.array(emg_values, dtype=np.float32)

            # Perform the actual inference
            start_time = time.time()
            try:
                prediction, details = actual_inference_caller(data_array)
                inference_time = (time.time() - start_time) * 1000 # in ms

                # Print the results
                latest_timestamp = window_data[-1][0]
                print(f"=====================================================")
                print(f"Inference Complete (Time: {inference_time:.2f} ms)")
                print(f"Window End Time: {latest_timestamp:.6f}s")
                print(f"Prediction: **{prediction}**")
                print(f"Details (Mean Abs): {np.round(details, 2)}")
                print(f"=====================================================")
                
            except Exception as e:
                print(f"❌ Worker: Error during inference: {e}")
                
        else:
            time.sleep(0.01) # 10 ms sleep
            
    print("🧠 Worker: Thread stopped.")

# --------------------------------------------------------------------------
# --- Main Execution ---
# --------------------------------------------------------------------------

def main():
    """Starts the two threads and handles graceful shutdown."""
    
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