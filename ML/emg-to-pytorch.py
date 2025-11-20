# This file implements the EMG listener and realtime ML inference portions of the pipeline
# TODO: send inferencing results to Unity via TCP

import socket
import threading
import json
import time
import collections
import numpy as np

# --- Configuration ---
HOST = '127.0.0.1'  # Must match the C++ sender's host
PORT = 9002         # Must match the C++ sender's port
BUFFER_SIZE = 1024  # Total number of samples to store
INFERENCE_WINDOW = 256 # Number of latest samples for inference

# The deque will hold tuples: (timestamp, [emg_channel_1, ..., emg_channel_8])
# We use a deque for efficient appending and popping from either end, and it
# automatically handles the fixed size of the buffer.
emg_buffer = collections.deque(maxlen=BUFFER_SIZE)

# Lock for safe access to the shared emg_buffer
buffer_lock = threading.Lock()

# Flag to control the main loops
stop_event = threading.Event()

# --------------------------------------------------------------------------
# --- Placeholder ML Inference Function ---
# --------------------------------------------------------------------------

def placeholder_inference(data_window: np.ndarray):
    """
    Placeholder for the actual PyTorch inference function.
    
    The input `data_window` is a NumPy array of shape (INFERENCE_WINDOW, 8).
    
    You will replace the body of this function with your model loading,
    pre-processing, and inference logic.
    """
    
    # --- YOUR CUSTOM ML INFERENCE CODE GOES HERE ---
    
    # 1. Example: Simple processing (e.g., calculating mean absolute value)
    mean_abs_emg = np.mean(np.abs(data_window), axis=0)
    
    # 2. Example: Mock inference result (e.g., gesture classification)
    # The actual result will depend on your model's output layer.
    
    # Mocking a simple "gesture" prediction based on the mean of the first channel
    if mean_abs_emg[0] > 10:
        prediction = "GESTURE_A (Flex)"
    else:
        prediction = "GESTURE_B (Rest)"

    # --- END OF CUSTOM CODE ---
    
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

            # Using conn.makefile() to read data line by line from the socket
            # This is robust for TCP streams where data can arrive in chunks.
            with conn.makefile('r') as client_socket_file:
                while not stop_event.is_set():
                    try:
                        # Read a single line (a single JSON packet from the C++ sender)
                        line = client_socket_file.readline()
                        if not line:
                            # Connection closed by the client
                            print("⚠️ Listener: Sender disconnected.")
                            break
                        
                        # The C++ sender sends a JSON string followed by '\n'
                        data = json.loads(line)
                        
                        # Extract the required data points
                        timestamp = data.get('timestamp')
                        emg_data = data.get('emg')
                        
                        if timestamp is not None and emg_data is not None:
                            sample = (timestamp, emg_data)
                            
                            # Safely add the sample to the shared buffer
                            with buffer_lock:
                                emg_buffer.append(sample)
                        
                    except json.JSONDecodeError:
                        print(f"❌ Listener: Failed to decode JSON: {line.strip()}")
                    except BlockingIOError:
                        # Should not happen with makefile() but good practice
                        pass
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
    
    # We poll the buffer and process as fast as possible
    while not stop_event.is_set():
        
        # 1. Safely check the buffer size
        current_buffer_size = 0
        with buffer_lock:
            current_buffer_size = len(emg_buffer)
        
        # 2. Only proceed if we have enough data for the inference window
        if current_buffer_size >= INFERENCE_WINDOW:
            
            # 3. Safely copy the latest INFERENCE_WINDOW samples
            # Using list(emg_buffer)[-INFERENCE_WINDOW:] creates a copy of the
            # required subset without blocking the listener for too long.
            with buffer_lock:
                # Samples are (timestamp, [emg_data])
                window_data = list(emg_buffer)[-INFERENCE_WINDOW:]
            
            # 4. Extract only the EMG values and convert to a NumPy array
            # The shape will be (INFERENCE_WINDOW, 8)
            emg_values = [sample[1] for sample in window_data]
            data_array = np.array(emg_values, dtype=np.float32)

            # 5. Perform the abstract inference
            start_time = time.time()
            try:
                # The placeholder function is called here
                prediction, details = placeholder_inference(data_array)
                inference_time = (time.time() - start_time) * 1000 # in ms

                # 6. Print the results
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
            # If not enough data, wait briefly before checking again
            # This reduces CPU usage when the buffer is filling up.
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
            # If the listener thread dies (e.g., client disconnects), 
            # it sets the stop_event, and the loop will exit.
            
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