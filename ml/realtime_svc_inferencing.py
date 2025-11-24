import socket
import threading
import json
import time
import collections
import numpy as np
import joblib  # for loading the SVR model
from constants import N_CHANNELS, WINDOW_SIZE
from feature_extraction import rms

# --- Configuration ---
HOST = '127.0.0.1'
PORT = 9002
BUFFER_SIZE = 1024  # number of EMG samples to keep in buffer

emg_buffer = collections.deque(maxlen=BUFFER_SIZE)
buffer_lock = threading.Lock()
stop_event = threading.Event()

# --------------------------------------------------------------------------
# --- Model Loading ---
# --------------------------------------------------------------------------
def load_svr_model(filepath="weights/svr_force_model.pkl"):
    try:
        model = joblib.load(filepath)
        print(f"🧠 Model: Successfully loaded SVR pipeline from {filepath}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found at {filepath}. Train the SVR model first.")
        stop_event.set()
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        stop_event.set()
        return None

SVR_MODEL = load_svr_model()

# --------------------------------------------------------------------------
# --- ML Inference Function ---
# --------------------------------------------------------------------------
def actual_inference_caller(data_window: np.ndarray):
    """
    Extract RMS features and predict force using SVR regression model.
    Returns:
        predicted_force: float
        rms_values: np.ndarray of RMS per channel
    """
    if SVR_MODEL is None:
        return -1.0, np.zeros(N_CHANNELS)

    # 1. Feature Extraction
    feature_vector = rms.preprocess_rms_realtime(data_window)

    # 2. Regression prediction
    predicted_force = SVR_MODEL.predict(feature_vector)[0]

    # 3. RMS values for logging
    rms_values = np.sqrt(np.mean(data_window**2, axis=0))

    return predicted_force, rms_values

# --------------------------------------------------------------------------
# --- Thread 1: Data Listener ---
# --------------------------------------------------------------------------
def data_listener_thread():
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
                        except json.JSONDecodeError:
                            next_start = json_buffer.find('{')
                            json_buffer = json_buffer[next_start:] if next_start != -1 else ""
                            break

    except Exception as e:
        if not stop_event.is_set():
            print(f"❌ Listener: Unexpected error: {e}")
    finally:
        print("📡 Listener: Thread stopped.")
        stop_event.set()

# --------------------------------------------------------------------------
# --- Thread 2: Inference Worker ---
# --------------------------------------------------------------------------
def inference_worker_thread():
    print(f"🧠 Worker: Starting SVR inference thread. Window size: {WINDOW_SIZE} samples.")
    if SVR_MODEL is None:
        print("🧠 Worker: Model failed to load. Exiting thread.")
        return

    while not stop_event.is_set():
        with buffer_lock:
            current_buffer_size = len(emg_buffer)

        if current_buffer_size >= WINDOW_SIZE:
            with buffer_lock:
                recent_data = list(emg_buffer)[-WINDOW_SIZE:]

            data_window = np.array([item[1] for item in recent_data], dtype=np.float64)
            latest_timestamp = recent_data[-1][0]

            try:
                predicted_force, rms_values = actual_inference_caller(data_window)
                print(f"\r[t={latest_timestamp:.3f}s | "
                      f"Predicted Force: {predicted_force:.2f} | "
                      f"RMS: {np.round(rms_values, 2)}]", end='', flush=True)
            except Exception as e:
                print(f"\n❌ Worker: Inference error: {e}")
        else:
            time.sleep(0.01)

    print("\n🧠 Worker: Thread stopped.")

# --------------------------------------------------------------------------
# --- Main ---
# --------------------------------------------------------------------------
def main():
    if SVR_MODEL is None and stop_event.is_set():
        print("🛑 Cannot run inference without a loaded SVR model. Exiting.")
        return

    listener_thread = threading.Thread(target=data_listener_thread)
    worker_thread = threading.Thread(target=inference_worker_thread)

    listener_thread.start()
    worker_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutdown signal received (Ctrl+C).")
    finally:
        stop_event.set()
        listener_thread.join()
        worker_thread.join()
        print("🎉 All threads terminated. Program finished.")

if __name__ == '__main__':
    main()
