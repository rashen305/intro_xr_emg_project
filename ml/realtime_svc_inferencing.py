import socket
import threading
import json
import time
import collections
import numpy as np
import joblib  # for loading the SVR model
from constants import N_CHANNELS, WINDOW_SIZE
from feature_extraction import rms
from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *
from Phidget22.BridgeGain import BridgeGain

# --- Configuration ---
HOST = '127.0.0.1'
PORT = 9002
BUFFER_SIZE = 1024  # number of EMG samples to keep in buffer

# Smoothing configuration
SMOOTHING_ALPHA_SLOW = 0.1  # Smoothing factor for stable force (lower = more smoothing)
SMOOTHING_ALPHA_FAST = 0.6  # Smoothing factor for rapid changes (higher = faster response)
CHANGE_THRESHOLD = 1.5      # Threshold to detect significant changes (in force units)

# Phidget calibration constants (from C++ code)
PHIDGET_GAIN = -4839.02337806184
PHIDGET_OFFSET = -1.3201e-5

emg_buffer = collections.deque(maxlen=BUFFER_SIZE)
buffer_lock = threading.Lock()
stop_event = threading.Event()

# Smoothing state
smoothed_force = None
smoothing_lock = threading.Lock()

# Phidget handle (global)
phidget_ch = None
phidget_lock = threading.Lock()
actual_force = 0.0

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

# --------------------------------------------------------------------------
# --- Phidget Initialization ---
# --------------------------------------------------------------------------
def initialize_phidget():
    """Initialize Phidget load cell"""
    global phidget_ch
    try:
        print("🔧 Initializing Phidget load cell...")
        phidget_ch = VoltageRatioInput()
        
        # Open and wait for attachment (5 second timeout)
        print("   Opening Phidget...")
        phidget_ch.openWaitForAttachment(5000)
        
        # Configure after opening
        print("   Setting bridge enabled...")
        phidget_ch.setBridgeEnabled(True)
        
        print("   Setting bridge gain...")
        phidget_ch.setBridgeGain(BridgeGain.BRIDGE_GAIN_128)
        
        print("   Setting data interval to 10ms...")
        phidget_ch.setDataInterval(10)
        
        print("✅ Phidget load cell connected and configured!")
        print(f"   Channel: {phidget_ch.getChannel()}")
        print(f"   Bridge enabled: {phidget_ch.getBridgeEnabled()}")
        print(f"   Bridge gain: {phidget_ch.getBridgeGain()}")
        print(f"   Data interval: {phidget_ch.getDataInterval()}ms")
        
        # Test read
        test_voltage = phidget_ch.getVoltageRatio()
        test_force = (test_voltage * PHIDGET_GAIN) + PHIDGET_OFFSET
        print(f"   Test voltage ratio: {test_voltage}")
        print(f"   Test force: {test_force:.3f} kg")
        
        return True
    except Exception as e:
        print(f"❌ Phidget initialization error: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  Continuing without actual force measurement...")
        phidget_ch = None
        return False

def get_actual_force():
    """Read current force from Phidget load cell"""
    global actual_force, phidget_ch
    
    if phidget_ch is None:
        return 0.0
    
    try:
        voltage_ratio = phidget_ch.getVoltageRatio()
        force = (voltage_ratio * PHIDGET_GAIN) + PHIDGET_OFFSET
        with phidget_lock:
            actual_force = force
        return force
    except Exception as e:
        # Return last known value on error
        print(f"\n⚠️  Phidget read error: {e}")
        with phidget_lock:
            return actual_force

SVR_MODEL = load_svr_model()

# --------------------------------------------------------------------------
# --- Bar Graph Visualization ---
# --------------------------------------------------------------------------
def create_bar_graph(predicted, actual, bar_width=20, max_force=3.0):
    """
    Create a compact single-line bar graph comparing predicted vs actual force.
    
    Args:
        predicted: Predicted force value
        actual: Actual force value
        bar_width: Width of each bar in characters
        max_force: Maximum force for scaling (default 3kg)
    
    Returns:
        String containing the bar graph visualization (single line)
    """
    # Calculate bar lengths (capped at max_force)
    pred_length = int((min(abs(predicted), max_force) / max_force) * bar_width)
    actual_length = int((min(abs(actual), max_force) / max_force) * bar_width)
    
    # Create bars
    pred_bar = '█' * pred_length
    actual_bar = '█' * actual_length
    
    # Build compact single-line visualization
    diff = abs(predicted - actual)
    result = (f"Pred:|{pred_bar:<{bar_width}}|{predicted:5.2f}kg | "
              f"Actual:|{actual_bar:<{bar_width}}|{actual:5.2f}kg | "
              f"Diff:{diff:5.2f}kg")
    
    return result

# --------------------------------------------------------------------------
# --- Adaptive Smoothing Function ---
# --------------------------------------------------------------------------
def adaptive_smooth_force(new_force: float):
    """
    Apply adaptive exponential moving average (EMA) smoothing.
    Uses fast response for large changes, slow smoothing for small changes.
    
    Args:
        new_force: The new predicted force value
    
    Returns:
        smoothed_force: The smoothed force value
    """
    global smoothed_force
    
    with smoothing_lock:
        if smoothed_force is None:
            # Initialize on first call
            smoothed_force = new_force
            return smoothed_force
        
        # Calculate the change magnitude
        change = abs(new_force - smoothed_force)
        
        # Adaptive alpha: use fast response for large changes, slow for small
        if change > CHANGE_THRESHOLD:
            alpha = SMOOTHING_ALPHA_FAST  # Quick reaction to big changes
        else:
            alpha = SMOOTHING_ALPHA_SLOW  # Smooth out small fluctuations
        
        # Apply exponential moving average
        smoothed_force = alpha * new_force + (1 - alpha) * smoothed_force
        
        return smoothed_force

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
    feature_vector = rms.preprocess_force_rms_realtime(data_window)

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

    # Print rate limiting for force: 5 times per second = 0.2 seconds interval
    last_force_update_time = 0
    force_update_interval = 0.2  # seconds (5 Hz)
    last_displayed_force = 0.0
    
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
                
                # Apply adaptive smoothing (always smooth, even if not printing)
                smoothed_force_value = adaptive_smooth_force(predicted_force)
                
                # Update displayed force only 5 times per second
                current_time = time.time()
                if current_time - last_force_update_time >= force_update_interval:
                    last_displayed_force = smoothed_force_value
                    last_force_update_time = current_time
                    
                    # Sample actual force from Phidget
                    actual_force_value = get_actual_force()
                    
                    # Display bar graph on same line
                    bar_graph = create_bar_graph(last_displayed_force, actual_force_value)
                    print(f"\r{bar_graph}", end='', flush=True)
                    
            except Exception as e:
                print(f"\n❌ Worker: Inference error: {e}")
        else:
            time.sleep(0.01)

    print("\n🧠 Worker: Thread stopped.")

# --------------------------------------------------------------------------
# --- Main ---
# --------------------------------------------------------------------------
def main():
    global phidget_ch
    
    if SVR_MODEL is None and stop_event.is_set():
        print("🛑 Cannot run inference without a loaded SVR model. Exiting.")
        return

    # Initialize Phidget load cell
    initialize_phidget()
    
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
        
        # Clean up Phidget
        if phidget_ch is not None:
            try:
                phidget_ch.close()
                print("🔧 Phidget closed.")
            except:
                pass
        
        print("🎉 All threads terminated. Program finished.")

if __name__ == '__main__':
    main()
