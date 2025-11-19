import socket
import json
import time
import random

# --- Configuration ---
# Use 127.0.0.1 (localhost) if Unity is running on the same machine.
# If running on different machines, use the IP address of the machine running Unity.
HOST = '127.0.0.1'
TCP_PORT = 9000  # Port for TCP streaming
UDP_PORT = 9001  # Port for UDP datagrams

# --- Global TCP Socket ---
global_tcp_socket = None

# --- Payload ---
# The dummy JSON packet we want to send
DUMMY_DATA = {
    "classification": "pinch",
    "confidence": 100
}
# Convert the Python dictionary to a JSON string and then to bytes
JSON_STRING = json.dumps(DUMMY_DATA)

# --- TCP Setup Function ---
def setup_tcp_connection(host, port):
    """Attempts to establish a persistent TCP connection."""
    global global_tcp_socket
    print(f"[TCP] Attempting to establish persistent connection to {host}:{port}...")
    try:
        # Create a TCP socket (SOCK_STREAM)
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Set a timeout so we don't block indefinitely
        s.settimeout(5) 
        s.connect((host, port))
        s.settimeout(None) # Remove timeout once connected
        global_tcp_socket = s
        print(f"[TCP] Persistent connection established to {host}:{port}.")
        return True
    except ConnectionRefusedError:
        print(f"[TCP] Connection refused. Unity receiver is not yet running on {host}:{port}.")
        return False
    except TimeoutError:
        print(f"[TCP] Connection attempt timed out.")
        return False
    except Exception as e:
        print(f"[TCP] An unexpected error occurred during setup: {e}")
        return False

# --- TCP Sender Function (Uses Global Socket) ---
def send_tcp_data(data_string):
    """Sends data using the existing persistent TCP connection."""
    global global_tcp_socket
    if global_tcp_socket is None:
        print("[TCP] Cannot send: Socket not connected.")
        return False

    # We append a newline character to help the receiver (Unity) delimit packets.
    message = (data_string + '\n').encode('utf-8')
    
    try:
        global_tcp_socket.sendall(message)
        print(f"[TCP] Sent: '{message.decode().strip()}'")
        return True
    except BrokenPipeError:
        # Unity client disconnected unexpectedly
        print("[TCP] Connection lost (Broken Pipe). Retrying connection...")
        global_tcp_socket.close()
        global_tcp_socket = None
        # Attempt to re-establish connection immediately
        setup_tcp_connection(HOST, TCP_PORT)
        return False
    except Exception as e:
        print(f"[TCP] An error occurred during send: {e}")
        return False

# --- UDP Sender Function ---
# This remains connectionless and works fine as is.
def send_udp_data(host, port, data_string):
    """Sends data via UDP without establishing a persistent connection."""
    message = data_string.encode('utf-8')

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(message, (host, port))
            print(f"[UDP] Sent: '{message.decode()}' to {host}:{port}")
            return True
    except Exception as e:
        print(f"[UDP] An error occurred: {e}")
        return False

# --- Main Execution Loop ---
if __name__ == "__main__":
    print("--- Dummy Socket Data Sender Initialized (Persistent TCP Mode) ---")
    print(f"Target: {HOST}")
    print(f"TCP Port: {TCP_PORT}, UDP Port: {UDP_PORT}")
    print("------------------------------------------")
    
    # 1. Attempt to establish persistent TCP connection before the main loop
    setup_tcp_connection(HOST, TCP_PORT)

    send_count = 0
    
    # Loop indefinitely to keep sending data
    while True:
        send_count += 1
        print(f"\n--- Sending Packet #{send_count} ---")

        # --- Payload ---
        # The dummy JSON packet we want to send
        dummy_data = {
            "classification": "pinch",
            "confidence": random.randint(0, 100)
        }
        # Convert the Python dictionary to a JSON string and then to bytes
        json_string = json.dumps(dummy_data)
        
        # 1. Send via TCP (only if connected)
        send_tcp_data(json_string)
        
        # 2. Send via UDP (always works)
        send_udp_data(HOST, UDP_PORT, json_string)
        
        # Check if TCP is currently disconnected and attempt to reconnect if needed
        if global_tcp_socket is None:
            setup_tcp_connection(HOST, TCP_PORT)
        
        # Wait for 200ms before sending the next packet
        time.sleep(0.2)