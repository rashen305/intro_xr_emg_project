import socket
import json
import time
import random

'''
This file implements a dummy socket data sender that transmits JSON-formatted data
to a Unity application via both TCP and UDP protocols. It is intended for testing
socket communication with Unity receivers.

The dummy data packet is sent every second and contains a simple classification and confidence value.

TODO: Parse the output of our ML inferencing and send real-time packets instead of dummy data.
'''

# --- Configuration ---
# Use 127.0.0.1 (localhost) if Unity is running on the same machine.
# If running on different machines, use the IP address of the machine running Unity.
HOST = '127.0.0.1'
TCP_PORT = 9000  # Port for TCP streaming
UDP_PORT = 9001  # Port for UDP datagrams

# --- Payload ---
# The dummy JSON packet we want to send
DUMMY_DATA = {
    "classification": "pinch",
    "confidence": 100
}
# Convert the Python dictionary to a JSON string and then to bytes
JSON_STRING = json.dumps(DUMMY_DATA)

# --- TCP Sender Function ---
def send_tcp_data(host, port, data_string):
    """Establishes a connection and sends data via TCP."""
    # TCP is stream-based, reliable, and connection-oriented.
    # We append a newline character to help the receiver (Unity) delimit packets.
    message = (data_string + '\n').encode('utf-8')
    
    print(f"[TCP] Attempting to connect to {host}:{port}...")
    
    try:
        # Create a TCP socket (SOCK_STREAM)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((host, port))
            s.sendall(message)
            print(f"[TCP] Sent: '{message.decode().strip()}'")
            return True
    except ConnectionRefusedError:
        print(f"[TCP] Connection refused. Ensure your Unity receiver is running on {host}:{port} and accepting connections.")
        return False
    except Exception as e:
        print(f"[TCP] An error occurred: {e}")
        return False

# --- UDP Sender Function ---
def send_udp_data(host, port, data_string):
    """Sends data via UDP without establishing a persistent connection."""
    # UDP is datagram-based, unreliable, and connectionless.
    # No delimiter is strictly needed, but we still send the JSON string as bytes.
    message = data_string.encode('utf-8')

    try:
        # Create a UDP socket (SOCK_DGRAM)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.sendto(message, (host, port))
            print(f"[UDP] Sent: '{message.decode()}' to {host}:{port}")
            return True
    except Exception as e:
        print(f"[UDP] An error occurred: {e}")
        return False

# --- Main Execution Loop ---
if __name__ == "__main__":
    print("--- Dummy Socket Data Sender Initialized ---")
    print(f"Target: {HOST}")
    print(f"TCP Port: {TCP_PORT}, UDP Port: {UDP_PORT}")
    print("------------------------------------------")
    
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
        
        # 1. Send via TCP
        # Note: TCP requires the receiver (Unity) to be listening and to accept the connection.
        send_tcp_data(HOST, TCP_PORT, json_string)
        # 2. Send via UDP
        # Note: UDP does not require a connection, so it sends immediately even if the receiver isn't ready.
        send_udp_data(HOST, UDP_PORT, json_string)

        # Wait for 1 second before sending the next packet
        time.sleep(1)