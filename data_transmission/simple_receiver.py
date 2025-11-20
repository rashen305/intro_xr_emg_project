#!/usr/bin/env python3
"""
Simple TCP socket receiver for EMG data from C++ program.
Receives and prints EMG data to terminal.
"""

import socket
import json
import sys

# Configuration
HOST = '127.0.0.1'
PORT = 9002
BUFFER_SIZE = 4096


def main():
    """Main function to receive and print EMG data"""
    # Create TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        # Bind and listen
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        
        print("=" * 60)
        print("EMG Data Receiver - Simple Listener")
        print("=" * 60)
        print(f"Listening on {HOST}:{PORT}")
        print("Waiting for connection from C++ program...")
        print("Press Ctrl+C to stop")
        print("=" * 60)
        
        # Accept connection
        conn, addr = server_socket.accept()
        print(f"\nConnected to {addr}")
        print("-" * 60)
        
        # Receive data
        buffer = ""
        sample_count = 0
        
        while True:
            try:
                # Receive data
                data = conn.recv(BUFFER_SIZE).decode('utf-8')
                
                if not data:
                    print("\nConnection closed by sender")
                    break
                
                # Add to buffer
                buffer += data
                
                # Process complete lines (messages end with \n)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    
                    if line.strip():
                        try:
                            # Parse JSON
                            emg_data = json.loads(line)
                            
                            sample_count += 1
                            timestamp = emg_data.get('timestamp', 0)
                            sample = emg_data.get('sample', sample_count - 1)
                            emg_values = emg_data.get('emg', [])
                            
                            # Print formatted output
                            print(f"Sample {sample:6d} | "
                                  f"Time: {timestamp:8.3f}s | "
                                  f"EMG: {[f'{v:4d}' for v in emg_values]}")
                            
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error: {e}")
                            print(f"Raw data: {line}")
                            
            except KeyboardInterrupt:
                print("\n\nStopping receiver...")
                break
            except Exception as e:
                print(f"\nError: {e}")
                break
        
        print(f"\nTotal samples received: {sample_count}")
        
    except OSError as e:
        print(f"Socket error: {e}")
        print(f"Make sure port {PORT} is not already in use")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        server_socket.close()
        print("Socket closed")


if __name__ == "__main__":
    main()

