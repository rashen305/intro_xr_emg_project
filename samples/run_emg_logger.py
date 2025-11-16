#!/usr/bin/env python3
"""
Python script to run emg-data-sample.exe, read its terminal output,
parse the EMG data, save to CSV, and display live visualization.
"""

import subprocess
import sys
import os
import re
import csv
import time
from datetime import datetime

def parse_emg_line(line):
    """
    Parse EMG data from a line like: [  12][  -5][   3][  -8][  15][  -2][   7][  -1]
    Returns list of 8 integers or None if parsing fails.
    """
    # Match pattern: [value] repeated 8 times
    pattern = r'\[([^\]]+)\]'
    matches = re.findall(pattern, line)
    
    if len(matches) == 8:
        try:
            values = [int(match.strip()) for match in matches]
            return values
        except ValueError:
            return None
    return None

def display_emg_data(values, sample_count):
    """
    Display EMG data with bar visualization from -100 to +100.
    """
    # Clear screen (ANSI escape codes)
    print("\033[2J\033[H", end='')
    
    # Print header
    print("=== EMG Data Live Display (Range: -100 to +100) ===")
    print(f"Samples recorded: {sample_count}")
    print("=" * 70)
    print()
    
    # Display each channel with bar visualization
    for channel in range(8):
        value = values[channel]
        
        # Clamp value to -100 to +100 range for display
        display_value = max(-100, min(100, value))
        
        # Print channel number and value
        print(f"Ch{channel}: {value:4d}  ", end='')
        
        # Create bar visualization (201 characters wide: -100 to +100)
        # Scale: -100 to +100 maps to 0 to 200
        bar_position = display_value + 100  # Shift from -100..100 to 0..200
        center = 100  # Center position (value 0)
        
        print("|", end='')
        for i in range(201):
            if i == center:
                print("|", end='')  # Center marker at 0
            elif value < 0 and i >= bar_position and i < center:
                print("=", end='')  # Negative value bar
            elif value > 0 and i > center and i <= bar_position:
                print("=", end='')  # Positive value bar
            elif value == 0 and i == center:
                print("|", end='')  # Zero at center
            else:
                print(" ", end='')
        print("|")
    
    print()
    print("        -100        0        +100")
    sys.stdout.flush()

def run_emg_logger(executable_path=None, csv_filename=None):
    """
    Run the EMG data sample executable, parse output, save to CSV, and display.
    
    Args:
        executable_path: Path to the compiled C++ executable (default: bin/emg-data-sample.exe)
        csv_filename: CSV output filename (default: auto-generated with timestamp)
    """
    # Default executable path
    if executable_path is None:
        # Get the script directory and construct path to bin folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bin_dir = os.path.join(os.path.dirname(script_dir), 'bin')
        executable_path = os.path.join(bin_dir, 'emg-data-sample.exe')
    
    # Check if executable exists
    if not os.path.exists(executable_path):
        print(f"Error: Executable not found at {executable_path}")
        return 1
    
    # Generate CSV filename if not provided
    if csv_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"emg_data_{timestamp}.csv"
    
    # Open CSV file for writing
    csv_file = open(csv_filename, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    
    # Write CSV header
    csv_writer.writerow(['Timestamp_ms', 'Channel_0', 'Channel_1', 'Channel_2', 
                         'Channel_3', 'Channel_4', 'Channel_5', 'Channel_6', 'Channel_7'])
    csv_file.flush()
    
    print(f"Running: {executable_path}")
    print(f"CSV file: {csv_filename}")
    print("Press Ctrl+C to stop recording...")
    print("=" * 60)
    time.sleep(1)  # Brief pause before starting
    
    start_time = time.time()
    sample_count = 0
    last_values = [0] * 8
    
    try:
        # Run the executable and capture output in real-time
        process = subprocess.Popen(
            executable_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1  # Line buffered
        )
        
        # Read and process output
        try:
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                
                if output:
                    # Parse EMG data from the line
                    values = parse_emg_line(output)
                    
                    if values:
                        # Calculate timestamp in milliseconds
                        elapsed_ms = int((time.time() - start_time) * 1000)
                        
                        # Save to CSV
                        row = [elapsed_ms] + values
                        csv_writer.writerow(row)
                        csv_file.flush()
                        
                        # Update display
                        last_values = values
                        sample_count += 1
                        display_emg_data(values, sample_count)
                        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Stopping...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        # Wait for process to complete
        if process.poll() is None:
            process.wait()
        
    except FileNotFoundError:
        print(f"Error: Could not find executable at {executable_path}")
        csv_file.close()
        return 1
    except Exception as e:
        print(f"Error running executable: {e}")
        csv_file.close()
        return 1
    finally:
        csv_file.close()
        print(f"\nRecording stopped. Total samples: {sample_count}")
        print(f"Data saved to: {csv_filename}")
    
    return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run EMG data sample and save output to CSV"
    )
    parser.add_argument(
        "-e", "--executable",
        help="Path to the emg-data-sample executable",
        default=None
    )
    parser.add_argument(
        "-o", "--output",
        help="CSV output filename (default: auto-generated with timestamp)",
        default=None
    )
    
    args = parser.parse_args()
    
    exit_code = run_emg_logger(
        executable_path=args.executable,
        csv_filename=args.output
    )
    
    sys.exit(exit_code)
