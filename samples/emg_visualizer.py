#!/usr/bin/env python3
"""
Python script to run emg-data-sample.exe, read its terminal output,
parse the EMG data, save to CSV, and display real-time line plots for 8 channels.
"""

import subprocess
import sys
import os
import re
import csv
import time
import threading
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# Global data storage for plotting
data_lock = threading.Lock()
timestamps = deque(maxlen=1000)  # Keep last 1000 samples
channel_data = [deque(maxlen=1000) for _ in range(8)]  # One deque per channel
sample_count = 0
process = None
csv_file = None
csv_writer = None
start_time = None

# Colors for each channel
channel_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                  '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

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

def read_emg_data(executable_path):
    """
    Read EMG data from the executable in a separate thread.
    """
    global process, sample_count, csv_writer, csv_file, start_time
    
    try:
        process = subprocess.Popen(
            executable_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            
            if output:
                values = parse_emg_line(output)
                
                if values:
                    elapsed_ms = (time.time() - start_time) * 1000
                    
                    # Save to CSV
                    if csv_writer:
                        row = [int(elapsed_ms)] + values
                        csv_writer.writerow(row)
                        csv_file.flush()
                    
                    # Store data for plotting
                    with data_lock:
                        timestamps.append(elapsed_ms)
                        for channel in range(8):
                            channel_data[channel].append(values[channel])
                        sample_count += 1
                        
    except Exception as e:
        print(f"Error reading data: {e}")

def update_plot(frame):
    """
    Update the plot with new data.
    """
    global sample_count
    
    with data_lock:
        if len(timestamps) == 0:
            return lines
        
        # Update each channel line
        for channel in range(8):
            if len(channel_data[channel]) > 0:
                lines[channel].set_data(list(timestamps), list(channel_data[channel]))
        
        # Update x-axis limits to show recent data (last 10 seconds)
        if len(timestamps) > 0:
            current_time = timestamps[-1]
            ax.set_xlim(max(0, current_time - 10000), current_time + 1000)
        
        # Update title with sample count
        ax.set_title(f'EMG Data - 8 Channels (Samples: {sample_count})', fontsize=12)
    
    return lines

def run_emg_visualizer(executable_path=None, csv_filename=None):
    """
    Run the EMG data sample executable, parse output, save to CSV, and display real-time plot.
    
    Args:
        executable_path: Path to the compiled C++ executable (default: bin/emg-data-sample.exe)
        csv_filename: CSV output filename (default: auto-generated with timestamp)
    """
    global process, csv_file, csv_writer, start_time, lines, ax
    
    # Default executable path
    if executable_path is None:
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
    print("Close the plot window or press Ctrl+C to stop recording...")
    print("=" * 60)
    
    start_time = time.time()
    
    # Start data reading thread
    data_thread = threading.Thread(target=read_emg_data, args=(executable_path,), daemon=True)
    data_thread.start()
    
    # Set up matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlabel('Time (ms)', fontsize=10)
    ax.set_ylabel('EMG Value', fontsize=10)
    ax.set_ylim(-100, 100)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Create lines for each channel
    lines = []
    for channel in range(8):
        line, = ax.plot([], [], color=channel_colors[channel], 
                       label=f'Channel {channel}', linewidth=1.5)
        lines.append(line)
    
    # Add legend
    ax.legend(loc='upper right', ncol=2, fontsize=9)
    
    # Set up animation
    ani = animation.FuncAnimation(fig, update_plot, interval=50, blit=False, cache_frame_data=False)
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Stopping...")
    finally:
        # Clean up
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        
        if csv_file:
            csv_file.close()
        
        print(f"\nRecording stopped. Total samples: {sample_count}")
        print(f"Data saved to: {csv_filename}")
    
    return 0

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run EMG data sample with real-time visualization"
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
    
    exit_code = run_emg_visualizer(
        executable_path=args.executable,
        csv_filename=args.output
    )
    
    sys.exit(exit_code)

