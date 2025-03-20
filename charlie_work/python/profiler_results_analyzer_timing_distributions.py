import pandas as pd
import numpy as np

# Load CSV
def analyze_execution_cycles(csv_file):
    # Read CSV
    df = pd.read_csv(csv_file, skiprows=1)
    
    # Filter ALL_RED_LOOP
    all_red_loop = df[df['  zone name'] == 'ALL_RED_LOOP']
    
    # Group by core_x, core_y, RISC processor type, and zone phase
    grouped = all_red_loop.groupby([' core_x', ' core_y', ' RISC processor type', ' zone phase'])
    
    # Extract latest begin and end for each core_x, core_y, and processor type
    execution_cycles = {}
    for (core_x, core_y, processor, phase), group in grouped:
        latest_entry = group.sort_values(' time[cycles since reset]', ascending=False).iloc[0]
        key = (core_x, core_y)
        if key not in execution_cycles:
            execution_cycles[key] = {'begin': {}, 'end': {}}
        execution_cycles[key][phase][processor] = latest_entry[' time[cycles since reset]']
    
    # Calculate execution times and store with core information
    core_times = []  # List to store tuples of (execution_time, start_time, core_x, core_y)
    for (core_x, core_y), phases in execution_cycles.items():
        if 'begin' in phases and 'end' in phases:
            latest_begin = max(phases['begin'].values())
            latest_end = max(phases['end'].values())
            exec_time = latest_end - latest_begin
            core_times.append((exec_time, latest_begin, core_x, core_y))
    
    # Sort by y coordinate, then x coordinate
    core_times.sort(key=lambda x: (x[3], x[2]))
    
    # Find earliest start time
    earliest_start = min(start_time for _, start_time, _, _ in core_times)
    
    # Print results for each core with normalized start times
    for exec_time, start_time, x, y in core_times:
        normalized_start = start_time - earliest_start
        print(f"Core ({x},{y}): normalized_start={normalized_start}, execution_time={exec_time} cycles")

# Example usage
csv_file = '/home/tenstorrent/tt-metal/generated/profiler/.logs/profile_log_device.csv'
analyze_execution_cycles(csv_file)
