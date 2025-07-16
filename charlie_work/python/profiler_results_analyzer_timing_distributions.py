import pandas as pd
import numpy as np

# Load CSV
def analyze_execution_cycles(csv_file):
    # Read CSV (skipping header row)
    df = pd.read_csv(csv_file, skiprows=1)
    
    # Filter rows for ALL_RED_LOOP zone
    all_red_loop = df[df['  zone name'] == 'ALL_RED_LOOP']
    
    # Group by relevant fields
    grouped = all_red_loop.groupby([' core_x', ' core_y', ' RISC processor type', ' type'])
    
    # Extract latest time entries for ZONE_START and ZONE_END per core
    execution_cycles = {}
    for (core_x, core_y, processor, phase), group in grouped:
        latest_entry = group.sort_values(' time[cycles since reset]', ascending=False).iloc[0]
        key = (core_x, core_y)
        if key not in execution_cycles:
            execution_cycles[key] = {'ZONE_START': {}, 'ZONE_END': {}}
        if phase not in execution_cycles[key]:
            execution_cycles[key][phase] = {}
        execution_cycles[key][phase][processor] = latest_entry[' time[cycles since reset]']
    
    # Collect execution time per core and store normalized values
    core_times = {}
    for (core_x, core_y), phases in execution_cycles.items():
        if 'ZONE_START' in phases and 'ZONE_END' in phases:
            latest_begin = max(phases['ZONE_START'].values())
            latest_end = max(phases['ZONE_END'].values())
            core_times[(core_x, core_y)] = (latest_begin, latest_end)
    
    # Find earliest start time to normalize
    if not core_times:
        print("No execution data found.")
        return
    
    earliest_start = min(start for start, _ in core_times.values())
    
    # Print normalized timing info for each core (1,1) to (9,9)
    for y in range(1, 10):
        for x in range(1, 10):
            if (x, y) in core_times:
                start_time, end_time = core_times[(x, y)]
                normalized_start = start_time - earliest_start
                normalized_end = end_time - earliest_start
                print(f"Core ({x},{y}): normalized_start={normalized_start}, normalized_end={normalized_end}")

# Example usage
csv_file = '/home/tenstorrent/tt-metal/generated/profiler/.logs/profile_log_device.csv'
analyze_execution_cycles(csv_file)
