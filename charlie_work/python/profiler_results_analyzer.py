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
    
    # Calculate execution times
    execution_times = []
    for (core_x, core_y), phases in execution_cycles.items():
        if 'begin' in phases and 'end' in phases:
            latest_begin = max(phases['begin'].values())
            latest_end = max(phases['end'].values())
            execution_times.append(latest_end - latest_begin)
    
    # Compute statistics
    execution_times = np.array(execution_times)
    min_time = np.min(execution_times)
    lower_quartile = np.percentile(execution_times, 25)
    mean = np.mean(execution_times)
    median = np.median(execution_times)
    upper_quartile = np.percentile(execution_times, 75)
    max_time = np.max(execution_times)
    
    # Print results
    print(f"Min: {min_time}")
    print(f"Lower Quartile: {lower_quartile}")
    print(f"Mean: {mean}")
    print(f"Median: {median}")
    print(f"Upper Quartile: {upper_quartile}")
    print(f"Max: {max_time}")

# Example usage
csv_file = '/home/tenstorrent/tt-metal/generated/profiler/.logs/profile_log_device.csv'
analyze_execution_cycles(csv_file)
