import subprocess
import csv
import os
import re
from itertools import product

# Define variables
modes = ["allred_LO_2D", "allred_BO_2D"]  # Fill in desired modes
swing_algos = [0, 1]  # Fill in desired swing algos
data_sizes = [1, 2, 3,4,5]  # Fill in desired data sizes
output_csv = "profiler_results.csv"

# CSV Header
csv_header = ["MODE", "SWING_ALGO", "DATA_SIZE", "Min", "Lower Quartile", "Mean", "Median", "Upper Quartile", "Max"]

# Check if CSV exists, if not create it
if not os.path.exists(output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

# Run for every combination
for mode, swing_algo, data_size in product(modes, swing_algos, data_sizes):
    print(f"Running: MODE={mode}, SWING_ALGO={swing_algo}, DATA_SIZE={data_size}")

    # Start profiler in one terminal
    profiler_cmd = ["/home/tenstorrent/tt-metal/build_Release_tracy/tools/profiler/bin/capture-release", "-o", "generated/tracy/output.tracy", "-f"]
    profiler_proc = subprocess.Popen(["gnome-terminal", "--", *profiler_cmd])

    # Run workload in another terminal
    workload_cmd = [
        "TT_METAL_DEVICE_PROFILER=1",
        "/home/tenstorrent/tt-metal/build_Release_tracy/programming_examples/charlie_work/{}".format(mode),
        str(swing_algo), "8", "13",
        "DATA_SIZE", str(data_size)
    ]
    workload_proc = subprocess.Popen(["gnome-terminal", "--", "bash", "-c", " ".join(workload_cmd)])

    # Wait for both processes to finish
    profiler_proc.wait()
    workload_proc.wait()

    # Run the results analyzer
    analyzer_cmd = ["python3", "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/python/profiler_results_analyzer.py"]
    analyzer_output = subprocess.run(analyzer_cmd, capture_output=True, text=True).stdout

    # Extract results with regex
    result_regex = r"Min: (\d+) .*?Lower Quartile: (\d+\.?\d*)\nMean: (\d+\.?\d*)\nMedian: (\d+\.?\d*)\nUpper Quartile: (\d+\.?\d*)\nMax: (\d+)"
    match = re.search(result_regex, analyzer_output)
    
    if match:
        min_val, lower_q, mean, median, upper_q, max_val = match.groups()
        result_row = [mode, swing_algo, data_size, min_val, lower_q, mean, median, upper_q, max_val]
    else:
        print("Failed to parse results!")
        result_row = [mode, swing_algo, data_size] + ["N/A"] * 6

    # Append to CSV
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(result_row)

    print(f"Results saved for MODE={mode}, SWING_ALGO={swing_algo}, DATA_SIZE={data_size}\n")

print(f"All runs completed. Results saved in {output_csv}")
