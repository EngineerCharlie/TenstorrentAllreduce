import subprocess
import csv
import os
import re
from itertools import product
from time import sleep

# Define variables
modes = ["allred_BO_2D","allred_LO_2D","allred_mem_2D"]  # Fill in desired modes
modes = ["allred_LO_2D"]
swing_algo_LO_BO = [0,1]  # Fill in desired swing algos
swing_algo_mem = [1]
data_sizes_LO = [1,2,4,8,16,32,64,128,192,256,320]#,2,4,8,16,32,64,128,192,256,320]  # Fill in desired data sizes
data_sizes_BO_mem = [1,2,3,4,5]  # Fill in desired data sizes
output_csv = "/home/tenstorrent/tt-metal/generated/allred_results/profiler_results.csv"
range_x = [1, 2, 3, 4, 6, 7, 8, 9]
range_y = [1, 2, 3, 4, 5, 7, 8, 9]

# CSV Header
csv_header = ["mode", "swing_algo", "data_size", "run_num"] + \
             [f"{x}{y}_start" for y in range_y for x in range_x] + \
             [f"{x}{y}_end" for y in range_y for x in range_x]

# Check if CSV exists, if not, create it
if not os.path.exists(output_csv):
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

# Iterate over workload combinations
for run_num in range(0,5):
    for mode in modes:
        if mode == "allred_BO_2D":
            path_mode = "allred_BO_2D"
            swing_algos = swing_algo_LO_BO
            data_sizes = data_sizes_BO_mem
            mode_bool = "1"
        elif mode == "allred_LO_2D":
            path_mode = "allred_BO_2D"
            swing_algos = swing_algo_LO_BO
            data_sizes = data_sizes_LO
            mode_bool = "0"
        else:
            path_mode = "allred_mem_2D"
            swing_algos = swing_algo_mem
            data_sizes = data_sizes_BO_mem

        for swing_algo, data_size in product(swing_algos, data_sizes):
            print(f"Running: MODE={mode}, SWING_ALGO={swing_algo}, DATA_SIZE={data_size}")

            # Start profiler
            profiler_cmd = [
                "/home/tenstorrent/tt-metal/build_Release_tracy/tools/profiler/bin/capture-release",
                "-o", "generated/tracy/output.tracy",
                "-f"
            ]
            profiler_proc = subprocess.Popen(profiler_cmd)

            # Run workload
            workload_cmd = [
                "TT_METAL_DEVICE_PROFILER=1",
                f"/home/tenstorrent/tt-metal/build_Release_tracy/programming_examples/charlie_work/{path_mode}",
                str(swing_algo), "1", "8", "13", str(data_size), "32", "0", mode_bool
            ]
            workload_proc = subprocess.Popen(" ".join(workload_cmd), shell=True)

            # Wait for both to finish
            profiler_proc.wait()
            workload_proc.wait()

            # Run results analyzer script
            analyzer_cmd = [
                "python3",
                "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/python/profiler_results_analyzer_timing_distributions.py"
            ]
            analyzer_output = subprocess.run(analyzer_cmd, capture_output=True, text=True).stdout

            # Extract normalized start/end data from output
            core_pattern = r"Core \((\d+),(\d+)\): normalized_start=(\d+|N/A), normalized_end=(\d+|N/A)"
            core_matches = re.findall(core_pattern, analyzer_output)

            # Prepare result row
            result_row = [mode, swing_algo, data_size, run_num]
            core_data = {}
            for x, y, start, end in core_matches:
                key = (int(x), int(y))
                core_data[key] = (start, end)

            for y in range_y:
                for x in range_x:
                    start, end = core_data.get((x, y), ("N/A", "N/A"))
                    result_row.append(start)
            for y in range_y:
                for x in range_x:
                    start, end = core_data.get((x, y), ("N/A", "N/A"))
                    result_row.append(end)

            # Append row to CSV
            with open(output_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(result_row)

            print(f"Results saved for MODE={mode}, SWING_ALGO={swing_algo}, DATA_SIZE={data_size}\n")

print(f"All runs completed. Results saved in {output_csv}")
