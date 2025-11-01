# Allreduce implementations for the Wormhole n150

This repo contains 5 implementations of the allreduce algorithm for the n150, with integrated testing and debugging options, as well as some other work that was done during the development to understand the workings of the n150. In the repo, if you're interested in a detailed evaluation of the performance, you can read the pdf found in the repo.

## The algorithms implemented

All algorithms are implemented for 64 of the 72 Tensix cores available. LO works on power of 2 data sizes between 2kB and 640kB. The BO and SM implementations work on data sizes that are a multiple of 128kB, up to 640kB.

### Bandwidth and latency optimal

There are implementations that are optimized for bandwidth (BO - less data is sent between nodes and computed) and also for latency (LO - the algorithm completes in fewer steps).

### Swing and Recursive Doubling

Each of the LO and BO implementations can utilize different communication patterns, namely Swing (a toroidal algorithm, in this case for 2D) and Recursive Doubling (in this case a 2D algorithm). The choice of algorithm defines which node every node will communicate with at each step, and the NoC that will be used for communication

### Shared Memory

Additionally, I implemented a version utilizes the DRAM. There are likely many optimizations possible for the algorithm, it was mostly just done as a (relatively) quick comparative data point.

## Running the BO and LO implementations

There are various input arguments
Arg 1: is swing version? 0 1 (0 = recdub)
Arg 2: Run the kernel? 0 1
Arg 3: Size of node array 1,2,4,8 (8x8 is almost full array utilization, note smaller arrays are unstable in some configurations)
Arg 4: Random seed, -1 for a fixed array of all 1s, or any integer
arg 5: Number of tiles, for bandwidth optimal 1-5 (for 128-640kB), for latency optimal 1-320 (for 2-640kB)
arg 6: Acceptible calculation error (due to bfloat16 rounding, the maximum error will be 32)
Arg 7: Which core should copy results to host (for debugging)
Arg 8: is bandwidth optimal? 0 1 (0 = latency optimal)

eg: allred_BO_2D 1 1 8 13 1 1 1 1

## Running the SM implementation

There are various input arguments
Arg 1: is swing version? 0 1 (0 = recdub), used for node to node syncs
Arg 2: Run the kernel? 0 1
Arg 3: Size of node array 1,2,4,8 (8x8 is almost full array utilization, note smaller arrays are unstable in some configurations)
Arg 4: Random seed, -1 for a fixed array of all 1s, or any integer
arg 5: Number of tiles, for bandwidth optimal 1-5 (for 128-640kB), for latency optimal 1-320 (for 2-640kB)
arg 6: Acceptible calculation error (due to bfloat16 rounding, the maximum error will be 32)

eg: allred_mem_2D 1 1 8 13 1 1

## Performance evaluation

The full results can be found in the pdf, however if you're interested in performing your own benchmarking, you may find the "python" folder interesting.
