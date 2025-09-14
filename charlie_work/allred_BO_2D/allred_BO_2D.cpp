#include <tt-metalium/device.hpp>
#include "allred_helper.hpp"

void get_swing_block_comm_indexes(int, int, uint32_t*, bool, int, int);
void get_recdub_block_comm_indexes(int, int, uint32_t*, bool, int, int, int, uint32_t&);

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);

    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    /*
    Arg 1: is swing version? 0 1 (0 = recdub)
    Arg 2: Run the kernel? 0 1
    Arg 3: Size of node array 1,2,4,8
    Arg 4: Random source, -1, or any I
    arg 5: Number of tiles, 1-5
    arg 6: Acceptible calculation error (due to bfloat16 rounding  )
    Arg 7: Which core should copy results to host
    Arg 8: is bandwidth optimal? 0 1 (0 = latency optimal)*/

    int SIDE_LENGTH = (argc >= 4) ? highest_power_of_two(std::stoi(argv[3])) : 1;
    int PRINT_CORE = (argc >= 8) ? std::stoi(argv[7]) : 0;
    bool BANDWIDTH_OPTIMAL = (argc >= 9) ? (bool) std::stoi(argv[8]) : false;

    CoreRange cores({0, 0}, {SIDE_LENGTH - 1, SIDE_LENGTH - 1});

    // Initialize the allreduce parameters
    AllredConfig arCfg(argc, argv, device, cq, program, cores, SIDE_LENGTH, BANDWIDTH_OPTIMAL);

    /*NOC kernel arg initialization*/
    std::vector<uint32_t> dataflow_args(14 + 2 * arCfg.SWING_ALGO_STEPS + 8 + 4 * arCfg.SWING_ALGO_STEPS);
    /*args:
    0-5 : src + dst dram
    6: num steps
    7-8: core x, y
    9: core i (x+ y*side length)
    10: is_SE
    11: step_directions
    12: num_tiles
    13: tiles_per_node
    14-25: core x, y for each step
    26-33: semaphores for each step
    34-45: block indexes to send at each step
    */
    dataflow_args[1] = arCfg.dst_dram_buffer->address();
    dataflow_args[3] = PRINT_CORE;
    dataflow_args[4] = arCfg.dst_bank_id;
    dataflow_args[5] = BANDWIDTH_OPTIMAL;
    dataflow_args[6] = arCfg.SWING_ALGO_STEPS;
    dataflow_args[12] = arCfg.NUM_TILES;
    dataflow_args[13] = arCfg.NUM_TILES / arCfg.TOTAL_NODES == 0 ? 1 : arCfg.NUM_TILES / arCfg.TOTAL_NODES;  // tiles per node
    for (int i = 0; i < 8; i++) {
        dataflow_args[14 + 2 * arCfg.SWING_ALGO_STEPS + i] = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);
    }

    /*Compute kernel arg initialization*/
    std::vector<uint32_t> compute_args(6 + 2 * arCfg.SWING_ALGO_STEPS);
    compute_args[0] = arCfg.SWING_ALGO_STEPS;
    compute_args[1] = BANDWIDTH_OPTIMAL;
    compute_args[4] = arCfg.NUM_TILES;
    compute_args[5] = arCfg.NUM_TILES / arCfg.TOTAL_NODES == 0 ? 1 : arCfg.NUM_TILES / arCfg.TOTAL_NODES;  // tiles per node

    /*reused variable initialization*/
    KernelHandle dataflow_0_kernel, dataflow_1_kernel, compute_kernel;
    CoreCoord logical_core, physical_core;
    bool horizontal_step,sending_SE;
    uint32_t step_directions = 0b00000;
    uint32_t dummy_step_directions = 0b00000;
    int node_position, node_other_position, message_pass_depth, recv_node, comm_partner_idx;

    /*create kernels for each core*/
    for (int core_i = 0; core_i < arCfg.core_array.size(); core_i++) {
        physical_core = device->worker_core_from_logical_core(arCfg.core_array[core_i]);
        dataflow_args[9] = (uint32_t)core_i;  // Added core_i
        if (arCfg.core_array[core_i].x % 2 == 0) {
            dataflow_args[0] = arCfg.src_1_dram_buffer->address();
            dataflow_args[2] = arCfg.src_1_bank_id;
        } else {
            dataflow_args[0] = arCfg.src_0_dram_buffer->address();
            dataflow_args[2] = arCfg.src_0_bank_id;
        }

        /* set block indexes to 0 */
        for (int i = 0; i < 2 * arCfg.SWING_ALGO_STEPS; i++) {
            dataflow_args[22 + 2 * arCfg.SWING_ALGO_STEPS + i] = 0;
        }
        for (int i = 0; i < 2 * arCfg.SWING_ALGO_STEPS; i++) {
            compute_args[6 + i] = 0;
        }

        horizontal_step = true;  // Start calcs on hrz step
        if (!arCfg.SWING_VERSION) {
            /*Recursive doubling algo partner node calculations*/
            message_pass_depth = 1;
            for (int algo_step = 0; algo_step < arCfg.SWING_ALGO_STEPS; algo_step++) {
                comm_partner_idx = get_comm_partner_recdub_2D(
                    core_i, algo_step, horizontal_step, message_pass_depth, step_directions, SIDE_LENGTH);

                logical_core = arCfg.core_array[comm_partner_idx];
                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[14 + 2 * algo_step] = (uint32_t)physical_core.x;
                dataflow_args[15 + 2 * algo_step] = (uint32_t)physical_core.y;

                uint32_t* blocks_to_send = &dataflow_args[22 + 2 * arCfg.SWING_ALGO_STEPS + 2 * algo_step];
                if (comm_partner_idx < 32) {
                    *blocks_to_send = *blocks_to_send | (1 << comm_partner_idx);
                } else {
                    *(blocks_to_send + 1) = *(blocks_to_send + 1) | (1 << (comm_partner_idx - 32));
                }

                uint32_t* blocks_to_recv = &compute_args[6 + 2 * algo_step];
                if (core_i < 32) {
                    *blocks_to_recv = *blocks_to_recv | (1 << core_i);
                } else {
                    *(blocks_to_recv + 1) = *(blocks_to_recv + 1) | (1 << (core_i - 32));
                }

                message_pass_depth = horizontal_step ? message_pass_depth : 2 * message_pass_depth;
                horizontal_step = !horizontal_step;

                get_recdub_block_comm_indexes(
                    comm_partner_idx,
                    algo_step + 1,
                    blocks_to_send,
                    horizontal_step,
                    SIDE_LENGTH,
                    arCfg.TOTAL_NODES,
                    message_pass_depth,
                    dummy_step_directions);

                get_recdub_block_comm_indexes(
                    core_i,
                    algo_step + 1,
                    blocks_to_recv,
                    horizontal_step,
                    SIDE_LENGTH,
                    arCfg.TOTAL_NODES,
                    message_pass_depth,
                    dummy_step_directions);

                dataflow_args[22 + 4 * arCfg.SWING_ALGO_STEPS + 2 * algo_step] = compute_args[6 + 2 * algo_step];
                dataflow_args[23 + 4 * arCfg.SWING_ALGO_STEPS + 2 * algo_step] = compute_args[7 + 2 * algo_step];
            }
        } else {
            /*Swing communication partner calculations*/
            for (int algo_step = 0; algo_step < arCfg.SWING_ALGO_STEPS; algo_step++) {
                comm_partner_idx =
                    get_comm_partner_swing_2D(core_i, algo_step, horizontal_step, SIDE_LENGTH, arCfg.TOTAL_NODES);

                logical_core = arCfg.core_array[comm_partner_idx];
                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[14 + 2 * algo_step] = (uint32_t)physical_core.x;
                dataflow_args[15 + 2 * algo_step] = (uint32_t)physical_core.y;

                uint32_t* blocks_to_send = &dataflow_args[22 + 2 * arCfg.SWING_ALGO_STEPS + 2 * algo_step];
                blocks_to_send[0] = 0;
                blocks_to_send[1] = 0;
                if (comm_partner_idx < 32) {
                    *blocks_to_send = *blocks_to_send | (1 << comm_partner_idx);
                } else {
                    *(blocks_to_send + 1) = *(blocks_to_send + 1) | (1 << (comm_partner_idx - 32));
                }

                uint32_t* blocks_to_recv = &compute_args[6 + 2 * algo_step];
                blocks_to_recv[0] = 0;
                blocks_to_recv[1] = 0;
                if (core_i < 32) {
                    *blocks_to_recv = *blocks_to_recv | (1 << core_i);
                } else {
                    *(blocks_to_recv + 1) = *(blocks_to_recv + 1) | (1 << (core_i - 32));
                }

                horizontal_step = !horizontal_step;

                get_swing_block_comm_indexes(
                    comm_partner_idx, algo_step + 1, blocks_to_send, horizontal_step, SIDE_LENGTH, arCfg.TOTAL_NODES);
                get_swing_block_comm_indexes(
                    core_i, algo_step + 1, blocks_to_recv, horizontal_step, SIDE_LENGTH, arCfg.TOTAL_NODES);
                dataflow_args[22 + 4 * arCfg.SWING_ALGO_STEPS + 2 * algo_step] = compute_args[6 + 2 * algo_step];
                dataflow_args[23 + 4 * arCfg.SWING_ALGO_STEPS + 2 * algo_step] = compute_args[7 + 2 * algo_step];
            }
            step_directions = get_SE(arCfg.core_array[core_i].x, arCfg.core_array[core_i].y);
        }

        dataflow_args[11] = step_directions;
        compute_args[3] = step_directions;

        /*SE Kernel*/
        dataflow_args[10] = (uint32_t)true;
        dataflow_0_kernel = CreateDataflowKernel(program, arCfg.core_array[core_i], dataflow_args, true, "allred_BO_2D");  // SE kernel
        /*NW Kernel*/
        dataflow_args[10] = (uint32_t)false;
        dataflow_1_kernel = CreateDataflowKernel(program, arCfg.core_array[core_i], dataflow_args, false,"allred_BO_2D"); // NW kernel
        compute_kernel = CreateComputeKernel(program, arCfg.core_array[core_i], compute_args,"allred_BO_2D");
    }
    
    arCfg.RunProgram(cq, program, device);
}

void get_swing_block_comm_indexes(
    int node, int step, uint32_t* blocks, bool horizontal_step, int SIDE_LENGTH, int TOTAL_NODES) {
    int num_steps = (int)log2((double)TOTAL_NODES);
    if (step >= num_steps) {
        return;
    }
    for (int s = step; s < num_steps; s++) {
        int peer = get_comm_partner_swing_2D(node, s, horizontal_step, SIDE_LENGTH, TOTAL_NODES);
        // blocks[peer] = 1;
        if (peer < 32) {
            *blocks = *blocks | (1 << peer);
        } else {
            *(blocks + 1) = *(blocks + 1) | (1 << (peer - 32));
        }
        // step_directions = sending_SE ? (step_directions | (1 << algo_step)) : (step_directions & ~(1 <<
        // algo_step));
        horizontal_step = !horizontal_step;
        get_swing_block_comm_indexes(peer, s + 1, blocks, horizontal_step, SIDE_LENGTH, TOTAL_NODES);
    }
    return;
}

void get_recdub_block_comm_indexes(
    int node,
    int step,
    uint32_t* blocks,
    bool horizontal_step,
    int SIDE_LENGTH,
    int TOTAL_NODES,
    int message_pass_depth,
    uint32_t& step_directions) {
    int num_steps = (int)log2((double)TOTAL_NODES);
    if (step >= num_steps) {
        return;
    }
    for (int s = step; s < num_steps; s++) {
        int peer =
            get_comm_partner_recdub_2D(node, s, horizontal_step, message_pass_depth, step_directions, SIDE_LENGTH);
        if (peer < 32) {
            *blocks = *blocks | (1 << peer);
        } else {
            *(blocks + 1) = *(blocks + 1) | (1 << (peer - 32));
        }

        message_pass_depth = horizontal_step ? message_pass_depth : 2 * message_pass_depth;
        horizontal_step = !horizontal_step;
        get_recdub_block_comm_indexes(
            peer, s + 1, blocks, horizontal_step, SIDE_LENGTH, TOTAL_NODES, message_pass_depth, step_directions);
    }
    return;
}
