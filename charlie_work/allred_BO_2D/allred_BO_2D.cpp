// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <array>
#include <cmath>  // For std::log2
#include <cstdint>
// #include "hostdevcommon/profiler_common.h"
#include "allred_helper.hpp"

using namespace tt;
using namespace tt::tt_metal;

void get_swing_block_comm_indexes(int, int, uint32_t*, bool, int, int);
void get_recdub_block_comm_indexes(int, int, uint32_t*, bool, int, int, int, uint32_t&);

std::string uint32_to_binary_string(uint32_t value) {
    std::string result(32, '0');
    for (int i = 0; i < 32; i++) {
        result[31 - i] = (value & (1 << i)) ? '1' : '0';
    }
    return result;
}

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);

    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    int SIDE_LENGTH = (argc >= 4) ? highest_power_of_two(std::stoi(argv[3])) : 1;
    CoreRange cores({0, 0}, {SIDE_LENGTH - 1, SIDE_LENGTH - 1});

    // Initialize the allreduce  setup
    AllredSetup arStp(argc, argv, device, cq, program, cores, SIDE_LENGTH, true);

    /*NOC kernel arg initialization*/
    std::vector<uint32_t> dataflow_args(14 + 2 * arStp.SWING_ALGO_STEPS + 8 + 2 * arStp.SWING_ALGO_STEPS);
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
    dataflow_args[1] = arStp.dst_dram_buffer->address();
    dataflow_args[4] = arStp.dst_bank_id;
    dataflow_args[6] = arStp.SWING_ALGO_STEPS;
    dataflow_args[12] = arStp.NUM_TILES;
    dataflow_args[13] = arStp.NUM_TILES / arStp.TOTAL_NODES;  // tiles per node
    for (int i = 0; i < 8; i++) {
        dataflow_args[14 + 2 * arStp.SWING_ALGO_STEPS + i] = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);
    }

    /*Compute kernel arg initialization*/
    std::vector<uint32_t> compute_args(6 + 2 * arStp.SWING_ALGO_STEPS);
    compute_args[0] = arStp.SWING_ALGO_STEPS;
    compute_args[4] = arStp.NUM_TILES;
    compute_args[5] = arStp.NUM_TILES / arStp.TOTAL_NODES;  // tiles per node

    /*reused variable initialization*/
    KernelHandle dataflow_0_kernel;
    KernelHandle dataflow_1_kernel;
    KernelHandle compute_kernel;
    CoreCoord logical_core;
    CoreCoord physical_core;
    bool horizontal_step;
    bool sending_SE;
    uint32_t step_directions = 0b00000;
    uint32_t dummy_step_directions = 0b00000;
    int node_position, node_other_position, message_pass_depth, recv_node, comm_partner_idx;

    /*create kernels for each core*/
    for (int core_i = 0; core_i < arStp.core_array.size(); core_i++) {
        physical_core = device->worker_core_from_logical_core(arStp.core_array[core_i]);
        dataflow_args[7] = (uint32_t)physical_core.x;
        dataflow_args[8] = (uint32_t)physical_core.y;
        dataflow_args[9] = (uint32_t)core_i;  // Added core_i
        compute_args[1] = (uint32_t)physical_core.x;
        compute_args[2] = (uint32_t)physical_core.y;
        if (arStp.core_array[core_i].x % 2 == 0) {
            dataflow_args[0] = arStp.src_1_dram_buffer->address();
            dataflow_args[2] = arStp.src_1_bank_id;
        } else {
            dataflow_args[0] = arStp.src_0_dram_buffer->address();
            dataflow_args[2] = arStp.src_0_bank_id;
        }

        /* set block indexes to 0 */
        for (int i = 0; i < 2 * arStp.SWING_ALGO_STEPS; i++) {
            dataflow_args[22 + 2 * arStp.SWING_ALGO_STEPS + i] = 0;
        }
        for (int i = 0; i < 2 * arStp.SWING_ALGO_STEPS; i++) {
            compute_args[6 + i] = 0;
        }

        horizontal_step = true;  // Start calcs on hrz step
        if (!arStp.SWING_VERSION) {
            /*Recursive doubling algo partner node calculations*/
            message_pass_depth = 1;
            for (int algo_step = 0; algo_step < arStp.SWING_ALGO_STEPS; algo_step++) {
                comm_partner_idx = get_comm_partner_recdub_2D(
                    core_i, algo_step, horizontal_step, message_pass_depth, step_directions, SIDE_LENGTH);

                logical_core = arStp.core_array[comm_partner_idx];
                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[14 + 2 * algo_step] = (uint32_t)physical_core.x;
                dataflow_args[15 + 2 * algo_step] = (uint32_t)physical_core.y;

                uint32_t* blocks_to_send = &dataflow_args[22 + 2 * arStp.SWING_ALGO_STEPS + 2 * algo_step];
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
                    arStp.TOTAL_NODES,
                    message_pass_depth,
                    dummy_step_directions);

                get_recdub_block_comm_indexes(
                    core_i,
                    algo_step + 1,
                    blocks_to_recv,
                    horizontal_step,
                    SIDE_LENGTH,
                    arStp.TOTAL_NODES,
                    message_pass_depth,
                    dummy_step_directions);
            }
        } else {
            /*Swing communication partner calculations*/
            for (int algo_step = 0; algo_step < arStp.SWING_ALGO_STEPS; algo_step++) {
                comm_partner_idx =
                    get_comm_partner_swing_2D(core_i, algo_step, horizontal_step, SIDE_LENGTH, arStp.TOTAL_NODES);

                logical_core = arStp.core_array[comm_partner_idx];
                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[14 + 2 * algo_step] = (uint32_t)physical_core.x;
                dataflow_args[15 + 2 * algo_step] = (uint32_t)physical_core.y;

                uint32_t* blocks_to_send = &dataflow_args[22 + 2 * arStp.SWING_ALGO_STEPS + 2 * algo_step];
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

                horizontal_step = !horizontal_step;

                get_swing_block_comm_indexes(
                    comm_partner_idx, algo_step + 1, blocks_to_send, horizontal_step, SIDE_LENGTH, arStp.TOTAL_NODES);
                get_swing_block_comm_indexes(
                    core_i, algo_step + 1, blocks_to_recv, horizontal_step, SIDE_LENGTH, arStp.TOTAL_NODES);
            }
            step_directions = get_SE(arStp.core_array[core_i].x, arStp.core_array[core_i].y);
        }

        dataflow_args[11] = step_directions;
        compute_args[3] = step_directions;

        /*SE Kernel*/
        dataflow_args[10] = (uint32_t)true;
        dataflow_1_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/allred_BO_2D/kernels/dataflow/"
            "dataflow_kernel.cpp",
            arStp.core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program, dataflow_1_kernel, arStp.core_array[core_i], dataflow_args);

        /*NW Kernel*/
        dataflow_args[10] = (uint32_t)false;
        dataflow_0_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/allred_BO_2D/kernels/dataflow/"
            "dataflow_kernel.cpp",
            arStp.core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, dataflow_0_kernel, arStp.core_array[core_i], dataflow_args);

        /* compute kernel */
        compute_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/allred_BO_2D/kernels/compute/"
            "compute_kernel.cpp",
            arStp.core_array[core_i],
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_args});
        SetRuntimeArgs(program, compute_kernel, arStp.core_array[core_i], compute_args);
    }
    if (arStp.RUN_KERNEL) {
        EnqueueProgram(cq, program, false);
        Finish(cq);
        tt_metal::detail::DumpDeviceProfileResults(device);
    }

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, arStp.dst_dram_buffer, result_vec, true);
    validate_result_vector(result_vec, arStp.src_vec_0, arStp.src_vec_1, arStp.num_els, arStp.ERROR, arStp.TOTAL_NODES);

    CloseDevice(device);
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
