// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "allred_helper.hpp"
#include <array>
#include <cmath>  // For std::log2
#include <cstdint>

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);

    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    int SIDE_LENGTH = (argc >= 4) ? highest_power_of_two(std::stoi(argv[3])) : 1;
    CoreRange cores({0, 0}, {SIDE_LENGTH - 1, SIDE_LENGTH - 1});

    // Initialize the allreduce  setup
    AllredSetup arStp(argc, argv, device, cq, program, cores, SIDE_LENGTH, false);

    /*NOC kernel arg initialization*/
    std::vector<uint32_t> dataflow_args(12 + 8 + 2 * arStp.SWING_ALGO_STEPS);
    dataflow_args[1] = arStp.dst_dram_buffer->address();
    dataflow_args[4] = arStp.dst_bank_id;
    dataflow_args[6] = arStp.SWING_ALGO_STEPS;
    dataflow_args[11] = arStp.NUM_TILES;
    for (int i = 0; i < 8; i++) {
        dataflow_args[12 + 2 * arStp.SWING_ALGO_STEPS + i] =
            (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);
    }

    /*Compute kernel arg initialization*/
    std::vector<uint32_t> compute_args(5);
    compute_args[0] = arStp.SWING_ALGO_STEPS;
    compute_args[4] = arStp.NUM_TILES;

    /*reused variable initialization*/
    KernelHandle dataflow_0_kernel;
    KernelHandle dataflow_1_kernel;
    KernelHandle compute_kernel;
    CoreCoord logical_core;
    CoreCoord physical_core;
    bool horizontal_step;
    bool sending_SE;
    uint32_t step_directions = 0b00000;
    int node_position, node_other_position, message_pass_depth, recv_node;
    /*create kernels for each core*/
    for (int core_i = 0; core_i < arStp.core_array.size(); core_i++) {
        physical_core = device->worker_core_from_logical_core(arStp.core_array[core_i]);
        dataflow_args[7] = (uint32_t)physical_core.x;
        dataflow_args[8] = (uint32_t)physical_core.y;
        compute_args[1] = (uint32_t)physical_core.x;
        compute_args[2] = (uint32_t)physical_core.y;
        if (arStp.core_array[core_i].x % 2 == 0) {
            dataflow_args[0] = arStp.src_1_dram_buffer->address();
            dataflow_args[2] = arStp.src_1_bank_id;
        } else {
            dataflow_args[0] = arStp.src_0_dram_buffer->address();
            dataflow_args[2] = arStp.src_0_bank_id;
        }

        horizontal_step = true;  // Start calcs on hrz step
        if (!arStp.SWING_VERSION) {
            /*Recursive doubling algo partner node calculations*/
            message_pass_depth = 1;
            for (int recdub_step = 0; recdub_step < arStp.SWING_ALGO_STEPS; recdub_step++) {
                int comm_partner_id = get_comm_partner_recdub_2D(
                    core_i, recdub_step, horizontal_step, message_pass_depth, step_directions, SIDE_LENGTH);

                logical_core = {comm_partner_id % SIDE_LENGTH, comm_partner_id / SIDE_LENGTH};

                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[12 + 2 * recdub_step] = (uint32_t)physical_core.x;
                dataflow_args[13 + 2 * recdub_step] = (uint32_t)physical_core.y;

                message_pass_depth = horizontal_step ? message_pass_depth : 2 * message_pass_depth;
                horizontal_step = !horizontal_step;
            }
        } else {
            /*Swing communication partner calculations*/
            int comm_partner_idx;
            for (int swing_step = 0; swing_step < arStp.SWING_ALGO_STEPS; swing_step++) {
                comm_partner_idx =
                    get_comm_partner_swing_2D(core_i, swing_step, horizontal_step, SIDE_LENGTH, arStp.TOTAL_NODES);
                logical_core = arStp.core_array[comm_partner_idx];

                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[12 + 2 * swing_step] = (uint32_t)physical_core.x;
                dataflow_args[13 + 2 * swing_step] = (uint32_t)physical_core.y;
                horizontal_step = !horizontal_step;
            }
            step_directions = get_SE(arStp.core_array[core_i].x, arStp.core_array[core_i].y);
        }

        dataflow_args[10] = step_directions;
        compute_args[3] = step_directions;

        /*SE Kernel*/
        dataflow_args[9] = (uint32_t)true;
        dataflow_1_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/allred_LO_2D/kernels/dataflow/"
            "dataflow_kernel.cpp",
            arStp.core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program, dataflow_1_kernel, arStp.core_array[core_i], dataflow_args);

        /*NW Kernel*/
        dataflow_args[9] = (uint32_t)false;
        dataflow_0_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/allred_LO_2D/kernels/dataflow/"
            "dataflow_kernel.cpp",
            arStp.core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, dataflow_0_kernel, arStp.core_array[core_i], dataflow_args);

        /* compute kernel */
        compute_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/allred_LO_2D/kernels/compute/"
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
