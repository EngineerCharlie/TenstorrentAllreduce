// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
// all reduce latnecy optimal 1D

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include "tt_metal/impl/dispatch/command_queue.hpp"
#include "tt_metal/detail/tt_metal.hpp"
#include "tt_metal/impl/device/device.hpp"
#include <stdlib.h>
#include <time.h>

using namespace tt;
using namespace tt::tt_metal;

int time_4dig_int() {
    // Get the current time
    time_t now = time(0);
    struct tm* time_info = localtime(&now);

    // Extract the minute and second
    int minute = (time_info->tm_min) % 10;
    int second = time_info->tm_sec;

    // Combine minute and second into a 4-figure integer
    int combined_time = minute * 100 + second;
    return 100000 * combined_time;
}

int get_comm_partner(int node, int step, int num_nodes) {
    int dist = (int)((1 - (int)pow(-2, step + 1)) / 3);
    int uncorrected_comm_partner = (node % 2 == 0) ? (node + dist) : (node - dist);
    return (uncorrected_comm_partner + num_nodes) % num_nodes;
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

int main(int argc, char** argv) {  // change
    int device_id = 0;
    Device* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    const uint32_t input_arr_size = 4;
    const uint32_t core_arr_size = 4;
    const uint32_t swing_algo_steps = 2;  // log2(core_arr_size)

    // uint32_t input_array[input_arr_size];
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {3, 0};

    uint32_t curr_time = (uint32_t)time_4dig_int();
    std::vector<uint32_t> input_array;  //(single_tile_size, 14);
    input_array = create_constant_vector_of_bfloat16(input_arr_size, (float)curr_time);
    // for (int i = 0; i < input_arr_size; i++) {
    //     input_array[i] = curr_time + i * 0;
    // }
    printf("curr_time was %d\n", curr_time);
    printf("Host's input number was %d\n", input_array[0]);
    printf("Output should be %d\n", core_arr_size * input_array[0]);

    constexpr uint32_t single_tile_size = 2 * 1024;
    InterleavedBufferConfig dram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> src_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
    auto src_dram_noc_coord = src_dram_buffer->noc_coordinates();
    auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();
    uint32_t src_dram_noc_x = src_dram_noc_coord.x;
    uint32_t src_dram_noc_y = src_dram_noc_coord.y;
    uint32_t dst_dram_noc_x = dst_dram_noc_coord.x;
    uint32_t dst_dram_noc_y = dst_dram_noc_coord.y;
    EnqueueWriteBuffer(cq, src_dram_buffer, input_array, false);

    // Populate the array using a loop
    std::array<CoreCoord, core_arr_size> core_array;
    for (uint32_t i = 0; i < core_array.size(); i++) {
        core_array[i] = {i % 8, i / 8};  // y is fixed at 0, x ranges from 0 to 7
        // printf(
        //     "Core (%d, %d) is physical core (%d, %d)\n",
        //     i % 8,
        //     i / 8,
        //     static_cast<uint32_t>(core_array[i].x),
        //     static_cast<uint32_t>(core_array[i].y));
    }
    CoreRange cores(start_core, end_core);

    uint32_t semaphore_NW = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);
    uint32_t semaphore_SE = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);

    uint32_t cb_local_index = CBIndex::c_0;
    uint32_t cb_noc_SE_index = CBIndex::c_1;
    uint32_t cb_noc_NW_index = CBIndex::c_2;
    uint32_t cb_compute_index = CBIndex::c_3;
    uint32_t cb_recv_index = CBIndex::c_4;
    uint32_t cb_recv2_index = CBIndex::c_5;
    tt::DataFormat cb_data_format = tt::DataFormat::UInt32;

    CircularBufferConfig cb_config_local =
        tt::tt_metal::CircularBufferConfig(single_tile_size, {{cb_local_index, cb_data_format}})
            .set_page_size(cb_local_index, single_tile_size);
    CircularBufferConfig cb_config_noc_SE =
        tt::tt_metal::CircularBufferConfig(single_tile_size, {{cb_noc_SE_index, cb_data_format}})
            .set_page_size(cb_noc_SE_index, single_tile_size);
    CircularBufferConfig cb_config_noc_NW =
        tt::tt_metal::CircularBufferConfig(single_tile_size, {{cb_noc_NW_index, cb_data_format}})
            .set_page_size(cb_noc_NW_index, single_tile_size);
    CircularBufferConfig cb_config_compute =
        tt::tt_metal::CircularBufferConfig(single_tile_size, {{cb_compute_index, cb_data_format}})
            .set_page_size(cb_compute_index, single_tile_size);
    CircularBufferConfig cb_config_recv =
        tt::tt_metal::CircularBufferConfig(single_tile_size, {{cb_recv_index, cb_data_format}})
            .set_page_size(cb_recv_index, single_tile_size);
    CircularBufferConfig cb_config_recv2 =
        tt::tt_metal::CircularBufferConfig(single_tile_size, {{cb_recv2_index, cb_data_format}})
            .set_page_size(cb_recv2_index, single_tile_size);

    auto cb_local = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config_local);
    auto cb_noc_SE = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config_noc_SE);
    auto cb_noc_NW = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config_noc_NW);
    auto cb_compute = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config_compute);
    auto cb_recv = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config_recv);
    auto cb_recv2 = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config_recv2);

    CoreCoord logical_core;
    CoreCoord physical_core;
    bool start_direction_SE = true;
    bool step_direction_SE;
    KernelHandle data_movement_kernel_step0;
    KernelHandle data_movement_kernel_step1;

    uint32_t compute_args[5];
    compute_args[2] = input_arr_size;

    std::vector<uint32_t> data_movement_args(12 + 2 * ceil_div(swing_algo_steps, 2));
    data_movement_args[3] = input_arr_size;
    data_movement_args[7] = src_dram_buffer->address();
    data_movement_args[8] = src_dram_noc_x;
    data_movement_args[9] = src_dram_noc_y;
    data_movement_args[10] = src_dram_buffer->address();
    data_movement_args[11] = src_dram_noc_x;
    data_movement_args[12] = src_dram_noc_y;

    for (int i = 0; i < core_array.size(); i++) {  // calcs the destination core for each core for each step
        physical_core = device->worker_core_from_logical_core(core_array[i]);

        start_direction_SE = !start_direction_SE;
        compute_args[0] = (uint32_t)start_direction_SE;
        compute_args[3] = (uint32_t)physical_core.x;
        compute_args[4] = (uint32_t)physical_core.y;
        KernelHandle compute_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore/kernels/"
            "/compute_add_to_local.cpp",
            core_array[i],
            ComputeConfig{
                // .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = {},
            });
        SetRuntimeArgs(program, compute_kernel, core_array[i], compute_args);

        data_movement_args[5] = (uint32_t)physical_core.x;
        data_movement_args[6] = (uint32_t)physical_core.y;

        // SWING STEP 0 data movement kernel
        data_movement_args[2] = (uint32_t)start_direction_SE;
        data_movement_args[4] = ceil_div(swing_algo_steps, 2);
        if (start_direction_SE) {
            data_movement_args[0] = semaphore_SE;
            data_movement_args[1] = semaphore_NW;

            data_movement_kernel_step0 = CreateKernel(
                program,
                "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore/kernels/"
                "dataflow_data_movement.cpp",
                core_array[i],
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        } else {
            data_movement_args[0] = semaphore_NW;
            data_movement_args[1] = semaphore_SE;

            data_movement_kernel_step0 = CreateKernel(
                program,
                "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore/kernels/"
                "dataflow_data_movement.cpp",
                core_array[i],
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        }

        for (int j = 0; j < swing_algo_steps; j += 2) {
            logical_core = {get_comm_partner(i, j, core_arr_size), 0};
            physical_core = device->worker_core_from_logical_core(logical_core);

            // printf("Core (%d,%d) step %d partner (%d, %d)\n", i, 0, j, (int)physical_core.x, (int)physical_core.y);
            data_movement_args[10 + 2 * j] = {(uint32_t)physical_core.x};
            data_movement_args[11 + 2 * j] = {(uint32_t)physical_core.y};
        }

        SetRuntimeArgs(program, data_movement_kernel_step0, core_array[i], data_movement_args);

        // SWING STEP 1 data movement kernel

        data_movement_args[2] = (uint32_t)!start_direction_SE;
        data_movement_args[4] = swing_algo_steps / 2;
        if (start_direction_SE) {
            data_movement_args[0] = semaphore_NW;
            data_movement_args[1] = semaphore_SE;

            data_movement_kernel_step1 = CreateKernel(
                program,
                "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore/kernels/"
                "dataflow_data_movement.cpp",
                core_array[i],
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        } else {
            data_movement_args[0] = semaphore_SE;
            data_movement_args[1] = semaphore_NW;

            data_movement_kernel_step1 = CreateKernel(
                program,
                "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore/kernels/"
                "dataflow_data_movement.cpp",
                core_array[i],
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        }

        for (int j = 1; j < swing_algo_steps; j += 2) {
            logical_core = {get_comm_partner(i, j, core_arr_size), 0};
            physical_core = device->worker_core_from_logical_core(logical_core);

            // printf("Core (%d,%d) step %d partner (%d, %d)\n", i, 0, j, (int)physical_core.x, (int)physical_core.y);
            data_movement_args[10 + 2 * j] = {(uint32_t)physical_core.x};
            data_movement_args[11 + 2 * j] = {(uint32_t)physical_core.y};
        }
        SetRuntimeArgs(program, data_movement_kernel_step1, core_array[i], data_movement_args);
    }
    EnqueueProgram(cq, program, false);
    printf("Program started, awaiting finish\n");
    Finish(cq);
    printf("Closing device\n");
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);
    // Unpack the two bfloat16 values from the packed uint32_t
    auto two_bfloats = unpack_two_bfloat16_from_uint32(result_vec[0]);

    // Convert the unpacked bfloat16 values back to float for printing
    float first_bfloat_value = two_bfloats.first.to_float();
    float second_bfloat_value = two_bfloats.second.to_float();
    printf("Result (nocast) = %d\n", result_vec[0]);           // 22 = 1102070192
    printf("Result to int = %d\n", (int)first_bfloat_value);   // 22 = 1102070192
    printf("Result to int = %d\n", (int)second_bfloat_value);  // 22 = 1102070192
    printf(
        "Expected = %d (or in human fkin numbers = %d\n",
        pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfloat16(22.0f), bfloat16(22.0f))),
        22);
    CloseDevice(device);
    printf("Program finished\n");
    // return 0;
}
