// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
// all reduce latnecy optimal 1D
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
// #include "tt_metal/common/constants.hpp"
// #include "tt_metal/detail/util.hpp"
// #include "tt_metal/impl/dispatch/command_queue.hpp"
// #include "tt_metal/detail/tt_metal.hpp"
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

int main(int argc, char** argv) {
    int device_id = 0;
    IDevice* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    const uint32_t input_arr_size = 48;
    const uint32_t core_arr_size = 8;
    const uint32_t swing_algo_steps = 3;  // log2(input_arr_size)
    uint32_t input_array[input_arr_size];
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {7, 7};
    uint32_t curr_time = (uint32_t)time_4dig_int();
    std::array<CoreCoord, core_arr_size> core_array;
    for (int i = 0; i < input_arr_size; i++) {
        input_array[i] = curr_time + i * 0;
    }
    printf("Host's input number was %d\n", input_array[0]);
    printf("Output should be %d\n", core_arr_size * input_array[0]);

    constexpr uint32_t single_tile_size = input_arr_size * sizeof(uint32_t);
    InterleavedBufferConfig dram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> src_dram_buffer = CreateBuffer(dram_config);
    auto src_dram_noc_coord = 0;  // src_dram_buffer->noc_coordinates();
    uint32_t src_dram_noc_x = 0;  // src_dram_noc_coord.x;
    uint32_t src_dram_noc_y = 0;  // src_dram_noc_coord.y;
    EnqueueWriteBuffer(cq, src_dram_buffer, input_array, false);

    // Populate the array using a loop
    for (uint32_t i = 0; i < core_array.size(); i++) {
        core_array[i] = {i % 8, i / 8};  // y is fixed at 0, x ranges from 0 to 7
        // printf(
        //     "Core (%d, %d) is physical core (%d, %d)\n",
        //     i / 8,
        //     i % 8,
        //     static_cast<uint32_t>(dst_core_coord.x),
        //     static_cast<uint32_t>(dst_core_coord.y));
    }
    CoreRange cores(start_core, end_core);

    uint32_t semaphore_1 = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);
    uint32_t semaphore_2 = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);

    uint32_t cb_id0 = CBIndex::c_0;
    tt::DataFormat cb_data_format = tt::DataFormat::UInt32;

    CircularBufferConfig cb_config0 =
        tt::tt_metal::CircularBufferConfig(input_arr_size * sizeof(uint32_t), {{cb_id0, cb_data_format}})
            .set_page_size(cb_id0, input_arr_size * sizeof(uint32_t));

    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config0);

    uint32_t cb_id1 = CBIndex::c_1;

    CircularBufferConfig cb_config1 =
        tt::tt_metal::CircularBufferConfig(input_arr_size * sizeof(uint32_t), {{cb_id1, cb_data_format}})
            .set_page_size(cb_id1, input_arr_size * sizeof(uint32_t));

    auto cb_src1 = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config1);
    CoreCoord logical_core;
    CoreCoord physical_core;
    uint32_t runtime_args[9 + 2 * swing_algo_steps];
    runtime_args[0] = input_arr_size;
    runtime_args[1] = src_dram_buffer->address();
    runtime_args[2] = src_dram_noc_x;
    runtime_args[3] = src_dram_noc_y;
    runtime_args[4] = semaphore_1;
    runtime_args[5] = semaphore_2;
    runtime_args[6] = swing_algo_steps;
    for (int i = 0; i < core_array.size(); i++) {  // calcs the destination core for each core for each step
        for (int j = 0; j < swing_algo_steps; j++) {
            logical_core = {get_comm_partner(i, j, core_arr_size), 0};
            physical_core = device->worker_core_from_logical_core(logical_core);

            // printf("Core (%d,%d) step %d partner (%d, %d)\n", i, 0, j, (int)physical_core.x, (int)physical_core.y);
            runtime_args[9 + 2 * j] = {(uint32_t)physical_core.x};
            runtime_args[10 + 2 * j] = {(uint32_t)physical_core.y};
        }
        KernelHandle allred_kernel = CreateKernel(
            program,
            "tt_metal/programming_examples/hello_charlie/kernels/allred_kernel.cpp",
            core_array[i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        physical_core = device->worker_core_from_logical_core(core_array[i]);
        runtime_args[7] = (uint32_t)physical_core.x;
        runtime_args[8] = (uint32_t)physical_core.y;
        SetRuntimeArgs(program, allred_kernel, core_array[i], runtime_args);
    }
    printf("Program starting\n");
    EnqueueProgram(cq, program, false);
    printf("Awaiting finish\n");
    Finish(cq);
    printf("Closing device\n");
    CloseDevice(device);
    printf("Program finished\n");
    return 0;
}
