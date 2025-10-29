// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/host_api.hpp>
#include "tt_metal/impl/device/device.hpp" #include < tt - metalium / host_api.hpp>
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
    return 1000000 * combined_time;
}

int main(int argc, char** argv) {
    int device_id = 0;
    Device* device = CreateDevice(device_id);
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    const uint32_t input_arr_size = 64;
    uint32_t input_array[input_arr_size];
    for (int i = 0; i < input_arr_size; i++) {
        input_array[i] = (uint32_t)time_4dig_int() + i * 100000;
    }
    printf("Host's input number was %d\n", input_array[0]);

    constexpr uint32_t single_tile_size = input_arr_size * sizeof(uint32_t);
    InterleavedBufferConfig dram_config{
        .device = device, .size = single_tile_size, .page_size = single_tile_size, .buffer_type = BufferType::DRAM};
    std::shared_ptr<Buffer> src_dram_buffer = CreateBuffer(dram_config);
    auto src_dram_noc_coord = src_dram_buffer->noc_coordinates();
    uint32_t src_dram_noc_x = src_dram_noc_coord.x;
    uint32_t src_dram_noc_y = src_dram_noc_coord.y;
    EnqueueWriteBuffer(cq, src_dram_buffer, input_array, false);

    std::array<CoreCoord, 64> core_array;
    // Populate the array using a loop
    for (uint32_t i = 0; i < core_array.size(); i++) {
        core_array[i] = {i / 8, i % 8};  // y is fixed at 0, x ranges from 0 to 7
        CoreCoord dst_core_coord = device->worker_core_from_logical_core(core_array[i]);
        // printf(
        //     "Core (%d, %d) is physical core (%d, %d)\n",
        //     i / 8,
        //     i % 8,
        //     static_cast<uint32_t>(dst_core_coord.x),
        //     static_cast<uint32_t>(dst_core_coord.y));
    }
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {7, 7};
    CoreRange cores(start_core, end_core);

    uint32_t receiver_semaphore_id = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);

    uint32_t cb_id = CBIndex::c_0;
    tt::DataFormat cb_data_format = tt::DataFormat::UInt32;

    CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(input_arr_size * sizeof(uint32_t), {{cb_id, cb_data_format}})
            .set_page_size(cb_id, input_arr_size * sizeof(uint32_t));

    auto cb_src = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config);
    // Configure and Create Void DataMovement Kernels

    KernelHandle lead_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hello_charlie/kernels/lead_kernel.cpp",
        core_array[0],
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Configure Program and Start Program Execution on Device
    CoreCoord dst_core_coord = device->worker_core_from_logical_core(core_array[1]);
    CoreCoord src_core_coord;
    SetRuntimeArgs(
        program,
        lead_kernel,
        core_array[0],
        {(uint32_t)dst_core_coord.x,
         (uint32_t)dst_core_coord.y,
         receiver_semaphore_id,
         input_arr_size,
         src_dram_buffer->address(),
         src_dram_noc_x,
         src_dram_noc_y});
    for (int i = 1; i < core_array.size(); i++) {
        if ((i / 8) % 2 == 0) {  // Even row (left to right)

            if (i % 8 != 0) {  // Not the last core in the row
                src_core_coord = device->worker_core_from_logical_core(core_array[i - 1]);
            } else {  // First core in the row, receive from above
                src_core_coord = device->worker_core_from_logical_core(core_array[i - 8]);
            }
            if (i % 8 != 7) {  // Not the last core in the row
                dst_core_coord = device->worker_core_from_logical_core(core_array[i + 1]);
            } else {  // Last core in the row, pass down
                dst_core_coord = device->worker_core_from_logical_core(core_array[i + 8]);
            }
        } else {               // Odd row (right to left)
            if (i % 8 != 0) {  // Not the first core in the row
                dst_core_coord = device->worker_core_from_logical_core(core_array[i - 1]);
            } else {  // First core in the row, pass down
                if (i != 56) {
                    dst_core_coord = device->worker_core_from_logical_core(core_array[i + 8]);
                } else {
                    dst_core_coord = device->worker_core_from_logical_core(core_array[0]);
                }
            }
            if (i % 8 != 7) {  // Not the first core in the row
                src_core_coord = device->worker_core_from_logical_core(core_array[i + 1]);
            } else {
                src_core_coord = device->worker_core_from_logical_core(core_array[i - 8]);
            }
        }
        // printf(
        //     "Core (%d, %d) from (%d, %d) to (%d, %d)\n",
        //     static_cast<uint32_t>((device->worker_core_from_logical_core(core_array[i])).x),
        //     static_cast<uint32_t>((device->worker_core_from_logical_core(core_array[i])).y),
        //     static_cast<uint32_t>(src_core_coord.x),
        //     static_cast<uint32_t>(src_core_coord.y),
        //     static_cast<uint32_t>(dst_core_coord.x),
        //     static_cast<uint32_t>(dst_core_coord.y));
        KernelHandle reader_kernel = CreateKernel(
            program,
            "tt_metal/programming_examples/hello_charlie/kernels/reader_kernel.cpp",
            core_array[i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(
            program,
            reader_kernel,
            core_array[i],
            {(uint32_t)core_array[i].x,
             (uint32_t)core_array[i].y,
             (uint32_t)src_core_coord.x,
             (uint32_t)src_core_coord.y,
             (uint32_t)dst_core_coord.x,
             (uint32_t)dst_core_coord.y,
             receiver_semaphore_id,
             input_arr_size});
    }
    EnqueueProgram(cq, program, false);

    // Wait Until Program Finishes, Print "Hello World!", and Close Device

    Finish(cq);
    CloseDevice(device);
    printf("Program finishing\n");
    return 0;
}
