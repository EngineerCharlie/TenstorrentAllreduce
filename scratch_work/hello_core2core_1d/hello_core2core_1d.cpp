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

    std::array<CoreCoord, 8> core_array;
    // Populate the array using a loop
    for (uint32_t i = 0; i < core_array.size(); i++) {
        core_array[i] = {i, 0};  // y is fixed at 0, x ranges from 0 to 7
        CoreCoord dst_core_coord = device->worker_core_from_logical_core(core_array[i]);
        printf(
            "Core (%d, 0) is physical core (%d, %d)\n",
            i,
            static_cast<uint32_t>(dst_core_coord.x),
            static_cast<uint32_t>(dst_core_coord.y));
    }

    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {7, 0};
    CoreRange cores(start_core, end_core);

    uint32_t receiver_semaphore_id = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);

    uint32_t cb_id = CBIndex::c_0;
    tt::DataFormat cb_data_format = tt::DataFormat::UInt32;

    CircularBufferConfig cb_config = tt::tt_metal::CircularBufferConfig(sizeof(uint32_t), {{cb_id, cb_data_format}})
                                         .set_page_size(cb_id, sizeof(uint32_t));

    auto cb_src = tt::tt_metal::CreateCircularBuffer(program, cores, cb_config);
    // Configure and Create Void DataMovement Kernels

    KernelHandle lead_kernel = CreateKernel(
        program,
        "tt_metal/programming_examples/hello_charlie/kernels/lead_kernel.cpp",
        core_array[0],
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    // Configure Program and Start Program Execution on Device
    uint32_t input_num = (uint32_t)time_4dig_int();
    CoreCoord dst_core_coord = device->worker_core_from_logical_core(core_array[1]);
    SetRuntimeArgs(
        program,
        lead_kernel,
        core_array[0],
        {input_num, (uint32_t)dst_core_coord.x, (uint32_t)dst_core_coord.y, receiver_semaphore_id});

    for (int i = 1; i < core_array.size(); i++) {
        CoreCoord src_core_coord = device->worker_core_from_logical_core(core_array[i - 1]);
        CoreCoord dst_core_coord = device->worker_core_from_logical_core(core_array[i + 1]);
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
             (uint32_t)src_core_coord.x,
             (uint32_t)src_core_coord.y,
             (uint32_t)dst_core_coord.x,
             (uint32_t)dst_core_coord.y,
             receiver_semaphore_id});
    }
    EnqueueProgram(cq, program, false);

    // Wait Until Program Finishes, Print "Hello World!", and Close Device

    Finish(cq);
    CloseDevice(device);
    printf("Program finishing\n");
    return 0;
}
