
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/host_api.hpp>
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"

using namespace tt;
using namespace tt::tt_metal;

int ceil_div(int a, int b) { return (a + b - 1) / b; }

int get_comm_partner(int node, int step, int num_nodes) {
    int dist = (int)((1 - (int)pow(-2, step + 1)) / 3);
    int uncorrected_comm_partner = (node % 2 == 0) ? (node + dist) : (node - dist);
    return (uncorrected_comm_partner + num_nodes) % num_nodes;
}

int main(int argc, char** argv) {
    /* Silicon accelerator setup */
    Device* device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    const uint32_t core_arr_size = 8;
    const uint32_t swing_algo_steps = 3;  // log2(core_arr_size)
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {7, 0};
    CoreRange cores(start_core, end_core);

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

    constexpr uint32_t single_tile_size = 2 * 1024;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> src_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    auto src_dram_noc_coord = src_dram_buffer->noc_coordinates();
    auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();
    uint32_t src_dram_noc_x = src_dram_noc_coord.x;
    uint32_t src_dram_noc_y = src_dram_noc_coord.y;
    uint32_t dst_dram_noc_x = dst_dram_noc_coord.x;
    uint32_t dst_dram_noc_y = dst_dram_noc_coord.y;

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    constexpr uint32_t num_input_tiles = 1;
    {
        constexpr uint32_t semaphore_tile_size = 32;

        constexpr uint32_t compute_cb_index = CBIndex::c_0;
        CircularBufferConfig cb_compute_config =
            CircularBufferConfig(semaphore_tile_size, {{compute_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(compute_cb_index, semaphore_tile_size);
        CBHandle cb_compute = tt_metal::CreateCircularBuffer(program, cores, cb_compute_config);

        constexpr uint32_t NW_cb_index = CBIndex::c_1;
        CircularBufferConfig cb_NW_config =
            CircularBufferConfig(semaphore_tile_size, {{NW_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(NW_cb_index, semaphore_tile_size);
        CBHandle cb_NW = tt_metal::CreateCircularBuffer(program, cores, cb_NW_config);

        constexpr uint32_t SE_cb_index = CBIndex::c_2;
        CircularBufferConfig cb_SE_config =
            CircularBufferConfig(semaphore_tile_size, {{SE_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(SE_cb_index, semaphore_tile_size);
        CBHandle cb_SE = tt_metal::CreateCircularBuffer(program, cores, cb_SE_config);

        constexpr uint32_t recv_cb_index = CBIndex::c_3;
        CircularBufferConfig cb_recv_config =
            CircularBufferConfig(num_input_tiles * single_tile_size, {{recv_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(recv_cb_index, single_tile_size);
        CBHandle cb_recv = tt_metal::CreateCircularBuffer(program, cores, cb_recv_config);

        constexpr uint32_t local_cb_index = CBIndex::c_16;
        constexpr uint32_t num_output_tiles = 1;
        CircularBufferConfig cb_output_config =
            CircularBufferConfig(num_output_tiles * single_tile_size, {{local_cb_index, tt::DataFormat::Float16_b}})
                .set_page_size(local_cb_index, single_tile_size);
        CBHandle cb_output = tt_metal::CreateCircularBuffer(program, cores, cb_output_config);
    }

    uint32_t semaphore_0 = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);
    uint32_t semaphore_1 = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);

    /* Create source data and write to DRAM */
    std::vector<uint32_t> src_vec;  //(single_tile_size, 14);
    src_vec = create_constant_vector_of_bfloat16(single_tile_size, 14.0f);

    EnqueueWriteBuffer(cq, src_dram_buffer, src_vec, false);

    std::vector<uint32_t> dataflow_args(15 + 2 * swing_algo_steps);
    dataflow_args[0] = src_dram_buffer->address();
    dataflow_args[1] = dst_dram_buffer->address();
    dataflow_args[2] = src_dram_noc_x;
    dataflow_args[3] = src_dram_noc_y;
    dataflow_args[4] = dst_dram_noc_x;
    dataflow_args[5] = dst_dram_noc_y;
    dataflow_args[6] = swing_algo_steps;
    dataflow_args[7] = semaphore_0;
    dataflow_args[8] = semaphore_1;

    std::vector<uint32_t> compute_args(4);
    compute_args[0] = swing_algo_steps;

    KernelHandle dataflow_kernel;
    KernelHandle compute_kernel;
    CoreCoord logical_core;
    CoreCoord physical_core;
    bool start_direction_SE = true;

    /*create kernels for each core*/
    for (int core_i = 0; core_i < core_array.size(); core_i++) {
        physical_core = device->worker_core_from_logical_core(core_array[core_i]);

        /* Set the parameters that the dataflow kernel will use */
        dataflow_args[9] = (uint32_t)physical_core.x;
        dataflow_args[10] = (uint32_t)physical_core.y;
        dataflow_args[11] = (uint32_t)true;  // this_core_SE
        dataflow_args[12] = (uint32_t)start_direction_SE;
        for (int swing_step = 0; swing_step < swing_algo_steps; swing_step += 1) {
            logical_core = {get_comm_partner(core_i, swing_step, core_arr_size), 0};
            physical_core = device->worker_core_from_logical_core(logical_core);

            // printf("Core (%d,%d) step %d partner (%d, %d)\n", core_i, 0, j, (int)physical_core.x,
            // (int)physical_core.y);
            dataflow_args[13 + 2 * swing_step] = {(uint32_t)physical_core.x};
            dataflow_args[14 + 2 * swing_step] = {(uint32_t)physical_core.y};
        }
        /* Specify data movement kernels for reading/writing data to/from DRAM */
        dataflow_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore2/kernels/"
            "dataflow/dataflow_kernel.cpp",
            core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program, dataflow_kernel, core_array[core_i], dataflow_args);

        dataflow_args[11] = (uint32_t)false;  // this_core_SE
        dataflow_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore2/kernels/"
            "dataflow/dataflow_kernel.cpp",
            core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, dataflow_kernel, core_array[core_i], dataflow_args);

        /* Set the parameters that the compute kernel will use */
        compute_args[1] = (uint32_t)physical_core.x;
        compute_args[2] = (uint32_t)physical_core.y;
        compute_args[3] = (uint32_t)start_direction_SE;

        /* Use the add_tiles operation in the compute kernel */
        compute_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore2/kernels/"
            "compute/compute_kernel.cpp",
            core_array[core_i],
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_args,
            });
        SetRuntimeArgs(program, compute_kernel, core_array[core_i], compute_args);
        start_direction_SE = !start_direction_SE;
    }

    EnqueueProgram(cq, program, false);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);
    printf("Source = %d\n", (int)src_vec[0]);  // 22 = 1102070192
    // Unpack the two bfloat16 values from the packed uint32_t
    int array_size = single_tile_size/sizeof(uint32_t);
    for (int i=0;i<array_size; i+=10){
        auto two_bfloats = unpack_two_bfloat16_from_uint32(result_vec[i]);
        // Convert the unpacked bfloat16 values back to float for printing
        float first_bfloat_value = two_bfloats.first.to_float();
        float second_bfloat_value = two_bfloats.second.to_float();
        printf("First bfloat to int = %d\n", (int)first_bfloat_value);    // 22 = 1102070192
    }
    auto two_bfloats = unpack_two_bfloat16_from_uint32(result_vec[0]);

    // Convert the unpacked bfloat16 values back to float for printing
    float first_bfloat_value = two_bfloats.first.to_float();
    float second_bfloat_value = two_bfloats.second.to_float();
    printf("Result (nocast) = %d\n", result_vec[0]);                  // 22 = 1102070192
    printf("First bfloat to int = %d\n", (int)first_bfloat_value);    // 22 = 1102070192



    printf(
        "Expected = %d (or in human fkin numbers = %d\n",
        pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(bfloat16(28.0f), bfloat16(28.0f))),
        28);
    CloseDevice(device);
}
