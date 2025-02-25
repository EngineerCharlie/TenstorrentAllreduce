// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include <array>
#include <cmath>  // For std::log2
#include <cstdint>

using namespace tt;
using namespace tt::tt_metal;

int highest_power_of_two(int);
void printBinary(unsigned int num) {
    for (int i = 31; i >= 0; i--) {  // Adjust 31 for smaller integers
        printf("%d", (num >> i) & 1);
    }
    printf("\n");
}

int main(int argc, char** argv) {
    /* Silicon accelerator setup */
    Device* device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    bool RUN_KERNEL = false;
    if (argc >= 2 && std::stoi(argv[2]) == 1) {
        RUN_KERNEL = true;
    }
    int SIDE_LENGTH;
    if (argc >= 3 ) {
        SIDE_LENGTH = highest_power_of_two(std::stoi(argv[3]));
    } else {
        SIDE_LENGTH = 1;
    }
    int RND_SRC = 0;
    if (argc >= 4) {
        RND_SRC = std::stoi(argv[4]);
    }

    /*Setup core array (full grid or subsection)*/
    uint32_t TOTAL_NODES = SIDE_LENGTH * SIDE_LENGTH;
    uint32_t SWING_ALGO_STEPS = static_cast<uint32_t>(std::log2(TOTAL_NODES));
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {SIDE_LENGTH - 1, SIDE_LENGTH - 1};
    CoreRange cores(start_core, end_core);

    std::vector<CoreCoord> core_array(TOTAL_NODES);
    for (uint32_t i = 0; i < core_array.size(); i++) {
        core_array[i] = {i % SIDE_LENGTH, i / SIDE_LENGTH};
    }

    /*Setup dram to pass data to/from cores*/
    constexpr uint32_t single_tile_size = 1 * 256;
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
    constexpr uint32_t num_output_tiles = 1;
    constexpr uint32_t num_semaphore_tiles = 1;
    constexpr uint32_t semaphore_tile_size = 32;
    constexpr tt::DataFormat data_format = tt::DataFormat::Float16_b;

    constexpr uint32_t cb_index_compute = CBIndex::c_0;
    CircularBufferConfig cb_config_compute =
        CircularBufferConfig(semaphore_tile_size * num_semaphore_tiles, {{cb_index_compute, data_format}})
            .set_page_size(cb_index_compute, semaphore_tile_size);
    CBHandle cb_compute = tt_metal::CreateCircularBuffer(program, cores, cb_config_compute);

    constexpr uint32_t cb_index_NW = CBIndex::c_1;
    CircularBufferConfig cb_config_NW =
        CircularBufferConfig(semaphore_tile_size * num_semaphore_tiles, {{cb_index_NW, data_format}})
            .set_page_size(cb_index_NW, semaphore_tile_size);
    CBHandle cb_NW = tt_metal::CreateCircularBuffer(program, cores, cb_config_NW);

    constexpr uint32_t cb_index_SE = CBIndex::c_2;
    CircularBufferConfig cb_config_SE =
        CircularBufferConfig(semaphore_tile_size * num_semaphore_tiles, {{cb_index_SE, data_format}})
            .set_page_size(cb_index_SE, semaphore_tile_size);
    CBHandle cb_SE = tt_metal::CreateCircularBuffer(program, cores, cb_config_SE);

    constexpr uint32_t cb_index_recv = CBIndex::c_3;
    CircularBufferConfig cb_config_recv =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{cb_index_recv, data_format}})
            .set_page_size(cb_index_recv, single_tile_size);
    CBHandle cb_recv = tt_metal::CreateCircularBuffer(program, cores, cb_config_recv);

    constexpr uint32_t cb_index_local = CBIndex::c_16;
    CircularBufferConfig cb_config_local =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{cb_index_local, data_format}})
            .set_page_size(cb_index_local, single_tile_size);
    CBHandle cb_local = tt_metal::CreateCircularBuffer(program, cores, cb_config_local);

    /* Create source data and write to DRAM */
    std::vector<uint32_t> src_vec;  //(single_tile_size, 14);
    if (RND_SRC == -1) {
        src_vec = create_constant_vector_of_bfloat16(single_tile_size, 14.0f);
    } else {
        src_vec = create_random_vector_of_bfloat16(single_tile_size, 100, RND_SRC);
    }

    EnqueueWriteBuffer(cq, src_dram_buffer, src_vec, false);

    /*NOC kernel arg initialization*/
    std::vector<uint32_t> dataflow_args(11 + 8 + 2 * SWING_ALGO_STEPS);  // args + semaphore + NOC partners
    dataflow_args[0] = src_dram_buffer->address();
    dataflow_args[1] = dst_dram_buffer->address();
    dataflow_args[2] = src_dram_noc_x;
    dataflow_args[3] = src_dram_noc_y;
    dataflow_args[4] = dst_dram_noc_x;
    dataflow_args[5] = dst_dram_noc_y;
    dataflow_args[6] = SWING_ALGO_STEPS;
    for (int i = 0; i < 8; i++) {
        dataflow_args[11 + 2 * SWING_ALGO_STEPS + i] = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);
    }

    /*Compute kernel arg initialization*/
    std::vector<uint32_t> compute_args(4);
    compute_args[0] = SWING_ALGO_STEPS;

    /*reused variable initialization*/
    KernelHandle dataflow_kernel;
    KernelHandle compute_kernel;
    CoreCoord logical_core;
    CoreCoord physical_core;
    bool start_direction_SE = true;
    bool horizontal_step, sending_SE;
    uint32_t step_directions = 0b00000;
    int node_position, node_other_position, message_pass_depth, recv_node;

    /*create kernels for each core*/
    for (int core_i = 0; core_i < core_array.size(); core_i++) {
        physical_core = device->worker_core_from_logical_core(core_array[core_i]);
        /* core position*/
        dataflow_args[7] = (uint32_t)physical_core.x;
        dataflow_args[8] = (uint32_t)physical_core.y;
        compute_args[1] = (uint32_t)physical_core.x;
        compute_args[2] = (uint32_t)physical_core.y;

        /*Swing algo partner node calculations*/
        horizontal_step = true;
        message_pass_depth = 1;  // distance of pass
        for (int recdub_step = 0; recdub_step < SWING_ALGO_STEPS; recdub_step += 1) {
            // Gets x and y/y and x coordinates depending on if hztl step
            node_position = horizontal_step ? (int)core_array[core_i].x : (int)core_array[core_i].y;
            node_other_position = !horizontal_step ? (int)core_array[core_i].x : (int)core_array[core_i].y;

            // If this node is sending from SE core on this step
            sending_SE = node_position % (2 * message_pass_depth) < message_pass_depth;
            if (sending_SE) {
                step_directions |= (1 << recdub_step);
            } else {
                step_directions &= ~(1 << recdub_step);
            }
            // Partner node
            recv_node = node_position + (sending_SE ? message_pass_depth : -message_pass_depth);

            // Actual core of comm partner
            logical_core =
                horizontal_step ? CoreCoord{recv_node, node_other_position} : CoreCoord{node_other_position, recv_node};
            physical_core = device->worker_core_from_logical_core(logical_core);
            dataflow_args[11 + 2 * recdub_step] = (uint32_t)physical_core.x;
            dataflow_args[12 + 2 * recdub_step] = (uint32_t)physical_core.y;
            // printf(
            //     "Node %d,%d step %d %d partner %d,%d (or %d, %d)\n",
            //     (int)core_array[core_i].x,
            //     (int)core_array[core_i].y,
            //     recdub_step,
            //     (int)horizontal_step,
            //     (int)logical_core.x,
            //     (int)logical_core.y,
            //     recv_node,
            //     node_other_position);

            !horizontal_step ? message_pass_depth = 2 * message_pass_depth : message_pass_depth;
            horizontal_step = !horizontal_step;
        }

        /*Order of operations (whether each step is NW or SE NOC)*/
        dataflow_args[10] = step_directions;
        compute_args[3] = step_directions;

        /*SE Kernel*/
        dataflow_args[9] = (uint32_t)true;  // this_core_SE
        dataflow_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/recdub_multicore_2D/kernels/"
            "dataflow/dataflow_kernel.cpp",
            core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program, dataflow_kernel, core_array[core_i], dataflow_args);

        /*NW Kernel*/
        dataflow_args[9] = (uint32_t)false;  // this_core_SE
        dataflow_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/recdub_multicore_2D/kernels/"
            "dataflow/dataflow_kernel.cpp",
            core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, dataflow_kernel, core_array[core_i], dataflow_args);

        /* compute kernel */
        compute_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/recdub_multicore_2D/kernels/"
            "compute/compute_kernel.cpp",
            core_array[core_i],
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_args,
            });
        SetRuntimeArgs(program, compute_kernel, core_array[core_i], compute_args);
    }

    EnqueueProgram(cq, program, false);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);
    // printf("Source = %d\n", (int)src_vec[0]);  // 22 = 1102070192
    // Unpack the two bfloat16 values from the packed uint32_t
    int vector_size = single_tile_size / (32 * sizeof(uint32_t));
    // print_vec_of_uint32_as_packed_bfloat16(result_vec, vector_size);
    std::vector<bfloat16> result_vec_b16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    auto two_bfloats = unpack_two_bfloat16_from_uint32(result_vec[0]);

    // Convert the unpacked bfloat16 values back to float for printing
    float first_bfloat_value = two_bfloats.first.to_float();
    float second_bfloat_value = two_bfloats.second.to_float();
    printf("        Result (nocast) = %d, and after casting %d\n", result_vec[0], (int)first_bfloat_value);
    two_bfloats = unpack_two_bfloat16_from_uint32(src_vec[0]);

    // Convert the unpacked bfloat16 values back to float for printing
    first_bfloat_value = two_bfloats.first.to_float();
    second_bfloat_value = two_bfloats.second.to_float();
    first_bfloat_value = first_bfloat_value * TOTAL_NODES;
    second_bfloat_value = second_bfloat_value * TOTAL_NODES;
    uint32_t output =
        pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(first_bfloat_value, second_bfloat_value));
    printf("Expected result (nocast) = %d, and after casting %d\n", output, (int)first_bfloat_value);
    CloseDevice(device);
}

int highest_power_of_two(int value) {
    if (value >= 8) {
        return 8;
    }
    if (value >= 4) {
        return 4;
    }
    if (value >= 2) {
        return 2;
    }
    return 1;
}
