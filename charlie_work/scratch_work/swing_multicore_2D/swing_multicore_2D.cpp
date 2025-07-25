// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/host_api.hpp>
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include <array>
#include <cmath>  // For std::log2
#include <cstdint>

using namespace tt;
using namespace tt::tt_metal;

int ceil_div(int, int);
int get_comm_partner_2D(int, int, bool, int, int);
uint32_t get_SE(int, int);
int highest_power_of_two(int);

int main(int argc, char** argv) {
    /* Silicon accelerator setup */
    Device* device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    bool RUN_KERNEL = false;
    if (argc >= 2 && std::stoi(argv[1]) == 1) {
        RUN_KERNEL = true;
    }
    int SIDE_LENGTH;
    if (argc >= 3) {
        SIDE_LENGTH = highest_power_of_two(std::stoi(argv[2]));
    } else {
        SIDE_LENGTH = 1;
    }
    int RND_SRC = 0;
    if (argc >= 4) {
        RND_SRC = std::stoi(argv[3]);
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
    constexpr uint32_t single_tile_size = 1 * 2048;
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
    constexpr uint32_t num_data_tiles = 1;
    constexpr uint32_t num_recv_tiles = 1;
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
        CircularBufferConfig(num_recv_tiles * single_tile_size, {{cb_index_recv, data_format}})
            .set_page_size(cb_index_recv, single_tile_size);
    CBHandle cb_recv = tt_metal::CreateCircularBuffer(program, cores, cb_config_recv);

    constexpr uint32_t cb_index_local = CBIndex::c_16;
    CircularBufferConfig cb_config_local =
        CircularBufferConfig(num_data_tiles * single_tile_size, {{cb_index_local, data_format}})
            .set_page_size(cb_index_local, single_tile_size);
    CBHandle cb_local = tt_metal::CreateCircularBuffer(program, cores, cb_config_local);

    /* Create source data and write to DRAM */
    std::vector<uint32_t> src_vec;  //(single_tile_size, 14);
    if (RND_SRC < 0) {
        src_vec = create_constant_vector_of_bfloat16(single_tile_size * num_data_tiles, 1.0f);
    } else {
        src_vec = create_random_vector_of_bfloat16(single_tile_size * num_data_tiles, 100, RND_SRC);
    }

    std::vector<uint32_t> result_vec;
    result_vec = create_constant_vector_of_bfloat16(single_tile_size * num_data_tiles, 0.0f);

    EnqueueWriteBuffer(cq, src_dram_buffer, src_vec, true);
    EnqueueWriteBuffer(cq, dst_dram_buffer, result_vec, true);

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
    bool horizontal_step;

    /*create kernels for each core*/
    for (int core_i = 0; core_i < core_array.size(); core_i++) {
        physical_core = device->worker_core_from_logical_core(core_array[core_i]);
        /* core position*/
        dataflow_args[7] = (uint32_t)physical_core.x;
        dataflow_args[8] = (uint32_t)physical_core.y;
        compute_args[1] = (uint32_t)physical_core.x;
        compute_args[2] = (uint32_t)physical_core.y;

        /*Order of operations (whether each step is NW or SE NOC)*/
        dataflow_args[10] = get_SE(core_array[core_i].x, core_array[core_i].y);
        compute_args[3] = get_SE(core_array[core_i].x, core_array[core_i].y);

        dataflow_args[9] = (uint32_t)true;  // this_core_SE

        /*Swing algo partner node calculations*/
        horizontal_step = true;
        // printf("Node %d,%d communicates with:\n", (int)core_array[core_i].x, (int)core_array[core_i].y);
        for (int swing_step = 0; swing_step < SWING_ALGO_STEPS; swing_step += 1) {
            int comm_partner_idx = get_comm_partner_2D(core_i, swing_step, horizontal_step, SIDE_LENGTH, TOTAL_NODES);
            logical_core = core_array[comm_partner_idx];
            physical_core = device->worker_core_from_logical_core(logical_core);
            dataflow_args[11 + 2 * swing_step] = (uint32_t)physical_core.x;
            dataflow_args[12 + 2 * swing_step] = (uint32_t)physical_core.y;
            horizontal_step = !horizontal_step;
            // printf("Step %d - %d %d\n", swing_step, (int)logical_core.x, (int)logical_core.y);
        }

        /*SE Kernel*/
        dataflow_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore_2D/kernels/"
            "dataflow/dataflow_kernel.cpp",
            core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program, dataflow_kernel, core_array[core_i], dataflow_args);

        /*NW Kernel*/
        dataflow_args[9] = (uint32_t)false;  // this_core_SE
        dataflow_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore_2D/kernels/"
            "dataflow/dataflow_kernel.cpp",
            core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, dataflow_kernel, core_array[core_i], dataflow_args);

        /* compute kernel */
        compute_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/swing_multicore_2D/kernels/"
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
    if (RUN_KERNEL) {
        EnqueueProgram(cq, program, false);
        Finish(cq);
    }
    /* Read in result into a host vector */
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);
    // printf("Source = %d\n", (int)src_vec[0]);  // 22 = 1102070192
    // Unpack the two bfloat16 values from the packed uint32_t
    int vector_size = single_tile_size / (32 * sizeof(uint32_t));
    // print_vec_of_uint32_as_packed_bfloat16(result_vec, vector_size);
    std::vector<bfloat16> result_vec_b16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    // for (int i = 0; i < vector_size; i += 10) {
    //     auto two_bfloats = unpack_two_bfloat16_from_uint32(result_vec[i]);
    //     // Convert the unpacked bfloat16 values back to float for printing
    //     bfloat16 first_bfloat_value = two_bfloats.first.to_float();
    //     bfloat16 second_bfloat_value = two_bfloats.second.to_float();
    //     printf("First bfloat to int = %d\n", (int)first_bfloat_value);  // 22 = 1102070192
    // }
    auto two_bfloats = unpack_two_bfloat16_from_uint32(result_vec[0]);

    // Convert the unpacked bfloat16 values back to float for printing
    float first_bfloat_value = two_bfloats.first.to_float();
    float second_bfloat_value = two_bfloats.second.to_float();
    printf("         Result (nocast) = %d, and after casting %d\n", result_vec[0], (int)first_bfloat_value);
    two_bfloats = unpack_two_bfloat16_from_uint32(src_vec[0]);

    // Convert the unpacked bfloat16 values back to float for printing
    first_bfloat_value = two_bfloats.first.to_float();
    second_bfloat_value = two_bfloats.second.to_float();
    for (int i = 0; i < SWING_ALGO_STEPS; i++) {
        first_bfloat_value = first_bfloat_value * 2;
        second_bfloat_value = second_bfloat_value * 2;
        uint32_t output =
            pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(first_bfloat_value, second_bfloat_value));
        printf("Expected result (nocast) = %d, and after casting %d\n", output, (int)first_bfloat_value);
    }
    CloseDevice(device);
}

int ceil_div(int a, int b) { return (a + b - 1) / b; }

int get_comm_partner_2D(int node, int step, bool horizontal_step, int SIDE_LENGTH, int TOTAL_NODES) {
    int row = node / SIDE_LENGTH;
    int col = node % SIDE_LENGTH;
    step = step / 2;

    // straight line distnce
    int dist = (int)((1 - (int)pow(-2, step + 1)) / 3);

    int comm_partner;
    if (horizontal_step) {
        comm_partner = (node % 2 == 0) ? (node + dist) : (node - dist);  // can return -ve number
        if (comm_partner / SIDE_LENGTH < row || comm_partner < 0) {
            comm_partner += SIDE_LENGTH;
        } else if (comm_partner / SIDE_LENGTH > row) {
            comm_partner -= SIDE_LENGTH;
        }
    } else {
        comm_partner = (row % 2 == 0) ? (node + SIDE_LENGTH * dist) : (node - SIDE_LENGTH * dist);
        if (comm_partner < 0) {
            comm_partner += TOTAL_NODES;
        } else if (comm_partner >= TOTAL_NODES) {
            comm_partner -= TOTAL_NODES;
        }
    }
    return comm_partner;  // will  loop round  to  always be in  range
}

// Returns a uint32_t where bits 0-5 store boolean direction_SE
uint32_t get_SE(int node_x, int node_y) {
    if (node_x % 2 == 0) {
        if (node_y % 2 == 0) {  // node 0,0
            return 0b110011;    // Binary: 110011 (true, true, false, false, true, true)
        } else {                // node 0,1
            return 0b011001;    // Binary: 100110 (true, false, false, true, true, false)
        }
    } else {  // node 1,0
        if (node_y % 2 == 0) {
            return 0b100110;  // Binary: 011001 (false, true, true, false, false, true)
        } else {              // node 1,1
            return 0b001100;  // Binary: 001100 (false, false, true, true, false, false)
        }
    }
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
