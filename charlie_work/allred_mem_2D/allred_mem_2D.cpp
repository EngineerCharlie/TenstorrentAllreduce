// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <array>
#include <cmath>  // For std::log2
#include <cstdint>
#include "hostdevcommon/profiler_common.h"
#include "allred_helper.hpp"

using namespace tt;
using namespace tt::tt_metal;

int get_comm_partner_swing_2D(int, int, bool, int, int);
int get_comm_partner_recdub_2D(int, int, bool, int, uint32_t&, int);

std::string uint32_to_binary_string(uint32_t value) {
    std::string result(32, '0');
    for (int i = 0; i < 32; i++) {
        result[31 - i] = (value & (1 << i)) ? '1' : '0';
    }
    return result;
}

int main(int argc, char** argv) {
    /* Silicon accelerator setup */
    IDevice* device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    /*Assign input args*/
    bool SWING_VERSION = true;
    if (argc >= 2 && std::stoi(argv[1]) == 1) {
        SWING_VERSION = true;
    } else {
        SWING_VERSION = false;
    }

    bool RUN_KERNEL = false;
    if (argc >= 3 && std::stoi(argv[2]) == 1) {
        RUN_KERNEL = true;
    }

    int SIDE_LENGTH;
    if (argc >= 4) {
        SIDE_LENGTH = highest_power_of_two(std::stoi(argv[3]));
    } else {
        SIDE_LENGTH = 1;
    }
    uint32_t TOTAL_NODES = SIDE_LENGTH * SIDE_LENGTH;

    int RND_SRC = 0;
    if (argc >= 5) {
        RND_SRC = std::stoi(argv[4]);
    }

    int NUM_TILES_PER_NODE = 1;
    if (argc >= 6) {
        NUM_TILES_PER_NODE = std::stoi(argv[5]);
    }
    int NUM_TILES = NUM_TILES_PER_NODE * TOTAL_NODES;

    int ERROR = 1;
    if (argc >= 7) {
        ERROR = std::stoi(argv[6]);
    }

    /*Setup core array (full grid or subsection)*/
    uint32_t SWING_ALGO_STEPS = static_cast<uint32_t>(std::log2(TOTAL_NODES));
    CoreCoord start_core = {0, 0};
    CoreCoord end_core = {SIDE_LENGTH - 1, SIDE_LENGTH - 1};
    CoreRange cores(start_core, end_core);

    std::vector<CoreCoord> core_array(TOTAL_NODES);
    for (uint32_t i = 0; i < core_array.size(); i++) {
        core_array[i] = {i % SIDE_LENGTH, i / SIDE_LENGTH};
    }

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    uint32_t num_data_tiles = NUM_TILES;
    uint32_t num_recv_tiles = NUM_TILES;
    constexpr uint32_t num_semaphore_tiles = 1;
    constexpr uint32_t semaphore_tile_size = 32;
    uint32_t single_tile_size = 2048;
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

    /*Setup dram to pass data to/from cores*/
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size * NUM_TILES,
        .page_size = single_tile_size * NUM_TILES,
        .buffer_type = tt_metal::BufferType::DRAM};

    tt_metal::InterleavedBufferConfig common_dram_config{
        .device = device,
        .size = single_tile_size * NUM_TILES * TOTAL_NODES,
        .page_size = single_tile_size * NUM_TILES * TOTAL_NODES,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> src_0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> src_1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> common_dram_buffer = CreateBuffer(common_dram_config);

    auto src_0_dram_noc_coord = 0;   // src_0_dram_buffer->noc_coordinates();
    auto src_1_dram_noc_coord = 0;   // src_1_dram_buffer->noc_coordinates();
    auto common_dram_noc_coord = 0;  // common_dram_buffer->noc_coordinates();
    auto dst_dram_noc_coord = 0;     // dst_dram_buffer->noc_coordinates();
    uint32_t src_0_bank_id = 0;      // src_0_dram_noc_coord.x;
    uint32_t src_1_bank_id = 0;      // src_1_dram_noc_coord.x;
    uint32_t common_bank_id = 0;     // common_dram_noc_coord.x;
    uint32_t dst_bank_id = 0;        // dst_dram_noc_coord.x;
    uint32_t dst_dram_noc_y = 0;     // dst_dram_noc_coord.y;

    /* Create source data and write to DRAM */
    std::vector<uint32_t> src_vec_0;   //(single_tile_size, 14);
    std::vector<uint32_t> src_vec_1;   //(single_tile_size, 14);
    std::vector<uint32_t> result_vec;  //(single_tile_size, 14);
    int num_els = single_tile_size * NUM_TILES / sizeof(uint32_t);
    if (RND_SRC < 0) {
        src_vec_0 = create_constant_vector_of_bfloat16(single_tile_size * num_data_tiles, 1.0f);
        src_vec_1 = src_vec_0;
    } else {
        src_vec_0 = create_random_vector_of_bfloat16(single_tile_size * num_data_tiles, 100, RND_SRC);
        src_vec_1 = create_random_vector_of_bfloat16(single_tile_size * num_data_tiles, 100, RND_SRC + 100);
    }

    EnqueueWriteBuffer(cq, src_0_dram_buffer, src_vec_0, true);
    EnqueueWriteBuffer(cq, src_1_dram_buffer, src_vec_1, true);

    /*NOC kernel arg initialization*/
    std::vector<uint32_t> dataflow_args(17 + 2 * SWING_ALGO_STEPS + 8 + 2 * SWING_ALGO_STEPS);
    /*args:
    0-5 : src + dst dram
    6-8: common dram
    9: num steps
    10-11: core x, y
    12: core i (x+ y*side length)
    13: is_SE
    14: step_directions
    15: num_tiles
    16: tiles_per_node
    17-28: core x, y for each step
    29-36: semaphores for each step
    37-48: block indexes to send at each step
    */
    dataflow_args[1] = dst_dram_buffer->address();
    dataflow_args[4] = dst_bank_id;
    dataflow_args[6] = common_dram_buffer->address();
    dataflow_args[7] = common_bank_id;
    dataflow_args[9] = SWING_ALGO_STEPS;
    dataflow_args[15] = NUM_TILES;
    dataflow_args[16] = NUM_TILES / TOTAL_NODES;  // tiles per node
    for (int i = 0; i < 8; i++) {
        dataflow_args[17 + 2 * SWING_ALGO_STEPS + i] = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);
    }

    /*Compute kernel arg initialization*/
    std::vector<uint32_t> compute_args(7 + 2 * SWING_ALGO_STEPS);
    compute_args[0] = SWING_ALGO_STEPS;
    compute_args[5] = NUM_TILES;
    compute_args[6] = NUM_TILES / TOTAL_NODES;  // tiles per node

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
    for (int core_i = 0; core_i < core_array.size(); core_i++) {
        physical_core = device->worker_core_from_logical_core(core_array[core_i]);
        dataflow_args[10] = (uint32_t)physical_core.x;
        dataflow_args[11] = (uint32_t)physical_core.y;
        dataflow_args[12] = (uint32_t)core_i;  // Added core_i
        compute_args[1] = (uint32_t)physical_core.x;
        compute_args[2] = (uint32_t)physical_core.y;
        compute_args[3] = (uint32_t)core_i;
        if (core_array[core_i].x % 2 == 0) {
            dataflow_args[0] = src_1_dram_buffer->address();
            dataflow_args[2] = src_1_bank_id;
        } else {
            dataflow_args[0] = src_0_dram_buffer->address();
            dataflow_args[2] = src_0_bank_id;
        }

        /* set block indexes to 0 */
        for (int i = 0; i < 2 * SWING_ALGO_STEPS; i++) {
            dataflow_args[25 + 2 * SWING_ALGO_STEPS + i] = 0;
        }
        for (int i = 0; i < 2 * SWING_ALGO_STEPS; i++) {
            compute_args[7 + i] = 0;
        }

        horizontal_step = true;  // Start calcs on hrz step
        if (!SWING_VERSION) {
            /*Recursive doubling algo partner node calculations*/
            message_pass_depth = 1;
            for (int algo_step = 0; algo_step < SWING_ALGO_STEPS; algo_step++) {
                comm_partner_idx = get_comm_partner_recdub_2D(
                    core_i, algo_step, horizontal_step, message_pass_depth, step_directions, SIDE_LENGTH);

                logical_core = core_array[comm_partner_idx];
                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[17 + 2 * algo_step] = (uint32_t)physical_core.x;
                dataflow_args[18 + 2 * algo_step] = (uint32_t)physical_core.y;

                uint32_t* blocks_to_send = &dataflow_args[25 + 2 * SWING_ALGO_STEPS + 2 * algo_step];
                if (comm_partner_idx < 32) {
                    *blocks_to_send = *blocks_to_send | (1 << comm_partner_idx);
                } else {
                    *(blocks_to_send + 1) = *(blocks_to_send + 1) | (1 << (comm_partner_idx - 32));
                }

                uint32_t* blocks_to_recv = &compute_args[7 + 2 * algo_step];
                if (core_i < 32) {
                    *blocks_to_recv = *blocks_to_recv | (1 << core_i);
                } else {
                    *(blocks_to_recv + 1) = *(blocks_to_recv + 1) | (1 << (core_i - 32));
                }

                message_pass_depth = horizontal_step ? message_pass_depth : 2 * message_pass_depth;
                horizontal_step = !horizontal_step;
            }
        } else {
            /*Swing communication partner calculations*/
            for (int algo_step = 0; algo_step < SWING_ALGO_STEPS; algo_step++) {
                comm_partner_idx =
                    get_comm_partner_swing_2D(core_i, algo_step, horizontal_step, SIDE_LENGTH, TOTAL_NODES);

                logical_core = core_array[comm_partner_idx];
                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[17 + 2 * algo_step] = (uint32_t)physical_core.x;
                dataflow_args[18 + 2 * algo_step] = (uint32_t)physical_core.y;

                uint32_t* blocks_to_send = &dataflow_args[25 + 2 * SWING_ALGO_STEPS + 2 * algo_step];
                if (comm_partner_idx < 32) {
                    *blocks_to_send = *blocks_to_send | (1 << comm_partner_idx);
                } else {
                    *(blocks_to_send + 1) = *(blocks_to_send + 1) | (1 << (comm_partner_idx - 32));
                }

                uint32_t* blocks_to_recv = &compute_args[7 + 2 * algo_step];
                if (core_i < 32) {
                    *blocks_to_recv = *blocks_to_recv | (1 << core_i);
                } else {
                    *(blocks_to_recv + 1) = *(blocks_to_recv + 1) | (1 << (core_i - 32));
                }

                horizontal_step = !horizontal_step;
            }
            step_directions = get_SE(core_array[core_i].x, core_array[core_i].y);
        }

        dataflow_args[14] = step_directions;
        compute_args[4] = step_directions;

        /*SE Kernel*/
        dataflow_args[13] = (uint32_t)true;
        dataflow_1_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/allred_mem_2D/kernels/dataflow/"
            "dataflow_kernel.cpp",
            core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
        SetRuntimeArgs(program, dataflow_1_kernel, core_array[core_i], dataflow_args);

        /*NW Kernel*/
        dataflow_args[13] = (uint32_t)false;
        dataflow_0_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/allred_mem_2D/kernels/dataflow/"
            "dataflow_kernel.cpp",
            core_array[core_i],
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});
        SetRuntimeArgs(program, dataflow_0_kernel, core_array[core_i], dataflow_args);

        /* compute kernel */
        compute_kernel = CreateKernel(
            program,
            "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/allred_mem_2D/kernels/compute/"
            "compute_kernel.cpp",
            core_array[core_i],
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = false,
                .math_approx_mode = false,
                .compile_args = compute_args});
        SetRuntimeArgs(program, compute_kernel, core_array[core_i], compute_args);
    }
    if (RUN_KERNEL) {
        EnqueueProgram(cq, program, false);
        Finish(cq);
        // DumpDeviceProfileResults(device, program);
    }
    /* Read in result into a host vector */
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

    validate_result_vector(result_vec, src_vec_0, src_vec_1, num_els, ERROR, TOTAL_NODES);
    CloseDevice(device);
}

