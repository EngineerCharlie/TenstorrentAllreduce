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

using namespace tt;
using namespace tt::tt_metal;

int get_comm_partner_swing_2D(int, int, bool, int, int);
void get_swing_block_comm_indexes(int, int, uint32_t*, bool, int, int);
int get_comm_partner_recdub_2D(int, int, bool, int, uint32_t&, int);
void get_recdub_block_comm_indexes(int, int, uint32_t*, bool, int, int, int, uint32_t&);
uint32_t get_SE(int, int);
int highest_power_of_two(int);

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

    int NUM_TILES = 1;
    if (argc >= 6) {
        NUM_TILES = std::stoi(argv[5]);
    }
    NUM_TILES = NUM_TILES * TOTAL_NODES;

    int TILE_SIZE_FACTOR = 1;
    if (argc >= 7) {
        TILE_SIZE_FACTOR = std::stoi(argv[6]);
        if (TILE_SIZE_FACTOR < 1 || TILE_SIZE_FACTOR > 4) {
            TILE_SIZE_FACTOR = 1;
        }
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
    uint32_t single_tile_size = TILE_SIZE_FACTOR * 2048;
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
    uint32_t src_0_dram_noc_x = 0;   // src_0_dram_noc_coord.x;
    uint32_t src_0_dram_noc_y = 0;   // src_0_dram_noc_coord.y;
    uint32_t src_1_dram_noc_x = 0;   // src_1_dram_noc_coord.x;
    uint32_t src_1_dram_noc_y = 0;   // src_1_dram_noc_coord.y;
    uint32_t common_dram_noc_x = 0;  // common_dram_noc_coord.x;
    uint32_t common_dram_noc_y = 0;  // common_dram_noc_coord.y;
    uint32_t dst_dram_noc_x = 0;     // dst_dram_noc_coord.x;
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
        src_vec_1 = create_random_vector_of_bfloat16(single_tile_size * num_data_tiles, 100, RND_SRC + 1);
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
    dataflow_args[4] = dst_dram_noc_x;
    dataflow_args[5] = dst_dram_noc_y;
    dataflow_args[6] = common_dram_buffer->address();
    dataflow_args[7] = common_dram_noc_x;
    dataflow_args[8] = common_dram_noc_y;
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
            dataflow_args[2] = src_1_dram_noc_x;
            dataflow_args[3] = src_1_dram_noc_y;
        } else {
            dataflow_args[0] = src_0_dram_buffer->address();
            dataflow_args[2] = src_0_dram_noc_x;
            dataflow_args[3] = src_0_dram_noc_y;
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

                get_recdub_block_comm_indexes(
                    comm_partner_idx,
                    algo_step + 1,
                    blocks_to_send,
                    horizontal_step,
                    SIDE_LENGTH,
                    TOTAL_NODES,
                    message_pass_depth,
                    dummy_step_directions);

                get_recdub_block_comm_indexes(
                    core_i,
                    algo_step + 1,
                    blocks_to_recv,
                    horizontal_step,
                    SIDE_LENGTH,
                    TOTAL_NODES,
                    message_pass_depth,
                    dummy_step_directions);
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

                get_swing_block_comm_indexes(
                    comm_partner_idx, algo_step + 1, blocks_to_send, horizontal_step, SIDE_LENGTH, TOTAL_NODES);
                get_swing_block_comm_indexes(
                    core_i, algo_step + 1, blocks_to_recv, horizontal_step, SIDE_LENGTH, TOTAL_NODES);
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

    bool all_match = true;
    int num_matches = 0;

    std::vector<bfloat16> result_vec_b16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    std::vector<bfloat16> src_vec_0_b16 = unpack_uint32_vec_into_bfloat16_vec(src_vec_0);
    std::vector<bfloat16> src_vec_1_b16 = unpack_uint32_vec_into_bfloat16_vec(src_vec_1);
    std::vector<bfloat16> trgt_vec_b16 = unpack_uint32_vec_into_bfloat16_vec(src_vec_1);

    int last_matching_index = 0;
    float error = 32.0;
    for (size_t i = 0; i < num_els * 2; i++) {
        trgt_vec_b16[i] = (bfloat16)(((float)src_vec_0_b16[i].to_float() + (float)src_vec_1_b16[i].to_float()) *
                                     (float)(TOTAL_NODES / 2));
        if (all_match && (result_vec_b16[i].to_float() > trgt_vec_b16[i].to_float() + error ||
                          result_vec_b16[i].to_float() < trgt_vec_b16[i].to_float())) {
            printf("Mismatch at index %zu:\n", i);
            printf("  Expected: %d\n", (int)trgt_vec_b16[i].to_float());
            printf("  Actual  : %d\n", (int)result_vec_b16[i].to_float());
            printf("  Original values: %f %f\n\n", src_vec_0_b16[i].to_float(), src_vec_1_b16[i].to_float());
            all_match = false;
        } else if (trgt_vec_b16[i].to_float() == result_vec_b16[i].to_float()) {
            last_matching_index = i;
            num_matches++;
        }
    }

    if (all_match) {
        printf("All values match!\n");
    } else {
        /* Print actual and expected results*/
        printf("Total matches: %d\n", num_matches);
        printf(
            "Last match at index %d: %d\n\n", last_matching_index, (int)result_vec_b16[last_matching_index].to_float());
        printf(
            "         Result (nocast) = %d, and after casting %d\n", result_vec[0], (int)result_vec_b16[0].to_float());

        uint32_t output =
            pack_two_bfloat16_into_uint32(std::pair<bfloat16, bfloat16>(trgt_vec_b16[0], trgt_vec_b16[1]));
        printf("Expected result (nocast) = %d, and after casting %d\n", output, (int)trgt_vec_b16[0].to_float());

        printf(
            "  Actual last result (nocast) = %d, and after casting %d\n",
            (int)result_vec[num_els - 1],
            (int)result_vec_b16[2 * num_els - 1].to_float());

        output = pack_two_bfloat16_into_uint32(
            std::pair<bfloat16, bfloat16>(trgt_vec_b16[2 * num_els - 2], trgt_vec_b16[2 * num_els - 1]));
        printf(
            "Expected last result (nocast) = %d, and after casting %d\n",
            output,
            (int)trgt_vec_b16[2 * num_els - 1].to_float());
    }
    CloseDevice(device);
}

int get_comm_partner_swing_2D(int node, int step, bool horizontal_step, int SIDE_LENGTH, int TOTAL_NODES) {
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

int get_comm_partner_recdub_2D(
    int node,
    int recdub_step,
    bool horizontal_step,
    int message_pass_depth,
    uint32_t& step_directions,
    int SIDE_LENGTH) {
    int row = node / SIDE_LENGTH;
    int col = node % SIDE_LENGTH;
    int node_position = horizontal_step ? col : row;
    int node_other_position = !horizontal_step ? col : row;

    bool sending_SE = node_position % (2 * message_pass_depth) < message_pass_depth;
    step_directions = sending_SE ? (step_directions | (1 << recdub_step)) : (step_directions & ~(1 << recdub_step));

    int recv_node = node_position + (sending_SE ? message_pass_depth : -message_pass_depth);
    return horizontal_step ? recv_node + node_other_position * SIDE_LENGTH
                           : recv_node * SIDE_LENGTH + node_other_position;
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
