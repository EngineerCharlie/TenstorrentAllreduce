// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"
#include <array>
#include <cmath>  // For std::log2
#include <cstdint>
#include "hostdevcommon/profiler_common.h"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    /* Silicon accelerator setup */
    Device* device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    /*Assign input args*/
    int NUM_TILES = 1;
    if (argc >= 2) {
        NUM_TILES = std::stoi(argv[1]);
        if (NUM_TILES < 1) {
            NUM_TILES = 0;
        }
    }

    int MIN_MEMORY_TILES = NUM_TILES;
    if (MIN_MEMORY_TILES < 1) {
        MIN_MEMORY_TILES = 1;
    }

    /*Setup core array (full grid or subsection)*/
    CoreCoord start_core = {0, 0};
    CoreRange cores(start_core, start_core);

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    constexpr uint32_t num_semaphore_tiles = 1;
    constexpr uint32_t semaphore_tile_size = 32;
    uint32_t single_tile_size = 2048;
    constexpr tt::DataFormat data_format = tt::DataFormat::Float16_b;

    constexpr uint32_t cb_index_SE = CBIndex::c_2;
    CircularBufferConfig cb_config_SE =
        CircularBufferConfig(semaphore_tile_size * num_semaphore_tiles, {{cb_index_SE, data_format}})
            .set_page_size(cb_index_SE, semaphore_tile_size);
    CBHandle cb_SE = tt_metal::CreateCircularBuffer(program, cores, cb_config_SE);

    constexpr uint32_t cb_index_recv = CBIndex::c_3;
    CircularBufferConfig cb_config_recv =
        CircularBufferConfig(MIN_MEMORY_TILES * single_tile_size, {{cb_index_recv, data_format}})
            .set_page_size(cb_index_recv, single_tile_size);
    CBHandle cb_recv = tt_metal::CreateCircularBuffer(program, cores, cb_config_recv);

    constexpr uint32_t cb_index_local = CBIndex::c_16;
    CircularBufferConfig cb_config_local = CircularBufferConfig(single_tile_size, {{cb_index_local, data_format}})
                                               .set_page_size(cb_index_local, single_tile_size);
    CBHandle cb_local = tt_metal::CreateCircularBuffer(program, cores, cb_config_local);

    /*Setup dram to pass data to/from cores*/
    tt_metal::InterleavedBufferConfig src_dram_config{
        .device = device,
        .size = single_tile_size * MIN_MEMORY_TILES,
        .page_size = single_tile_size * MIN_MEMORY_TILES,
        .buffer_type = tt_metal::BufferType::DRAM};

    /*Setup dram to pass data to/from cores*/
    tt_metal::InterleavedBufferConfig dst_dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> src_dram_buffer = CreateBuffer(src_dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dst_dram_config);

    auto src_0_dram_noc_coord = src_dram_buffer->noc_coordinates();
    auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();

    uint32_t dram_noc_x = src_0_dram_noc_coord.x;
    uint32_t dram_noc_y = src_0_dram_noc_coord.y;

    /* Create source data and write to DRAM */
    std::vector<uint32_t> src_vec;     //(single_tile_size, 14);
    std::vector<uint32_t> result_vec;  //(single_tile_size, 14);
    std::vector<uint32_t> trgt_vec;    //(single_tile_size, 14);
    int num_els = single_tile_size / sizeof(uint32_t);

    src_vec = create_constant_vector_of_bfloat16(single_tile_size * MIN_MEMORY_TILES, 1.0f);
    trgt_vec = create_constant_vector_of_bfloat16(single_tile_size, (float)(NUM_TILES + 1));

    EnqueueWriteBuffer(cq, src_dram_buffer, src_vec, true);

    /*NOC kernel arg initialization*/
    std::vector<uint32_t> dataflow_args(5);
    dataflow_args[0] = src_dram_buffer->address();
    dataflow_args[1] = dst_dram_buffer->address();
    dataflow_args[2] = dram_noc_x;
    dataflow_args[3] = dram_noc_y;
    dataflow_args[4] = NUM_TILES;

    /*Compute kernel arg initialization*/
    std::vector<uint32_t> compute_args(1);
    compute_args[0] = NUM_TILES;

    /*reused variable initialization*/
    KernelHandle dataflow_kernel;
    KernelHandle compute_kernel;

    /*create kernels for each core*/
    /*SE Kernel*/
    dataflow_kernel = CreateKernel(
        program,
        "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/circular_buffer_tile_addition/"
        "kernels/dataflow/"
        "dataflow_kernel.cpp",
        start_core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});
    SetRuntimeArgs(program, dataflow_kernel, start_core, dataflow_args);

    /* compute kernel */
    compute_kernel = CreateKernel(
        program,
        "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/circular_buffer_tile_addition/"
        "kernels/compute/"
        "compute_kernel.cpp",
        start_core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_args});
    SetRuntimeArgs(program, compute_kernel, start_core, compute_args);

    EnqueueProgram(cq, program, false);
    Finish(cq);
    DumpDeviceProfileResults(device, program);

    /* RESULTS EVALUATION */
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);

    bool all_match = true;
    int num_matches = 0;

    std::vector<bfloat16> result_vec_b16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    std::vector<bfloat16> src_vec_b16 = unpack_uint32_vec_into_bfloat16_vec(src_vec);
    std::vector<bfloat16> trgt_vec_b16 = unpack_uint32_vec_into_bfloat16_vec(trgt_vec);

    int last_matching_index = 0;
    float error = 32.0;
    for (size_t i = 0; i < num_els * 2; i++) {
        if (all_match && (result_vec_b16[i].to_float() > trgt_vec_b16[i].to_float() + error ||
                          result_vec_b16[i].to_float() < trgt_vec_b16[i].to_float())) {
            printf("Mismatch at index %zu:\n", i);
            printf("  Expected: %d\n", (int)trgt_vec_b16[i].to_float());
            printf("  Actual  : %d\n", (int)result_vec_b16[i].to_float());
            printf("  Original values: %f\n\n", src_vec_b16[i].to_float());
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
