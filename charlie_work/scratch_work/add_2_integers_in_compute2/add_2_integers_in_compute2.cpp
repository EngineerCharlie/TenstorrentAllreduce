// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <tt-metalium/host_api.hpp>
#include "tt_metal/impl/device/device.hpp"
#include "tt_metal/common/bfloat16.hpp"

using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    /* Silicon accelerator setup */
    Device* device = CreateDevice(0);

    /* Setup program to execute along with its buffers and kernels to use */
    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();
    constexpr CoreCoord core = {0, 0};

    constexpr uint32_t single_tile_size = 2 * 1024;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size,
        .page_size = single_tile_size,
        .buffer_type = tt_metal::BufferType::DRAM};

    std::shared_ptr<tt::tt_metal::Buffer> src0_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> src1_dram_buffer = CreateBuffer(dram_config);
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer = CreateBuffer(dram_config);

    auto src0_dram_noc_coord = src0_dram_buffer->noc_coordinates();
    auto src1_dram_noc_coord = src1_dram_buffer->noc_coordinates();
    auto dst_dram_noc_coord = dst_dram_buffer->noc_coordinates();
    uint32_t src0_dram_noc_x = src0_dram_noc_coord.x;
    uint32_t src0_dram_noc_y = src0_dram_noc_coord.y;
    uint32_t src1_dram_noc_x = src1_dram_noc_coord.x;
    uint32_t src1_dram_noc_y = src1_dram_noc_coord.y;
    uint32_t dst_dram_noc_x = dst_dram_noc_coord.x;
    uint32_t dst_dram_noc_y = dst_dram_noc_coord.y;

    /* Use L1 circular buffers to set input and output buffers that the compute engine will use */
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 1;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src0_cb_index, single_tile_size);
    CBHandle cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t src1_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_src1_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src1_cb_index, single_tile_size);
    CBHandle cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

    constexpr uint32_t src2_cb_index = CBIndex::c_2;
    CircularBufferConfig cb_src2_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src2_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src2_cb_index, single_tile_size);
    CBHandle cb_src2 = tt_metal::CreateCircularBuffer(program, core, cb_src2_config);

    constexpr uint32_t src3_cb_index = CBIndex::c_3;
    CircularBufferConfig cb_src3_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src3_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src3_cb_index, single_tile_size);
    CBHandle cb_src3 = tt_metal::CreateCircularBuffer(program, core, cb_src3_config);

    constexpr uint32_t src4_cb_index = CBIndex::c_4;
    CircularBufferConfig cb_src4_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src4_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src4_cb_index, single_tile_size);
    CBHandle cb_src4 = tt_metal::CreateCircularBuffer(program, core, cb_src4_config);
    constexpr uint32_t src5_cb_index = CBIndex::c_5;
    CircularBufferConfig cb_src5_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src5_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(src5_cb_index, single_tile_size);
    CBHandle cb_src5 = tt_metal::CreateCircularBuffer(program, core, cb_src5_config);

    constexpr uint32_t output_cb_index = CBIndex::c_16;
    constexpr uint32_t num_output_tiles = 1;
    CircularBufferConfig cb_output_config =
        CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(output_cb_index, single_tile_size);
    CBHandle cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    /* Specify data movement kernels for reading/writing data to/from DRAM */
    KernelHandle binary_reader_kernel_id = CreateKernel(
        program,
        "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/add_2_integers_in_compute2/kernels/"
        "dataflow/reader_binary_1_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/add_2_integers_in_compute2/kernels/"
        "dataflow/writer_1_tile.cpp",
        core,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    /* Set the parameters that the compute kernel will use */
    std::vector<uint32_t> compute_kernel_args = {};

    /* Use the add_tiles operation in the compute kernel */
    KernelHandle eltwise_binary_kernel_id = CreateKernel(
        program,
        "/home/tenstorrent/tt-metal/tt_metal/programming_examples/charlie_work/add_2_integers_in_compute2/kernels/"
        "compute/add_2_tiles.cpp",
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_kernel_args,
        });

    /* Create source data and write to DRAM */
    std::vector<uint32_t> src0_vec;  //(single_tile_size, 14);
    std::vector<uint32_t> src1_vec;  //(single_tile_size, 8);
    src0_vec = create_constant_vector_of_bfloat16(single_tile_size, 14.0f);
    src1_vec = create_constant_vector_of_bfloat16(single_tile_size, 8.0f);

    EnqueueWriteBuffer(cq, src0_dram_buffer, src0_vec, false);
    EnqueueWriteBuffer(cq, src1_dram_buffer, src1_vec, false);

    /* Configure program and runtime kernel arguments, then execute */
    SetRuntimeArgs(
        program,
        binary_reader_kernel_id,
        core,
        {src0_dram_buffer->address(),
         src1_dram_buffer->address(),
         src0_dram_noc_x,
         src0_dram_noc_y,
         src1_dram_noc_x,
         src1_dram_noc_y});
    SetRuntimeArgs(program, eltwise_binary_kernel_id, core, {});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), dst_dram_noc_x, dst_dram_noc_y});

    EnqueueProgram(cq, program, false);
    Finish(cq);

    /* Read in result into a host vector */
    std::vector<uint32_t> result_vec;
    EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);
    printf("Source = %d\n", (int)src0_vec[0]);  // 22 = 1102070192
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
}
