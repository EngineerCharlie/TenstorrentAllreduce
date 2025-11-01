#include "allred_helper.hpp"
#include <iostream>
#include <cmath>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <array>
#include <cstdint>

using namespace tt;
using namespace tt::tt_metal;

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

// Checks result vector to ensure it is correct
void validate_result_vector(
    const std::vector<uint32_t>& result_vec,
    const std::vector<uint32_t>& src_vec_0,
    const std::vector<uint32_t>& src_vec_1,
    size_t num_els,
    float ERROR,
    uint32_t total_nodes) {
    bool all_match = true;
    int num_matches = 0;

    std::vector<bfloat16> result_vec_b16 = unpack_uint32_vec_into_bfloat16_vec(result_vec);
    std::vector<bfloat16> src_vec_0_b16 = unpack_uint32_vec_into_bfloat16_vec(src_vec_0);
    std::vector<bfloat16> src_vec_1_b16 = unpack_uint32_vec_into_bfloat16_vec(src_vec_1);
    std::vector<bfloat16> trgt_vec_b16 = unpack_uint32_vec_into_bfloat16_vec(src_vec_1);  // reused buffer

    int last_matching_index = 0;
    int last_incorrect_index = 0;
    float error = static_cast<float>(ERROR);
    float max_error = 0.0f;
    int max_error_index = 0;
    // string to be added to during for loop
    std::string debug_info = "Mismatch blocks: ";

    for (size_t i = 0; i < num_els * 2; i++) {
        trgt_vec_b16[i] = static_cast<bfloat16>(
            (src_vec_0_b16[i].to_float() + src_vec_1_b16[i].to_float()) * static_cast<float>(total_nodes / 2));

        float actual = result_vec_b16[i].to_float();
        float expected = trgt_vec_b16[i].to_float();
        float diff = std::fabs(actual - expected);
        
        if (all_match && diff > error) {
            printf("Mismatch at index %zu:\n", i);
            printf("  Expected: %d\n", static_cast<int>(expected));
            printf("  Actual  : %d\n", static_cast<int>(actual));
            printf("  Original values: %f %f\n\n", src_vec_0_b16[i].to_float(), src_vec_1_b16[i].to_float());
            all_match = false;
            if (static_cast<int>(i)%1024==0){
                debug_info += std::to_string(static_cast<int>(i)/1024) + " ";
            }

        } else if (diff <= error) {
            last_matching_index = static_cast<int>(i);
            num_matches++;
        } else {
            last_incorrect_index = static_cast<int>(i);
            if (diff > max_error) {
                max_error = diff;
                max_error_index = static_cast<int>(i);
            }
            if (static_cast<int>(i)%1024==0){
                debug_info += std::to_string(static_cast<int>(i)/1024) + " ";
            }
        }
    }

    if (all_match) {
        printf("All values match!\n");
    } else {
        printf("Total matches: %d\n", num_matches);
        printf(
            "Last match at index %d: %d\n\n",
            last_matching_index,
            static_cast<int>(result_vec_b16[last_matching_index].to_float()));
        printf(
            "Last wrong at index %d: %d Shpuld be: %d\n\n",
            last_incorrect_index,
            static_cast<int>(result_vec_b16[last_incorrect_index].to_float()),
            static_cast<int>(trgt_vec_b16[last_incorrect_index].to_float()));

        printf("Result (nocast) = %d, casted = %d\n", result_vec[0], static_cast<int>(result_vec_b16[0].to_float()));

        uint32_t output = pack_two_bfloat16_into_uint32({trgt_vec_b16[0], trgt_vec_b16[1]});
        printf("Expected (nocast) = %d, casted = %d\n", output, static_cast<int>(trgt_vec_b16[0].to_float()));

        printf(
            "Actual last result (nocast) = %d, casted = %d\n",
            static_cast<int>(result_vec[num_els - 1]),
            static_cast<int>(result_vec_b16[2 * num_els - 1].to_float()));

        output = pack_two_bfloat16_into_uint32({trgt_vec_b16[2 * num_els - 2], trgt_vec_b16[2 * num_els - 1]});
        printf(
            "Expected last result (nocast) = %d, casted = %d\n",
            output,
            static_cast<int>(trgt_vec_b16[2 * num_els - 1].to_float()));

        printf("Max error: %f\n", max_error);
        printf(
            "Max error index: %d, values %f vs %f\n",
            max_error_index,
            result_vec_b16[max_error_index].to_float(),
            trgt_vec_b16[max_error_index].to_float());

        if (max_error_index + 10 < static_cast<int>(trgt_vec_b16.size())) {
            printf(
                "Max error index +10: %d, values %f vs %f",
                max_error_index + 10,
                result_vec_b16[max_error_index + 10].to_float(),
                trgt_vec_b16[max_error_index + 10].to_float());
        }
        printf("\n%s\n________________\n", debug_info.c_str());
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

// Returns the pattern of which NoC will be used for each comm step for swing algo
uint32_t get_step_directions(int node_x, int node_y) {
    if (node_x % 2 == 0) {
        return node_y % 2 == 0 ? 0b110011 : 0b011001;
    } else {
        return node_y % 2 == 0 ? 0b100110 : 0b001100;
    }
}

// Returns the 1D index of the communication partner for a given node at a given step
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

// Returns the 1D index of the communication partner for a given node at a given step
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

// Handles all the setting up given a specific config
AllredConfig::AllredConfig(
    int argc,
    char** argv,
    IDevice* device,
    CommandQueue& cq,
    Program& program,
    CoreRange cores,
    int SIDE_LENGTH, 
    bool large_buffer)
{
    // Assign input args
    SWING_VERSION = false;
    if (argc >= 2 && std::stoi(argv[1]) == 1) {
        SWING_VERSION = true;
    }

    RUN_KERNEL = false;
    if (argc >= 3 && std::stoi(argv[2]) == 1) {
        RUN_KERNEL = true;
    }

    RND_SRC = (argc >= 5) ? std::stoi(argv[4]) : 0;

    NUM_TILES = (argc >= 6) ? std::stoi(argv[5]) : 1;
    NUM_TILES = NUM_TILES < 1 ? 1 : NUM_TILES;

    ERROR = (argc >= 7) ? std::stoi(argv[6]) : 1;

    TOTAL_NODES = SIDE_LENGTH * SIDE_LENGTH;

    if (large_buffer) {
        NUM_TILES = NUM_TILES * TOTAL_NODES;
    } else if (NUM_TILES < 64){
        uint32_t power = 1;
        while (power < NUM_TILES) {
            power <<= 1; // multiply by 2
        }
        NUM_TILES = power;
    } else {
        NUM_TILES = ((NUM_TILES + 64 - 1) / 64) * 64; // multiple of 64
    }

    SWING_ALGO_STEPS = static_cast<uint32_t>(std::log2(TOTAL_NODES));

    core_array.resize(TOTAL_NODES);
    for (uint32_t i = 0; i < core_array.size(); i++) {
        core_array[i] = {i % SIDE_LENGTH, i / SIDE_LENGTH};
    }

    constexpr uint32_t num_semaphore_tiles = 1;
    constexpr uint32_t semaphore_tile_size = 1;
    constexpr uint32_t cb_tile_size = 2048;
    constexpr tt::DataFormat data_format = tt::DataFormat::Float16_b;

    uint32_t num_data_tiles = NUM_TILES;
    uint32_t num_recv_tiles = NUM_TILES;

    // Helper lambda for CB creation
    auto create_cb = [&](uint32_t index, uint32_t size, uint32_t page_size) {
        return tt_metal::CreateCircularBuffer(
            program, cores, CircularBufferConfig(size, {{index, data_format}}).set_page_size(index, page_size));
    };

    // Create circular buffers
    CBHandle cb_compute = create_cb(CBIndex::c_0, semaphore_tile_size * num_semaphore_tiles, semaphore_tile_size);
    CBHandle cb_NW = create_cb(CBIndex::c_1, semaphore_tile_size * num_semaphore_tiles, semaphore_tile_size);
    CBHandle cb_SE = create_cb(CBIndex::c_2, semaphore_tile_size * num_semaphore_tiles, semaphore_tile_size);
    CBHandle cb_recv = create_cb(CBIndex::c_3, num_recv_tiles * cb_tile_size, cb_tile_size);
    CBHandle cb_local = create_cb(CBIndex::c_16, num_data_tiles * cb_tile_size, cb_tile_size);

    // DRAM setup
    single_tile_size = 2048;
    tt_metal::InterleavedBufferConfig dram_config{
        .device = device,
        .size = single_tile_size * NUM_TILES,
        .page_size = single_tile_size * NUM_TILES,
        .buffer_type = tt_metal::BufferType::DRAM};

    src_0_dram_buffer = CreateBuffer(dram_config);
    src_1_dram_buffer = CreateBuffer(dram_config);
    dst_dram_buffer = CreateBuffer(dram_config);

    // Create source data and write to DRAM
    num_els = single_tile_size * NUM_TILES / sizeof(uint32_t);
    if (RND_SRC < 0) {
        src_vec_0 = create_constant_vector_of_bfloat16(single_tile_size * NUM_TILES, 1.0f);
        src_vec_1 = src_vec_0;
        result_vec = create_constant_vector_of_bfloat16(single_tile_size * NUM_TILES, 0.0f);
    } else {
        src_vec_0 = create_random_vector_of_bfloat16(single_tile_size * NUM_TILES, 100, RND_SRC);
        src_vec_1 = create_random_vector_of_bfloat16(single_tile_size * NUM_TILES, 100, RND_SRC + 1);
    }

    EnqueueWriteBuffer(cq, src_0_dram_buffer, src_vec_0, true);
    EnqueueWriteBuffer(cq, src_1_dram_buffer, src_vec_1, true);
}

// Sets up the kernel
KernelHandle CreateDataflowKernel(
    Program& program,
    const CoreCoord& core,
    std::vector<uint32_t>& args,
    bool is_SE,
    const std::string& kernel_base_dir)
{
    auto processor = is_SE ? DataMovementProcessor::RISCV_1 : DataMovementProcessor::RISCV_0;
    auto noc       = is_SE ? NOC::RISCV_1_default : NOC::RISCV_0_default;

    std::string kernel_path = OVERRIDE_KERNEL_PREFIX "charlie_work/"
        + kernel_base_dir
        + "/kernels/dataflow_kernel.cpp";

    auto kernel = CreateKernel(
        program,
        kernel_path,
        core,
        DataMovementConfig{.processor = processor, .noc = noc});

    SetRuntimeArgs(program, kernel, core, args);
    return kernel;
}


KernelHandle CreateComputeKernel(
    Program& program,
    const CoreCoord& core,
    const std::vector<uint32_t>& compute_args,
    const std::string& kernel_base_dir)
{
    std::string kernel_path = OVERRIDE_KERNEL_PREFIX "charlie_work/"
        + kernel_base_dir 
        + "/kernels/compute_kernel.cpp";

    auto kernel = CreateKernel(
        program,
        kernel_path,
        core,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = compute_args});

    SetRuntimeArgs(program, kernel, core, compute_args);
    return kernel;
}

