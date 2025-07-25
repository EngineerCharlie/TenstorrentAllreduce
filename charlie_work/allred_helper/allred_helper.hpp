#pragma once

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <vector>
#include <cstdint>
#include <cmath>
#include <memory>

using namespace tt;
using namespace tt::tt_metal;

void validate_result_vector(
    const std::vector<uint32_t>& result_vec,
    const std::vector<uint32_t>& src_vec_0,
    const std::vector<uint32_t>& src_vec_1,
    std::size_t num_els,
    float ERROR,
    uint32_t total_nodes);

    
int highest_power_of_two(int);

uint32_t get_SE(int, int);

int get_comm_partner_swing_2D(int, int, bool, int, int);

int get_comm_partner_recdub_2D(int, int, bool, int, uint32_t&, int);

KernelHandle CreateComputeKernel(
    Program&,
    const CoreCoord&,
    const std::vector<uint32_t>&,
    const std::string&);

KernelHandle CreateDataflowKernel(
    Program&,
    const CoreCoord&,
    std::vector<uint32_t>&,
    bool,
    const std::string&);

#ifndef ALLRED_HELPER_HPP
#define ALLRED_HELPER_HPP
class AllredConfig {
public:
    // Public member variables
    bool SWING_VERSION;
    bool RUN_KERNEL;
    int RND_SRC;
    int NUM_TILES;
    int TOTAL_NUM_TILES;
    int ERROR;
    int num_els;
    uint32_t TOTAL_NODES;
    uint32_t SWING_ALGO_STEPS;
    std::vector<CoreCoord> core_array;
    std::shared_ptr<tt::tt_metal::Buffer> src_0_dram_buffer;
    std::shared_ptr<tt::tt_metal::Buffer> src_1_dram_buffer;
    std::shared_ptr<tt::tt_metal::Buffer> dst_dram_buffer;
    std::vector<uint32_t> src_vec_0;
    std::vector<uint32_t> src_vec_1;
    std::vector<uint32_t> result_vec;
    uint32_t src_0_dram_noc_coord = 0;  // src_0_dram_buffer->noc_coordinates();
    uint32_t src_1_dram_noc_coord = 0;  // src_1_dram_buffer->noc_coordinates();
    uint32_t dst_dram_noc_coord = 0;    // dst_dram_buffer->noc_coordinates();
    uint32_t src_0_bank_id = 0;     // src_0_dram_noc_coord.x;
    uint32_t src_1_bank_id = 0;     // src_1_dram_noc_coord.x;
    uint32_t dst_bank_id = 0;       // dst_dram_noc_coord.x;
    uint32_t single_tile_size;

    // Constructor to initialize the setup
    AllredConfig(int argc, 
    char** argv, 
    IDevice* device, 
    CommandQueue& cq, 
    Program& program, 
    CoreRange cores, 
    int SIDE_LENGTH,
    bool large_buffer);

    void RunProgram(CommandQueue& cq, Program& program, IDevice* device) {
        if (RUN_KERNEL) {
            EnqueueProgram(cq, program, false);
            Finish(cq);
            tt_metal::detail::DumpDeviceProfileResults(device);
        }

        /* Read in result into a host vector */
        EnqueueReadBuffer(cq, dst_dram_buffer, result_vec, true);
        validate_result_vector(result_vec, src_vec_0, src_vec_1, num_els, ERROR, TOTAL_NODES);

        CloseDevice(device);
    }
};
#endif // ALLRED_HELPER_HPP
