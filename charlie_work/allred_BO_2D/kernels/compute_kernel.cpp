// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

namespace NAMESPACE {
void MAIN {
    uint32_t algo_steps = get_arg_val<uint32_t>(0);
    bool bandwidth_optimal = (bool) get_arg_val<uint32_t>(1);
    // uint32_t this_core_y = get_arg_val<uint32_t>(2);
    // uint32_t packed_bools = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t num_tiles_per_node = get_arg_val<uint32_t>(5);
    uint32_t total_nodes = bandwidth_optimal ? 64 : num_tiles < 64 ? num_tiles : 64;

    // constexpr uint32_t cb_id_compute = tt::CBIndex::c_0;
    // constexpr uint32_t cb_id_NW = tt::CBIndex::c_1;
    // constexpr uint32_t cb_id_SE = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_local = tt::CBIndex::c_16;

    uint64_t block_indexes[algo_steps]; // indexes of blocks to be exchanged

    for (uint32_t i = 0; i < algo_steps; i++) {
        uint64_t low_bits = get_arg_val<uint32_t>(6 + 2 * i);
        uint64_t high_bits = get_arg_val<uint32_t>(7 + 2 * i);
        block_indexes[i] = (high_bits << 32) | low_bits;
    }

    // Initialize the compute cores
    binary_op_init_common(cb_id_local, cb_id_recv, cb_id_local);
    add_tiles_init(cb_id_local, cb_id_recv);
    
    bool recv_block = true;
    for (uint32_t j = 0; j < 1; j++) { // # repeats of algorithm to get accurate timings
        for (uint32_t i = 0; i < algo_steps; i++) {
            // Signal appropriate NOC core to exchange data with other core
            uint32_t reg_index = 0;
            // Doesn't work if n_block is less than total_nodes
            for (uint32_t n_block = 0; n_block < total_nodes; n_block++) {
                if (bandwidth_optimal)
                    recv_block = (block_indexes[i] >> n_block) & 1;  // Extract bit i

                for (uint32_t tile_num = n_block * num_tiles_per_node; tile_num < (n_block + 1) * num_tiles_per_node;
                     tile_num++) {
                    cb_wait_front(cb_id_recv, 1); // Await blocks to be exchanged
                    cb_wait_front(cb_id_local, 1);                   // Unpack
                    if (recv_block) { // Only part that can be skipped without hangs
                        tile_regs_acquire();
                        add_tiles(cb_id_local, cb_id_recv, 0, 0, reg_index);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(reg_index, cb_id_local, tile_num);
                        tile_regs_release();
                    }
                    cb_pop_front(cb_id_recv, 1);
                    cb_pop_front(cb_id_local, 1);
                }
            }
        }
    }
    DPRINT_MATH(DPRINT << "Compute done " << ENDL());
}
}  // namespace NAMESPACE
