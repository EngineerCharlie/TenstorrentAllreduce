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
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t num_tiles_per_node = get_arg_val<uint32_t>(5);
    uint32_t total_nodes = bandwidth_optimal ? 64 : num_tiles < 64 ? num_tiles : 64;

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
    for (uint32_t j = 0; j < 1; j++) { // This loop simply repeats the algorithm to get accurate timings
        for (uint32_t i = 0; i < algo_steps; i++) {
            uint32_t reg_index = 0;

            // Iterate through each block of tiles
            for (uint32_t n_block = 0; n_block < total_nodes; n_block++) {

                //For the BO version, determine if we need to perform computation on this block of tiles
                //For the LO version, every block is computed
                if (bandwidth_optimal)
                    recv_block = (block_indexes[i] >> n_block) & 1;  // Extract bit i

                //Iterate through each tile in the block
                for (uint32_t tile_num = n_block * num_tiles_per_node; tile_num < (n_block + 1) * num_tiles_per_node;
                     tile_num++) {
                    cb_wait_front(cb_id_recv, 1); // Await blocks to be exchanged
                    cb_wait_front(cb_id_local, 1);                   // Unpack

                    //Perform computation only if this block is marked for computation (BO version)
                    if (recv_block) {
                        tile_regs_acquire();
                        add_tiles(cb_id_local, cb_id_recv, 0, 0, reg_index);
                        tile_regs_commit();
                        tile_regs_wait();
                        pack_tile<true>(reg_index, cb_id_local, tile_num);
                        tile_regs_release();
                    }

                    //Pop the blocks after computation
                    cb_pop_front(cb_id_recv, 1);
                    cb_pop_front(cb_id_local, 1);
                }
            }
        }
    }
    DPRINT_MATH(DPRINT << "Compute done " << ENDL());
}
}  // namespace NAMESPACE
