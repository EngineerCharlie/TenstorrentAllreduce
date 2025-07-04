// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

namespace NAMESPACE {
void MAIN {
    uint32_t algo_steps = get_arg_val<uint32_t>(0);
    uint32_t this_core_x = get_arg_val<uint32_t>(1);
    uint32_t this_core_y = get_arg_val<uint32_t>(2);
    uint32_t packed_bools = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t num_tiles_per_node = get_arg_val<uint32_t>(5);
    uint32_t total_nodes = num_tiles / num_tiles_per_node;
    constexpr uint32_t cb_id_compute = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_NW = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_SE = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_local = tt::CBIndex::c_16;

    uint64_t block_indexes[algo_steps];

    for (uint32_t i = 0; i < algo_steps; i++) {
        uint64_t low_bits = get_arg_val<uint32_t>(6 + 2 * i);
        uint64_t high_bits = get_arg_val<uint32_t>(7 + 2 * i);
        block_indexes[i] = (high_bits << 32) | low_bits;
    }

    cb_wait_front(cb_id_local, num_tiles);

    // Pre-initialize operations
    init_sfpu(cb_id_recv, cb_id_local);
    tile_regs_acquire();
    copy_tile_to_dst_init_short(cb_id_local);
    binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_id_recv);
    tile_regs_release();

    cb_pop_front(cb_id_local, num_tiles);
    bool SE, recv_block;
    for (uint32_t j = 0; j < 1; j++) {
        for (uint32_t i = 0; i < algo_steps; i++) {
            // Signal appropriate NOC core to exchange data with other core
            SE = (packed_bools >> i) & 1;  // Extract bit i

            if (SE) {
                cb_push_back(cb_id_SE, 1);
            } else {
                cb_push_back(cb_id_NW, 1);
            }

            // Await signal from NOC that data is on local memory
            cb_reserve_back(cb_id_local, num_tiles);
            cb_wait_front(cb_id_recv, num_tiles);

            // add vectors
            for (uint32_t n_block = 0; n_block < total_nodes; n_block++) {
                recv_block = (block_indexes[i] >> n_block) & 1;  // Extract bit i
                if (recv_block) {
                    for (uint32_t tile_num = n_block * num_tiles_per_node;
                         tile_num < (n_block + 1) * num_tiles_per_node;
                         tile_num++) {
                        // DPRINT_MATH(DPRINT << "adding tile: " << tile_num << ENDL());
                        tile_regs_acquire();
                        // add_tiles(cb_id_local, cb_id_recv, tile_num, tile_num, tile_num % 8);
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile(tile_num % 8, cb_id_local, tile_num);  // i must be lower than 8
                        tile_regs_release();
                    }
                }
            }

            cb_push_back(cb_id_local, num_tiles);
            cb_pop_front(cb_id_recv, num_tiles);
        }
        cb_push_back(cb_id_SE, 1);
        cb_push_back(cb_id_NW, 1);
    }
    DPRINT_MATH(DPRINT << "Compute " << this_core_x << this_core_y << " done " << ENDL());
}
}  // namespace NAMESPACE