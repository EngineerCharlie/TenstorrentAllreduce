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
    uint32_t this_core_x = get_arg_val<uint32_t>(1);
    uint32_t this_core_y = get_arg_val<uint32_t>(2);
    uint32_t this_core_i = get_arg_val<uint32_t>(3);
    uint32_t packed_bools = get_arg_val<uint32_t>(4);
    uint32_t num_tiles = get_arg_val<uint32_t>(5);
    uint32_t num_tiles_per_node = get_arg_val<uint32_t>(6);
    uint32_t total_nodes = num_tiles / num_tiles_per_node;
    constexpr uint32_t cb_id_compute = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_NW = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_SE = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_local = tt::CBIndex::c_16;

    uint64_t block_indexes[algo_steps];

    for (uint32_t i = 0; i < algo_steps; i++) {
        uint64_t low_bits = get_arg_val<uint32_t>(7 + 2 * i);
        uint64_t high_bits = get_arg_val<uint32_t>(8 + 2 * i);
        block_indexes[i] = (high_bits << 32) | low_bits;
    }

    cb_wait_front(cb_id_local, num_tiles);

    binary_op_init_common(cb_id_local, cb_id_recv, cb_id_local);
    add_tiles_init(cb_id_local, cb_id_recv, cb_id_local);

    cb_pop_front(cb_id_local, num_tiles);
    cb_reserve_back(cb_id_local, num_tiles);
    bool SE, recv_block;
    uint32_t offset = this_core_i * num_tiles_per_node;
    for (uint32_t j = 0; j < 1; j++) {
        // for (uint32_t n_tile = 0; n_tile < num_tiles; n_tile++) {
        //     tile_regs_acquire();
        //     tile_regs_wait();
        //     copy_tile_to_dst_init_short(cb_id_local);
        //     copy_tile(cb_id_local, n_tile % num_tiles_per_node, n_tile % num_tiles_per_node);
        //     cb_wait_front(cb_id_recv, 1);
        //     if (n_tile >= offset && n_tile < offset + num_tiles_per_node) {
        //         cb_pop_front(cb_id_recv, 1);
        //         continue;  // Skip tiles that are this core's responsibility
        //     }
        //     binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
        //         cb_id_recv);
        //     binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
        //         cb_id_recv, 0, n_tile % num_tiles_per_node);
        //     cb_pop_front(cb_id_recv, 1);
        //     tile_regs_commit();

        //     pack_tile(n_tile % num_tiles_per_node, cb_id_local, n_tile % num_tiles_per_node);
        //     tile_regs_release();
        // }

        tile_regs_acquire();
        tile_regs_wait();
        copy_tile_to_dst_init_short(cb_id_local);
        for (uint32_t n_tile = 0; n_tile < num_tiles_per_node; n_tile++) {
            copy_tile(cb_id_local, n_tile, n_tile);
        }
        
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_id_recv);

        for (uint32_t n_tile = 0; n_tile < num_tiles; n_tile++) {
            cb_wait_front(cb_id_recv, 1);
            if (n_tile >= offset && n_tile < offset + num_tiles_per_node) {
                cb_pop_front(cb_id_recv, 1);
                continue;  // Skip tiles that are this core's responsibility
            }
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCA>(
                cb_id_recv, 0, n_tile % num_tiles_per_node);
            cb_pop_front(cb_id_recv, 1);
        }
        tile_regs_commit();

        for (uint32_t n_tile = 0; n_tile < num_tiles_per_node; n_tile++) {
            pack_tile(n_tile, cb_id_local, n_tile);
        }
        tile_regs_release();
        cb_push_back(cb_id_local, num_tiles);
        cb_push_back(cb_id_SE, 1);
        cb_push_back(cb_id_NW, 1);
    }
    // DPRINT_MATH(DPRINT << "Compute done " << ENDL());
}
}  // namespace NAMESPACE
