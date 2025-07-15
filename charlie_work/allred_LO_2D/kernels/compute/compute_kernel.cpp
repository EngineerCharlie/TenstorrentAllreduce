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
    uint32_t packed_bools = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    constexpr uint32_t cb_id_compute = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_NW = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_SE = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_local = tt::CBIndex::c_16;
    cb_wait_front(cb_id_local, num_tiles);

    binary_op_init_common(cb_id_local, cb_id_recv, cb_id_local);
    add_tiles_init(cb_id_local, cb_id_recv, cb_id_local);

    cb_pop_front(cb_id_local, num_tiles);
    bool SE;
    for (uint32_t j = 0; j < 4; j++) {
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
            for (uint32_t tile_num = 0; tile_num < num_tiles; tile_num++) {
                tile_regs_acquire();
                // TODO: Do 8 tile registers at once
                add_tiles(cb_id_local, cb_id_recv, tile_num, tile_num, tile_num % 8);
                tile_regs_commit();

                tile_regs_wait();
                pack_tile(tile_num % 8, cb_id_local, tile_num);  // i must be lower than 8
                // TODO: Do 8 tile registers at once
                tile_regs_release();
            }

            cb_push_back(cb_id_local, num_tiles);
            cb_pop_front(cb_id_recv, num_tiles);
        }
        cb_push_back(cb_id_SE, 1);
        cb_push_back(cb_id_NW, 1);
    }
    // DPRINT_MATH(DPRINT << "Compute " << this_core_x << this_core_y << " done " << ENDL());
}
}  // namespace NAMESPACE
