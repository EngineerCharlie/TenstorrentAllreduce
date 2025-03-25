// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    uint32_t cb_wait_tiles = num_tiles;
    if (cb_wait_tiles < 1) {
        cb_wait_tiles = 1;
    }

    constexpr uint32_t cb_id_SE = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_local = tt::CBIndex::c_16;

    cb_wait_front(cb_id_local, 1);

    binary_op_init_common(cb_id_local, cb_id_recv, cb_id_local);
    add_tiles_init();
    cb_wait_front(cb_id_recv, cb_wait_tiles);
    for (uint32_t n_tile = 0; n_tile < num_tiles; n_tile++) {
        tile_regs_acquire();
        add_tiles(cb_id_local, cb_id_recv, 0, n_tile, 0);
        tile_regs_commit();

        tile_regs_wait();
        pack_tile(0, cb_id_local, 0);
        tile_regs_release();
    }
    cb_pop_front(cb_id_recv, cb_wait_tiles);
    cb_pop_front(cb_id_local, 1);

    cb_push_back(cb_id_SE, 1);
}
}  // namespace NAMESPACE
