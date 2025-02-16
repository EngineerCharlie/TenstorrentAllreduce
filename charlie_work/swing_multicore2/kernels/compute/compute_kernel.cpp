// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT

namespace NAMESPACE {
void MAIN {
    uint32_t swing_algo_steps = get_arg_val<uint32_t>(0);
    uint32_t this_core_x = get_arg_val<uint32_t>(1);
    uint32_t this_core_y = get_arg_val<uint32_t>(2);
    bool direction_SE = (bool)get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_compute = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_NW = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_SE = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3;
    constexpr auto cb_id_local = tt::CBIndex::c_16;
    cb_wait_front(cb_id_compute, 1);
    cb_pop_front(cb_id_compute, 1);

    for (uint32_t i = 0; i < swing_algo_steps; i++) {
        if (direction_SE) {
            cb_push_back(cb_id_SE, 1);
        } else {
            cb_push_back(cb_id_NW, 1);
        }
        cb_wait_front(cb_id_compute, 1);
        cb_pop_front(cb_id_compute, 1);
        binary_op_init_common(cb_id_local, cb_id_recv, cb_id_local);
        add_tiles_init();

        // wait for a block of tiles in each of input CBs

        tile_regs_acquire();  // acquire 8 tile registers

        add_tiles(cb_id_local, cb_id_recv, 0, 0, 0);

        tile_regs_commit();  // signal the packer

        tile_regs_wait();  // packer waits here
        pack_tile(0, cb_id_local);
        tile_regs_release();  // packer releases

        direction_SE = !direction_SE;
    }

    DPRINT_MATH(DPRINT << "Compute " << this_core_x << this_core_y << " done " << ENDL());
}
}  // namespace NAMESPACE
