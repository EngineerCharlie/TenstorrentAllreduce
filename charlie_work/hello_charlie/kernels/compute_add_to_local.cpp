// SPDX-FileCopyrightText: © 2024 Martin Chang
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "compute_kernel_api.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary.h"

namespace NAMESPACE {
void MAIN {
    bool pass_data_NW = (bool)get_arg_val<uint32_t>(0);
    uint32_t num_swing_steps = get_arg_val<uint32_t>(1);
    uint32_t in_arr_size = get_arg_val<uint32_t>(2);
    uint32_t this_core_x = get_arg_val<uint32_t>(3);
    uint32_t this_core_y = get_arg_val<uint32_t>(4);

    constexpr auto cb_id_local = tt::CBIndex::c_0;
    constexpr auto cb_id_recvSE = tt::CBIndex::c_1;
    constexpr auto cb_id_recvNW = tt::CBIndex::c_2;

    binary_op_init_common(cb_id_local, cb_id_recvSE, cb_id_recvNW);
    add_tiles_init();

    // wait for a block of tiles in each of input CBs
    cb_wait_front(cb_id_local, 1);

    // DPRINT << "Compute " << this_core_x << this_core_y << " prevalue: " << local_array[0] << ENDL();
    for (int i = 0; i < (int)num_swing_steps; i++) {
        pass_data_NW = !pass_data_NW;
        if (pass_data_NW) {
            cb_push_back(cb_id_recvNW, 1);
            // cb_wait_front(cb_id_local, 1);  // wait while data movement kernel gets next tranch of data

            tile_regs_acquire();  // Acquire tile registers for computation

            add_tiles(cb_id_local, cb_id_recvNW, 0, 0, 0);  // Perform addition of local and recvNW data
        } else {
            cb_push_back(cb_id_recvSE, 1);
            // cb_wait_front(cb_id_local, 1);  // wait while data movement kernel gets next tranch of data

            tile_regs_acquire();  // Acquire tile registers for computation

            add_tiles(cb_id_local, cb_id_recvSE, 0, 0, 0);  // Perform addition of local and recvNW data
        }
        tile_regs_commit();         // Commit the computed tile for packing
        tile_regs_wait();           // Ensure packing completes before continuing
        pack_tile(0, cb_id_local);  // Store the computed tile back in recvNW buffer
        tile_regs_release();        // Release tile registers
    }

    // DPRINT << "Compute " << this_core_x << this_core_y << " endvalue: " << local_array[0] << ENDL();
}
}  // namespace NAMESPACE