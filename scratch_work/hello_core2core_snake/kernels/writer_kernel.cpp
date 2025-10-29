// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
    uint32_t this_core_x = get_arg_val<uint32_t>(0);
    uint32_t dst_core_x = get_arg_val<uint32_t>(1);
    uint32_t dst_core_y = get_arg_val<uint32_t>(2);
    constexpr uint32_t cb_id = tt::CBIndex::c_0;  // index=0
    uint32_t l1_addr = get_write_ptr(cb_id);
    DPRINT << "Writer " << this_core_x << " value b4 change:  " << *(uint32_t*)l1_addr << ENDL();
    uint32_t tile_size = get_tile_size(cb_id);
    constexpr uint32_t onetile = 1;


    // cb_wait_front(cb_id, onetile);  // has to be before get_write_ptr??
    // DPRINT << "Writer " << this_core_x << " value befor chng: " << *(uint32_t*)l1_addr << ENDL();
    // *(uint32_t*)l1_addr = 100 * this_core_x;
    // if (this_core_x == 2) {
    //     *(uint32_t*)l1_addr = 2;
    //     DPRINT << "I'm special ebcause im core 2" << ENDL();
    // }

    DPRINT << "Writer " << this_core_x << " value after chng: " << *(uint32_t*)l1_addr << ENDL();

    // uint32_t dst_core;
    // if (this_core_x == 2) {
    //     if (this_core_x < 4) {
    //         dst_core = this_core_x + 2;
    //     } else {
    //         dst_core = this_core_x + 3;
    //     }
    // }

    uint64_t dst_noc_addr = get_noc_addr(dst_core_x, dst_core_y, l1_addr);
    // DPRINT << "Writer " << this_core_x << " dst addr: " << dst_noc_addr << ENDL();
    // *(uint32_t*)l1_addr += 1;
    // DPRINT << "Value at l1_addr after increment: " << *(uint32_t*)l1_addr << " from core x: " << this_core_x <<
    // ENDL();
    if (this_core_x < 7) {
        noc_async_write(l1_addr, dst_noc_addr, tile_size);
        // noc_async_write_barrier();
        cb_pop_front(cb_id, onetile);
    }
    DPRINT << "Writer " << this_core_x << " finished." << ENDL();
}
