// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
    uint32_t this_core_x = get_arg_val<uint32_t>(0);
    constexpr uint32_t cb_id = tt::CBIndex::c_0;  // index=0
    uint32_t tile_size = get_tile_size(cb_id);
    constexpr uint32_t onetile = 1;

    cb_wait_front(cb_id, onetile);  // has to be before get_write_ptr??
    uint32_t l1_addr = get_write_ptr(cb_id);
    *(uint32_t*)l1_addr += 1;

    uint64_t dst_noc_addr = get_noc_addr(1, 0, l1_addr);
    DPRINT << "Value at l1_addr after read: " << *(uint32_t*)l1_addr << ENDL();
    // *(uint32_t*)l1_addr += 1;
    // DPRINT << "Value at l1_addr after increment: " << *(uint32_t*)l1_addr << " from core x: " << this_core_x <<
    // ENDL();
    noc_async_write(l1_addr, dst_noc_addr, tile_size);
    noc_async_write_barrier();
    cb_pop_front(cb_id, onetile);
}
