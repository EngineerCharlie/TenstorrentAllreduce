// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
    uint32_t this_core_x = get_arg_val<uint32_t>(0);
    uint32_t src_core_x = get_arg_val<uint32_t>(1);
    uint32_t src_core_y = get_arg_val<uint32_t>(2);
    uint32_t dst_core_x = get_arg_val<uint32_t>(3);
    uint32_t dst_core_y = get_arg_val<uint32_t>(4);
    uint32_t semaphore_addr = get_semaphore(get_arg_val<uint32_t>(5));
    constexpr uint32_t cb_id = tt::CBIndex::c_0;  // index=0
    constexpr uint32_t onetile = 1;
    volatile tt_l1_ptr uint32_t* semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    cb_reserve_back(cb_id, onetile);  // has to be before get_write_ptr??
    uint32_t tile_size = get_tile_size(cb_id);
    uint32_t l1_addr = get_write_ptr(cb_id);
    uint64_t src_noc_addr = get_noc_addr(src_core_x, src_core_y, l1_addr);
    uint64_t dst_noc_addr = get_noc_addr(dst_core_x, dst_core_y, semaphore_addr);
    // Perform asynchronous read and barrier
    uint32_t addition = 1;
    for (uint32_t i = 1; i < this_core_x; i++) {
        addition *= 10;
    }
    noc_semaphore_wait(semaphore_addr_ptr, VALID);
    noc_async_read(src_noc_addr, l1_addr, tile_size);
    noc_async_read_barrier();

    *(uint32_t*)l1_addr += addition;

    DPRINT << "Core " << this_core_x << " value: " << *(uint32_t*)l1_addr << ENDL();

    cb_push_back(cb_id, onetile);
    noc_semaphore_inc(dst_noc_addr, 1);
}
