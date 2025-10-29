// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
    uint32_t this_core_x = get_arg_val<uint32_t>(0);
    uint32_t this_core_y = get_arg_val<uint32_t>(1);
    uint32_t src_core_x = get_arg_val<uint32_t>(2);
    uint32_t src_core_y = get_arg_val<uint32_t>(3);
    uint32_t dst_core_x = get_arg_val<uint32_t>(4);
    uint32_t dst_core_y = get_arg_val<uint32_t>(5);
    uint32_t semaphore_addr = get_semaphore(get_arg_val<uint32_t>(6));
    uint32_t input_arr_size = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_id = tt::CBIndex::c_0;  // index=0
    constexpr uint32_t onetile = 1;
    volatile tt_l1_ptr uint32_t* semaphore_addr_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr);
    cb_reserve_back(cb_id, onetile);  // has to be before get_write_ptr??
    uint32_t tile_size = get_tile_size(cb_id);
    uint32_t l1_addr = get_write_ptr(cb_id);
    uint32_t* in_array = reinterpret_cast<uint32_t*>(l1_addr);
    uint64_t src_noc_addr = get_noc_addr(src_core_x, src_core_y, l1_addr);
    uint64_t dst_noc_addr = get_noc_addr(dst_core_x, dst_core_y, semaphore_addr);
    // Perform asynchronous read and barrier
    noc_semaphore_wait(semaphore_addr_ptr, VALID);
    noc_async_read(src_noc_addr, l1_addr, tile_size);
    noc_async_read_barrier();

    in_array[this_core_x] += 1;

    DPRINT << "Core (" << this_core_x << ", " << this_core_y << ") value: " << in_array[this_core_x] << ENDL();

    cb_push_back(cb_id, onetile);
    noc_semaphore_inc(dst_noc_addr, 1);
}
