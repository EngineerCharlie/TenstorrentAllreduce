// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
    uint32_t in_arr_size = get_arg_val<uint32_t>(0);
    uint32_t src_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t src_dram_noc_x = get_arg_val<uint32_t>(2);
    uint32_t src_dram_noc_y = get_arg_val<uint32_t>(3);
    uint32_t semaphore_addr_1 = get_semaphore(get_arg_val<uint32_t>(4));
    uint32_t semaphore_addr_2 = get_semaphore(get_arg_val<uint32_t>(5));
    uint32_t num_swing_steps = get_arg_val<uint32_t>(6);
    uint32_t this_core_x = get_arg_val<uint32_t>(7);
    uint32_t this_core_y = get_arg_val<uint32_t>(8);
    uint64_t src_dram_noc_addr = get_noc_addr(src_dram_noc_x, src_dram_noc_y, src_dram_addr);

    constexpr uint32_t cb_id0 = tt::CBIndex::c_0;  // index=0
    constexpr uint32_t cb_id1 = tt::CBIndex::c_1;  // index=0
    uint32_t tile_size = get_tile_size(cb_id0);
    uint32_t recv_addr = get_write_ptr(cb_id0);
    uint32_t local_addr = get_write_ptr(cb_id1);
    uint32_t* recv_array = reinterpret_cast<uint32_t*>(recv_addr);
    uint32_t* local_array = reinterpret_cast<uint32_t*>(local_addr);

    constexpr uint32_t onetile = 1;
    noc_async_read(src_dram_noc_addr, recv_addr, tile_size);
    noc_async_read_barrier();

    uint32_t dst_core_x[num_swing_steps];
    uint32_t dst_core_y[num_swing_steps];
    for (int i = 0; i < (int)num_swing_steps; i++) {
        dst_core_x[i] = get_arg_val<uint32_t>(9 + 2 * i);
        dst_core_y[i] = get_arg_val<uint32_t>(10 + 2 * i);
    }
    volatile tt_l1_ptr uint32_t* semaphore_addr_ptr_1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr_1);
    volatile tt_l1_ptr uint32_t* semaphore_addr_ptr_2 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_addr_2);
    // cb_reserve_back(cb_id0, in_arr_size);  // has to be before get_write_ptr??
    // cb_reserve_back(cb_id1, in_arr_size);  // has to be before get_write_ptr??
    for (uint32_t i = 0; i < in_arr_size; i++) {
        local_array[i] = recv_array[i];
    }
    DPRINT << "Core " << this_core_x << this_core_y << " prevalue: " << local_array[0] << ENDL();
    // DPRINT << "Core " << this_core_x << this_core_y << " prerecvvalue: " << recv_array[0] << ENDL();
    uint64_t dst_noc_semaphore_1;
    uint64_t dst_noc_semaphore_2;
    uint64_t dst_noc_addr;
    for (int i = 0; i < (int)num_swing_steps; i++) {
        dst_noc_semaphore_1 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_addr_1);
        dst_noc_semaphore_2 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_addr_2);
        dst_noc_addr = get_noc_addr(dst_core_x[i], dst_core_y[i], recv_addr);
        noc_semaphore_inc(dst_noc_semaphore_1, this_core_x);
        // DPRINT << "Core " << this_core_x << this_core_y << " wait for " << dst_core_x[i] << dst_core_y[i] << ENDL();

        noc_semaphore_wait(semaphore_addr_ptr_1, dst_core_x[i]);
        noc_semaphore_set(semaphore_addr_ptr_1, 0);

        noc_async_write(local_addr, dst_noc_addr, in_arr_size * sizeof(uint32_t));
        noc_async_write_barrier();
        noc_semaphore_inc(dst_noc_semaphore_2, this_core_x);
        // DPRINT << "Core " << this_core_x << this_core_y << " wait2 for " << dst_core_x[i] << dst_core_y[i] << ENDL();

        noc_semaphore_wait(semaphore_addr_ptr_2, dst_core_x[i]);
        noc_semaphore_set(semaphore_addr_ptr_2, 0);
        // DPRINT << "Core " << this_core_x << this_core_y << " recvvalue: " << recv_array[0] << ENDL();
        for (uint32_t i = 0; i < in_arr_size; i++) {
            local_array[i] += recv_array[i];
        }
        // DPRINT << "Core " << this_core_x << this_core_y << " midvalue: " << local_array[0] << ENDL();
    }

    DPRINT << "Core " << this_core_x << this_core_y << " endvalue: " << local_array[0] << ENDL();

    cb_push_back(cb_id0, onetile);
    cb_push_back(cb_id1, onetile);
}
