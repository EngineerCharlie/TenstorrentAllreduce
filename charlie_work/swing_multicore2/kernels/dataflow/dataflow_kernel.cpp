// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t src0_dram_noc_x = get_arg_val<uint32_t>(2);
    uint32_t src0_dram_noc_y = get_arg_val<uint32_t>(3);
    uint32_t dst_dram_noc_x = get_arg_val<uint32_t>(4);
    uint32_t dst_dram_noc_y = get_arg_val<uint32_t>(5);

    uint32_t swing_algo_steps = get_arg_val<uint32_t>(6);

    uint32_t semaphore_0 = get_semaphore(get_arg_val<uint32_t>(7));
    uint32_t semaphore_1 = get_semaphore(get_arg_val<uint32_t>(8));

    uint32_t this_core_x = get_arg_val<uint32_t>(9);
    uint32_t this_core_y = get_arg_val<uint32_t>(10);

    bool this_core_SE = (bool)get_arg_val<uint32_t>(11);
    bool direction_SE = (bool)get_arg_val<uint32_t>(12);

    uint64_t src0_noc_addr = get_noc_addr(src0_dram_noc_x, src0_dram_noc_y, src0_addr);
    uint64_t dst_noc_addr = get_noc_addr(dst_dram_noc_x, dst_dram_noc_y, dst_addr);

    constexpr uint32_t cb_id_compute = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_NW = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_SE = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3;
    constexpr auto cb_id_local = tt::CBIndex::c_16;

    uint32_t cb_id_this;
    if (this_core_SE) {
        cb_id_this = cb_id_SE;
    } else {
        cb_id_this = cb_id_NW;
    }

    // single-tile ublocks
    uint32_t ublock_size_bytes_semaphore = get_tile_size(cb_id_compute);
    uint32_t ublock_size_bytes_data = get_tile_size(cb_id_NW);

    uint32_t l1_write_addr_compute = get_write_ptr(cb_id_compute);
    uint32_t l1_write_addr_NW = get_write_ptr(cb_id_NW);
    uint32_t l1_write_addr_SE = get_write_ptr(cb_id_SE);
    uint32_t l1_write_addr_recv = get_write_ptr(cb_id_recv);
    uint32_t l1_write_addr_local = get_write_ptr(cb_id_local);

    uint32_t* local_array = reinterpret_cast<uint32_t*>(l1_write_addr_local);

    // read ublocks from src to local, then push ublocks to compute (unpacker)
    if (!this_core_SE) {
        cb_reserve_back(cb_id_compute, 1);
        noc_async_read(src0_noc_addr, l1_write_addr_local, ublock_size_bytes_data);
        noc_async_read_barrier();
        cb_push_back(cb_id_compute, 1);
    }
    
    //read in swing partner addresses
    uint32_t dst_core_x[swing_algo_steps];
    uint32_t dst_core_y[swing_algo_steps];

    for (int i = 0; i < (int)swing_algo_steps; i++) {
        dst_core_x[i] = get_arg_val<uint32_t>(13 + 2 * i);
        dst_core_y[i] = get_arg_val<uint32_t>(14 + 2 * i);
    }

    for (uint32_t i = 0; i < swing_algo_steps; i++) {
    }

    cb_wait_front(cb_id_this, 1);
    noc_async_write(l1_write_addr_local, dst_noc_addr, ublock_size_bytes_data);
    noc_async_write_barrier();
    cb_pop_front(cb_id_this, 1);
    DPRINT << "Core " << this_core_x << this_core_y << " sum: " << local_array[0] << ENDL();
}
