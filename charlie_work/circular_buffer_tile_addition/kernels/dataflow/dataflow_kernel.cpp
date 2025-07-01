// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "third_party/tracy/public/tracy/Tracy.hpp"

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t dst0_addr = get_arg_val<uint32_t>(1);
    uint32_t dram_noc_x = get_arg_val<uint32_t>(2);
    uint32_t dram_noc_y = get_arg_val<uint32_t>(3);

    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    if (num_tiles < 1) {
        num_tiles = 1;
    }

    // setup circular buffers
    constexpr uint32_t cb_id_SE = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_local = tt::CBIndex::c_16;

    // single-tile ublocks
    uint32_t ublock_size_bytes_data = get_tile_size(cb_id_local);
    uint32_t total_vector_size = ublock_size_bytes_data * num_tiles;

    uint32_t l1_write_addr_recv = get_write_ptr(cb_id_recv);
    uint32_t l1_write_addr_local = get_write_ptr(cb_id_local);

    uint64_t src0_noc_addr = get_noc_addr(dram_noc_x, dram_noc_y, src0_addr);
    uint64_t dst0_noc_addr = get_noc_addr(dram_noc_x, dram_noc_y, dst0_addr);

    // Read arrays of 1s to recv buffer and local buffer
    cb_reserve_back(cb_id_local, 1);
    noc_async_read(src0_noc_addr, l1_write_addr_local, ublock_size_bytes_data);
    noc_async_read(src0_noc_addr, l1_write_addr_recv, total_vector_size);
    noc_async_read_barrier();
    cb_push_back(cb_id_local, 1);
    cb_push_back(cb_id_recv, num_tiles);

    //Await compute buffer to finish before writing arrays
    cb_wait_front(cb_id_SE, 1);
    cb_pop_front(cb_id_SE, 1);

    //Write summed tiles back to device memory
    noc_async_write(l1_write_addr_local, dst0_noc_addr, total_vector_size);
    noc_async_write_barrier();
}
