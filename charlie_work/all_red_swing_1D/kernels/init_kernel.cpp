// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
    uint32_t array_size = get_arg_val<uint32_t>(0);
    uint32_t src_dram_addr = get_arg_val<uint32_t>(1);
    uint32_t src_dram_noc_x = get_arg_val<uint32_t>(2);
    uint32_t src_dram_noc_y = get_arg_val<uint32_t>(3);

    uint64_t src_dram_noc_addr = get_noc_addr(src_dram_noc_x, src_dram_noc_y, src_dram_addr);

    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.
    constexpr uint32_t cb_id0 = tt::CBIndex::c_0;  // index=0
    uint32_t size = get_tile_size(cb_id0);
    uint32_t l1_addr = get_write_ptr(cb_id0);
    constexpr uint32_t onetile = 1;
    noc_async_read(src_dram_noc_addr, l1_addr, size);
    noc_async_read_barrier();
    DPRINT << "Init kernel input address is: " << (uint32_t)l1_addr << ENDL();
    DPRINT << "Init kernel input is: " << *(uint32_t*)l1_addr << ENDL();
}
