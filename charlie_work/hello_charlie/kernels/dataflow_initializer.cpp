// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_dram_addr = get_arg_val<uint32_t>(0);
    uint32_t src_dram_noc_x = get_arg_val<uint32_t>(1);
    uint32_t src_dram_noc_y = get_arg_val<uint32_t>(2);
    uint32_t this_core_x = get_arg_val<uint32_t>(3);
    uint32_t this_core_y = get_arg_val<uint32_t>(4);
    uint32_t in_arr_size = get_arg_val<uint32_t>(5);
    uint64_t src_dram_noc_addr = get_noc_addr(src_dram_noc_x, src_dram_noc_y, src_dram_addr);

    constexpr uint32_t cb_id_local = tt::CBIndex::c_0;  // index=0
    uint32_t ublock_size_local = get_tile_size(cb_id_local);
    uint32_t write_addr_local = get_write_ptr(cb_id_local);

    cb_reserve_back(cb_id_local, 1);
    noc_async_read(src_dram_noc_addr, write_addr_local, ublock_size_local);
    noc_async_read_barrier();
    cb_push_back(cb_id_local, 1);

    uint32_t* local_array = reinterpret_cast<uint32_t*>(write_addr_local);
    DPRINT << "Core " << this_core_x << this_core_y << " read from SRAM " << local_array[0] << ENDL();
}
