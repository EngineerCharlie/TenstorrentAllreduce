// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "debug/dprint.h"  // required in all kernels using DPRINT
#include "dataflow_api.h"

void kernel_main() {
    uint32_t input_arg = get_arg_val<uint32_t>(0);
    uint32_t dst_core_x = get_arg_val<uint32_t>(1);
    uint32_t dst_core_y = get_arg_val<uint32_t>(2);
    uint32_t semaphore_addr = get_semaphore(get_arg_val<uint32_t>(3));

    // Nothing to move. Print respond message.
    // Make sure to export TT_METAL_DPRINT_CORES=0,0 before runtime.
    constexpr uint32_t cb_id = tt::CBIndex::c_0;  // index=0
    uint32_t size = get_tile_size(cb_id);
    uint32_t l1_addr = get_write_ptr(cb_id);
    uint64_t dst_sem_addr = get_noc_addr(dst_core_x, dst_core_y, semaphore_addr);
    // uint32_t* l1_ptr = reinterpret_cast<uint32_t*>(l1_addr);
    *(uint32_t*)l1_addr = input_arg;
    DPRINT << "Hello, I'm 0,0 and the input is: " << *(uint32_t*)l1_addr << ENDL();

    // noc_async_write(l1_addr, dst_noc_addr, sizeof(uint32_t));
    *(uint32_t*)l1_addr += 2;

    noc_semaphore_inc(dst_sem_addr, 1);
    DPRINT << "Lead 0 finished with mod input " << *(uint32_t*)l1_addr << ENDL();
}
