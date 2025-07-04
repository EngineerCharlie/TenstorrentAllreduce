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
    uint32_t src_0_bank_id = get_arg_val<uint32_t>(2);
    uint32_t src0_dram_noc_y = get_arg_val<uint32_t>(3);  // unused
    uint32_t dst_bank_id = get_arg_val<uint32_t>(4);
    uint32_t dst0_dram_noc_y = get_arg_val<uint32_t>(5);  // unused

    uint32_t algo_steps = get_arg_val<uint32_t>(6);
    uint32_t num_tiles = get_arg_val<uint32_t>(11);

    uint32_t this_core_x = get_arg_val<uint32_t>(7);
    uint32_t this_core_y = get_arg_val<uint32_t>(8);

    bool this_core_SE = (bool)get_arg_val<uint32_t>(9);
    uint32_t packed_direction_bools = get_arg_val<uint32_t>(10);

    uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src_0_bank_id, src0_addr);
    uint64_t dst0_noc_addr = get_noc_addr_from_bank_id<true>(dst_bank_id, dst0_addr);

    // setup circular buffers
    constexpr uint32_t cb_id_compute = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_NW = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_SE = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_local = tt::CBIndex::c_16;

    uint32_t cb_id_this;
    uint32_t cb_id_that;
    if (this_core_SE) {
        cb_id_this = cb_id_SE;
        cb_id_that = cb_id_NW;
    } else {
        cb_id_this = cb_id_NW;
        cb_id_that = cb_id_SE;
    }

    // single-tile ublocks
    uint32_t ublock_size_bytes_semaphore = get_tile_size(cb_id_compute);
    uint32_t ublock_size_bytes_data = get_tile_size(cb_id_local);

    uint32_t l1_write_addr_recv = get_write_ptr(cb_id_recv);
    uint32_t l1_write_addr_local = get_write_ptr(cb_id_local);

    uint32_t* local_array = reinterpret_cast<uint32_t*>(l1_write_addr_local);
    uint32_t* recv_array = reinterpret_cast<uint32_t*>(l1_write_addr_recv);

    // read in partner addresses
    uint32_t dst_core_x[algo_steps];
    uint32_t dst_core_y[algo_steps];

    for (int i = 0; i < (int)algo_steps; i++) {
        dst_core_x[i] = get_arg_val<uint32_t>(12 + 2 * i);
        dst_core_y[i] = get_arg_val<uint32_t>(13 + 2 * i);
    }

    // Read and setup semaphores
    const int num_sem_0 = 6;
    const int num_sem_1 = 8 - num_sem_0;
    uint32_t semaphore_0[num_sem_0];
    volatile tt_l1_ptr uint32_t* semaphore_0_ptr[num_sem_0];
    uint32_t semaphore_1[num_sem_1];
    volatile tt_l1_ptr uint32_t* semaphore_1_ptr[num_sem_1];
    for (int i = 0; i < num_sem_0; i++) {
        semaphore_0[i] = get_semaphore(get_arg_val<uint32_t>(12 + 2 * algo_steps + i));
        semaphore_0_ptr[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_0[i]);
    }

    for (int i = 0; i < num_sem_1; i++) {
        semaphore_1[i] = get_semaphore(get_arg_val<uint32_t>(12 + 2 * algo_steps + num_sem_0 + i));
        semaphore_1_ptr[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_1[i]);
    }

    // read ublocks from src to local
    if (!this_core_SE) {
        cb_reserve_back(cb_id_local, num_tiles);
        noc_async_read(src0_noc_addr, l1_write_addr_local, ublock_size_bytes_data * num_tiles);
        noc_async_read_barrier();
        cb_push_back(cb_id_local, num_tiles);
    }

    uint64_t dst_noc_semaphore_0;
    uint64_t dst_noc_semaphore_1;
    uint64_t dst_noc_addr;
    bool direction_SE;

    // Signal appropriate NOC core to exchange data with other core
    for (uint32_t j = 0; j < 1; j++) {
        DeviceZoneScopedN("ALL_RED_LOOP");
        for (uint32_t i = 0; i < algo_steps; i++) {
            direction_SE = (packed_direction_bools >> i) & 1;  // Extract bit i
            if (this_core_SE == direction_SE) {
                dst_noc_semaphore_0 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_0[i % num_sem_0]);
                dst_noc_semaphore_1 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_1[i % num_sem_1]);
                dst_noc_addr = get_noc_addr(dst_core_x[i], dst_core_y[i], l1_write_addr_recv);

                // await sem from compute then reserve cb
                cb_wait_front(cb_id_this, 1);
                cb_wait_front(cb_id_local, num_tiles);
                // DPRINT << "NOC " << this_core_x << this_core_y << (int)this_core_SE
                //        << " arr4096 post compute: " << recv_array[4095] << ENDL();
                cb_pop_front(cb_id_this, 1);

                // await first sem from comm partner
                noc_semaphore_inc(dst_noc_semaphore_0, 1);
                noc_semaphore_wait(semaphore_0_ptr[i % num_sem_0], 1);
                noc_semaphore_set(semaphore_0_ptr[i % num_sem_0], 0);

                // write local array to com partner
                noc_async_write(l1_write_addr_local, dst_noc_addr, ublock_size_bytes_data * num_tiles);
                noc_async_write_barrier();
                cb_pop_front(cb_id_local, num_tiles);

                // await second sem from comm partner
                noc_semaphore_inc(dst_noc_semaphore_1, 1);
                noc_semaphore_wait(semaphore_1_ptr[i % num_sem_1], 1);
                noc_semaphore_set(semaphore_1_ptr[i % num_sem_1], 0);
                // DPRINT << "NOC " << this_core_x << this_core_y << (int)this_core_SE << " arr4096: " <<
                // recv_array[4096]
                //        << ENDL();
                cb_reserve_back(cb_id_recv, num_tiles);
                cb_push_back(cb_id_recv, num_tiles);
            }
        }
        cb_wait_front(cb_id_this, 1);
        cb_pop_front(cb_id_this, 1);
    }
    if (this_core_SE == direction_SE && this_core_x == 18 && this_core_y == 18) {
        noc_async_write(l1_write_addr_local, dst0_noc_addr, ublock_size_bytes_data * num_tiles);
        noc_async_write_barrier();

        int num_els = ublock_size_bytes_data * num_tiles / sizeof(uint32_t);
        DPRINT << "NOC " << this_core_x << this_core_y << (int)this_core_SE << " sum[0]: " << local_array[0]
               << " and sum[last]" << local_array[num_els - 1] << ENDL();
        DPRINT << "NOC " << this_core_x << this_core_y << (int)this_core_SE << " arr512: " << local_array[512]
               << ENDL();
    }
}
