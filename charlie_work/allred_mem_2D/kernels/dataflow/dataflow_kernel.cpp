// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "third_party/tracy/public/tracy/Tracy.hpp"

void sync_nodes(
    uint32_t,
    bool,
    uint32_t,
    uint32_t*,
    volatile tt_l1_ptr uint32_t**,
    volatile tt_l1_ptr uint32_t**,
    uint32_t*,
    uint32_t*);
void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t dst0_addr = get_arg_val<uint32_t>(1);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(2);
    uint32_t dst0_bank_id = get_arg_val<uint32_t>(4);
    uint32_t common_addr = get_arg_val<uint32_t>(6);
    uint32_t common_bank_id = get_arg_val<uint32_t>(7);

    uint32_t algo_steps = get_arg_val<uint32_t>(9);
    uint32_t this_core_x = get_arg_val<uint32_t>(10);
    uint32_t this_core_y = get_arg_val<uint32_t>(11);
    uint32_t this_core_i = get_arg_val<uint32_t>(12);
    bool this_core_SE = (bool)get_arg_val<uint32_t>(13);
    uint32_t packed_direction_bools = get_arg_val<uint32_t>(14);

    uint32_t num_tiles = get_arg_val<uint32_t>(15);
    uint32_t num_tiles_per_node = get_arg_val<uint32_t>(16);
    uint32_t total_nodes = num_tiles / num_tiles_per_node;
    uint32_t side_length;
    if (total_nodes == 64) {
        side_length = 8;
    } else if (total_nodes == 16) {
        side_length = 4;
    } else if (total_nodes == 4) {
        side_length = 2;
    } else if (total_nodes == 1) {
        side_length = 1;
    }

    uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);

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
    uint32_t total_vector_size = ublock_size_bytes_data * num_tiles;
    uint32_t tile_block_size = ublock_size_bytes_data * num_tiles_per_node;
    uint32_t num_els_per_tile = ublock_size_bytes_data / sizeof(uint32_t);

    uint32_t l1_write_addr_recv = get_write_ptr(cb_id_recv);
    uint32_t l1_write_addr_local = get_write_ptr(cb_id_local);

    uint32_t* local_array = reinterpret_cast<uint32_t*>(l1_write_addr_local);
    uint32_t* recv_array = reinterpret_cast<uint32_t*>(l1_write_addr_recv);

    // read in partner addresses and blocks to send indexes
    uint32_t dst_core_x[algo_steps];
    uint32_t dst_core_y[algo_steps];
    uint64_t block_indexes[algo_steps];

    for (uint32_t i = 0; i < algo_steps; i++) {
        dst_core_x[i] = get_arg_val<uint32_t>(17 + 2 * i);
        dst_core_y[i] = get_arg_val<uint32_t>(18 + 2 * i);
        uint64_t low_bits = get_arg_val<uint32_t>(37 + 2 * i);
        uint64_t high_bits = get_arg_val<uint32_t>(38 + 2 * i);
        block_indexes[i] = (high_bits << 32) | low_bits;
    }

    uint32_t all_core_x[8] = {1, 2, 3, 4, 6, 7, 8, 9};
    uint32_t all_core_y[8] = {1, 2, 3, 4, 5, 7, 8, 9};

    // Read and setup semaphores
    const uint32_t num_sem_0 = 6;
    const uint32_t num_sem_1 = 8 - num_sem_0;
    uint32_t semaphore_0[num_sem_0];
    volatile tt_l1_ptr uint32_t* semaphore_0_ptr[num_sem_0];
    uint32_t semaphore_1[num_sem_1];
    volatile tt_l1_ptr uint32_t* semaphore_1_ptr[num_sem_1];
    for (uint32_t i = 0; i < num_sem_0; i++) {
        semaphore_0[i] = get_semaphore(get_arg_val<uint32_t>(17 + 2 * algo_steps + i));
        semaphore_0_ptr[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_0[i]);
    }

    for (uint32_t i = 0; i < num_sem_1; i++) {
        semaphore_1[i] = get_semaphore(get_arg_val<uint32_t>(17 + 2 * algo_steps + num_sem_0 + i));
        semaphore_1_ptr[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_1[i]);
    }

    // read ublocks from src to local
    if (!this_core_SE) {
        cb_reserve_back(cb_id_local, num_tiles);
        noc_async_read(src0_noc_addr, l1_write_addr_local, total_vector_size);
        noc_async_read_barrier();
        cb_push_back(cb_id_local, num_tiles);
    }

    sync_nodes(
        algo_steps, this_core_SE, num_sem_0, semaphore_0, semaphore_0_ptr, semaphore_1_ptr, dst_core_x, dst_core_y);

    for (uint32_t j = 0; j < 1; j++) {
        DeviceZoneScopedN("ALL_RED_LOOP");
        {
            uint32_t write_offset = total_vector_size * this_core_i;
            uint64_t common_noc_addr = get_noc_addr_from_bank_id<true>(common_bank_id, common_addr + write_offset);
            if (!this_core_SE) {
                noc_async_write(l1_write_addr_local, common_noc_addr, total_vector_size);
                noc_async_write_barrier();
            }

            sync_nodes(
                algo_steps,
                this_core_SE,
                num_sem_0,
                semaphore_0,
                semaphore_0_ptr,
                semaphore_1_ptr,
                dst_core_x,
                dst_core_y);
            for (uint32_t el = 0; el < num_els_per_tile * num_tiles_per_node; el++) {
                local_array[el] = local_array[this_core_i * num_els_per_tile * num_tiles_per_node + el];
            }

            uint32_t i_start, i_end;
            if (this_core_SE) {
                i_start = 0;
                i_end = total_nodes / 2;
            } else {
                i_start = total_nodes / 2;
                i_end = total_nodes;
            }
            for (uint32_t i = i_start; i < i_end; i++) {
                uint32_t read_offset = i * total_vector_size + this_core_i * tile_block_size;  // i * tile_block_size;
                // total_vector_size* i + this_core_i* tile_block_size;
                // if(this_core_SE)
                //     cb_reserve_back(cb_id_recv, num_tiles_per_node);
                common_noc_addr = get_noc_addr_from_bank_id<true>(common_bank_id, common_addr + read_offset);
                noc_async_read(common_noc_addr, l1_write_addr_recv + i * tile_block_size, tile_block_size);

                // if(this_core_SE)
                //     cb_push_back(cb_id_recv, num_tiles_per_node);
            }
            noc_async_read_barrier();
            // DPRINT << " Data read to L1 " << ENDL();
            // if(!this_core_SE)
            cb_push_back(cb_id_recv, num_tiles / 2);

            cb_wait_front(cb_id_this, 1);
            cb_pop_front(cb_id_this, 1);

            // DPRINT << " fin" << ENDL();
        }
    }
    uint32_t num_els = ublock_size_bytes_data * num_tiles / sizeof(uint32_t);

    if (this_core_SE) {
        uint32_t offset = tile_block_size * this_core_i;
        DPRINT << " Num tiles: " << num_tiles << " Num tiles/node: " << num_tiles_per_node << " num_els "
               << num_els_per_tile << ENDL();
        uint64_t dst0_noc_addr = get_noc_addr_from_bank_id<true>(dst0_bank_id, dst0_addr + offset);
        noc_async_write(l1_write_addr_local, dst0_noc_addr, tile_block_size);
        noc_async_write_barrier();
        DPRINT << "NOC sum[first]: " << local_array[num_tiles_per_node * num_els_per_tile * this_core_i]
               << " and sum[last]" << local_array[num_tiles_per_node * num_els_per_tile * (this_core_i + 1) - 1]
               << ENDL();
    }
}

void sync_nodes(
    uint32_t algo_steps,
    bool this_core_SE,
    uint32_t num_sem_0,
    uint32_t* semaphore_0,
    volatile tt_l1_ptr uint32_t** semaphore_0_ptr,
    volatile tt_l1_ptr uint32_t** semaphore_1_ptr,
    uint32_t* dst_core_x,
    uint32_t* dst_core_y) {
    if (!this_core_SE) {
        // NW core to sync with all other NW cores via swing algo
        for (uint32_t i = 0; i < algo_steps; i++) {
            // DPRINT << " step " << i << ENDL();
            uint64_t dst_noc_semaphore_0 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_0[i % num_sem_0]);
            noc_semaphore_inc(dst_noc_semaphore_0, 1);
            noc_semaphore_wait(semaphore_0_ptr[i % num_sem_0], 1);
            noc_semaphore_set(semaphore_0_ptr[i % num_sem_0], 0);
        }
        // NW core syncs with SE core
        noc_semaphore_set(semaphore_1_ptr[0], 1);
        noc_semaphore_wait(semaphore_1_ptr[1], 1);
        noc_semaphore_set(semaphore_1_ptr[1], 0);
    } else {
        // SE core syncs with NW core
        noc_semaphore_set(semaphore_1_ptr[1], 1);
        noc_semaphore_wait(semaphore_1_ptr[0], 1);
        noc_semaphore_set(semaphore_1_ptr[0], 0);
    }
}
