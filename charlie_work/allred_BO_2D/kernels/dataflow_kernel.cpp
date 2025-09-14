// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "third_party/tracy/public/tracy/Tracy.hpp"

void sync_NOC(int cb_id_this, int cb_id_that) {
    cb_reserve_back(cb_id_that, 1);
    cb_push_back(cb_id_that, 1);
    cb_wait_front(cb_id_this, 1);
    cb_pop_front(cb_id_this, 1);
}

bool shouldSendBlock(bool bandwidth_optimal,
                     uint64_t send_block_index,
                     uint32_t n_block,
                     uint32_t num_tiles)
{
    if (bandwidth_optimal) {
        return (send_block_index >> n_block) & 1;
    } else {
        return n_block < num_tiles;
    }
}

void kernel_main() {
    uint32_t src0_addr = get_arg_val<uint32_t>(0);
    uint32_t dst0_addr = get_arg_val<uint32_t>(1);
    uint32_t src0_bank_id = get_arg_val<uint32_t>(2);
    uint32_t print_core = get_arg_val<uint32_t>(3);
    uint32_t dst0_bank_id = get_arg_val<uint32_t>(4);
    uint32_t bandwidth_optimal = (bool) get_arg_val<uint32_t>(5);

    uint32_t algo_steps = get_arg_val<uint32_t>(6);
    uint32_t num_tiles = get_arg_val<uint32_t>(12);
    uint32_t num_tiles_per_node = get_arg_val<uint32_t>(13);
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

    // uint32_t this_core_x = get_arg_val<uint32_t>(7);
    // uint32_t this_core_y = get_arg_val<uint32_t>(8);
    uint32_t this_core_i = get_arg_val<uint32_t>(9);
    bool this_core_SE = (bool)get_arg_val<uint32_t>(10);
    uint32_t packed_direction_bools = get_arg_val<uint32_t>(11);

    uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);

    // setup circular buffers
    // constexpr uint32_t cb_id_compute = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_NW = tt::CBIndex::c_1; // used as semaphore
    constexpr uint32_t cb_id_SE = tt::CBIndex::c_2; // used as semaphore
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3; // recieve buffer
    constexpr uint32_t cb_id_local = tt::CBIndex::c_16; // Local data

    uint32_t cb_id_this; // Represents the semaphore for this core
    uint32_t cb_id_that; // represents the semaphore for the other NoC core
    if (this_core_SE) {
        cb_id_this = cb_id_SE;
        cb_id_that = cb_id_NW;
    } else {
        cb_id_this = cb_id_NW;
        cb_id_that = cb_id_SE;
    }

    // single-tile ublocks
    uint32_t ublock_size_bytes_semaphore = get_tile_size(cb_id_SE);
    uint32_t ublock_size_bytes_data = get_tile_size(cb_id_local);
    uint32_t tile_block_size = ublock_size_bytes_data * num_tiles_per_node;
    uint32_t total_vector_size  = ublock_size_bytes_data*num_tiles;
    // uint32_t num_els = ublock_size_bytes_data * num_tiles / sizeof(uint32_t);

    uint32_t l1_write_addr_recv = get_write_ptr(cb_id_recv);
    uint32_t l1_write_addr_local = get_write_ptr(cb_id_local);

    // uint32_t* local_array = reinterpret_cast<uint32_t*>(l1_write_addr_local);
    // uint32_t* recv_array = reinterpret_cast<uint32_t*>(l1_write_addr_recv);

    // read in partner addresses and blocks to send indexes
    uint32_t dst_core_x[algo_steps];
    uint32_t dst_core_y[algo_steps];
    uint64_t send_block_indexes[algo_steps];
    uint64_t recv_block_indexes[algo_steps];

    for (uint32_t i = 0; i < algo_steps; i++) {
        dst_core_x[i] = get_arg_val<uint32_t>(14 + 2 * i);
        dst_core_y[i] = get_arg_val<uint32_t>(15 + 2 * i);
        uint64_t low_bits = get_arg_val<uint32_t>(22 + 2 * algo_steps + 2 * i);
        uint64_t high_bits = get_arg_val<uint32_t>(23 + 2 * algo_steps + 2 * i);
        send_block_indexes[i] = (high_bits << 32) | low_bits;
        low_bits = get_arg_val<uint32_t>(22 + 4 * algo_steps + 2 * i);
        high_bits = get_arg_val<uint32_t>(23 + 4 * algo_steps + 2 * i);
        recv_block_indexes[i] = (high_bits << 32) | low_bits;
    }

    // Read and setup semaphores
    const uint32_t num_sem_0 = 6;
    const uint32_t num_sem_1 = 8 - num_sem_0;

    uint32_t semaphore_0[num_sem_0];
    volatile tt_l1_ptr uint32_t* semaphore_0_ptr[num_sem_0];
    for (uint32_t i = 0; i < num_sem_0; i++) {
        semaphore_0[i] = get_semaphore(get_arg_val<uint32_t>(14 + 2 * algo_steps + i));
        semaphore_0_ptr[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_0[i]);
    }

    uint32_t semaphore_1[num_sem_1];
    volatile tt_l1_ptr uint32_t* semaphore_1_ptr[num_sem_1];
    for (uint32_t i = 0; i < num_sem_1; i++) {
        semaphore_1[i] = get_semaphore(get_arg_val<uint32_t>(14 + 2 * algo_steps + num_sem_0 + i));
        semaphore_1_ptr[i] = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(semaphore_1[i]);
    }

    // read ublocks from src to local
    if (!this_core_SE) {
        noc_async_read(src0_noc_addr, l1_write_addr_local, ublock_size_bytes_data * num_tiles);
        noc_async_read_barrier();
        uint64_t dst0_noc_addr = get_noc_addr_from_bank_id<true>(dst0_bank_id, dst0_addr );
    }

    uint64_t dst_noc_semaphore_0, dst_noc_semaphore_1, dst_noc_addr;
    bool direction_SE, send_block;
    uint32_t num_syncs = 32;  // Peak at 16, 32 causes hanging
    if (num_tiles < 64 && num_tiles > 2) {
        num_syncs = num_tiles / 2;
    } else if (num_tiles == 2) {
        num_syncs = 2;
    } else if (num_tiles <= 1) {
        num_syncs = 1;
    }

    for (uint32_t j = 0; j < 1; j++) { // # repeats of algorithm to get accurate timings
        DeviceZoneScopedN("ALL_RED_LOOP");
        {
        sync_NOC(cb_id_this, cb_id_that);
        noc_semaphore_set(semaphore_1_ptr[0], 0);
        // bandwidth optimal ? reduce scatter : allreduce
        for (uint32_t i = 0; i < algo_steps; i++) {
            direction_SE = (packed_direction_bools >> i) & 1;  // Extract bit i
            sync_NOC(cb_id_this, cb_id_that);
            // DPRINT << "passed cbs" << i << ENDL();

            uint32_t n_block_sync = total_nodes / num_syncs;
            if (this_core_SE == direction_SE) {
                dst_noc_semaphore_0 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_0[i % num_sem_0]);
                dst_noc_semaphore_1 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_1[0]);

                cb_reserve_back(cb_id_local, num_tiles);

                // await first sem from comm partner
                noc_semaphore_inc(dst_noc_semaphore_0, 1);
                int semaphore_wait_count = bandwidth_optimal ? 2 * j + 1 : j + 1;
                noc_semaphore_wait_min(semaphore_0_ptr[i % num_sem_0], semaphore_wait_count);

                for (uint32_t n_block = 0; n_block < total_nodes; ) {
                    send_block = shouldSendBlock(bandwidth_optimal, send_block_indexes[i],
                        n_block, num_tiles);
                    uint32_t blocks_to_send = 0;
                    if (send_block) {
                        uint32_t offset = tile_block_size * n_block;
                        dst_noc_addr = get_noc_addr(dst_core_x[i], dst_core_y[i], l1_write_addr_recv + offset);
                        while (send_block && n_block < total_nodes && n_block < n_block_sync) { // Send contiguous blocks
                            blocks_to_send++;
                            n_block++;
                            send_block = shouldSendBlock(bandwidth_optimal, send_block_indexes[i],
                                n_block, num_tiles);
                        }
                        noc_async_write(l1_write_addr_local + offset, dst_noc_addr, tile_block_size * blocks_to_send);
                    } else {
                        n_block++;
                    }
                    if (n_block >= n_block_sync) {
                        // Periodically (every num_tiles/num_sync blocks) synchronize the nodes and
                        // increment the the circular buffers, allowing computation to proceed
                        noc_async_write_barrier();
                        noc_semaphore_inc(dst_noc_semaphore_1, 1);
                        cb_push_back(cb_id_local, num_tiles / num_syncs);
                        n_block_sync = n_block_sync + (total_nodes / num_syncs);
                    }
                }
            } else {
                cb_reserve_back(cb_id_recv, num_tiles);
                // idle core monitors semaphore and pushes data to compute for greater parallelism
                for (uint32_t n_block = 0; n_block < num_syncs; n_block++) {
                    noc_semaphore_wait_min(semaphore_1_ptr[0], i * num_syncs + n_block + 1);
                    cb_push_back(cb_id_recv, num_tiles / num_syncs);
                }
            }
        }
        if (this_core_SE){
            cb_reserve_back(cb_id_recv, num_tiles);
        }
        if (bandwidth_optimal){
            sync_NOC(cb_id_this, cb_id_that);
            noc_semaphore_set(semaphore_1_ptr[0], 0);
            noc_semaphore_set(semaphore_1_ptr[1], 0);
            //all gather
            for (uint32_t i = algo_steps; i-- > 0; ) {
                direction_SE = (packed_direction_bools >> i) & 1;  // Extract bit i
                sync_NOC(cb_id_this, cb_id_that);

                if (this_core_SE == direction_SE) {
                    dst_noc_semaphore_0 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_0[i % num_sem_0]);
                    dst_noc_semaphore_1 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_1[i % num_sem_1]);

                    // await first sem from comm partner
                    noc_semaphore_inc(dst_noc_semaphore_0, 1);
                    noc_semaphore_wait_min(semaphore_0_ptr[i % num_sem_0], 2 * j + 2);

                    for (uint32_t n_block = 0; n_block < total_nodes; ) {
                        send_block = (recv_block_indexes[i] >> n_block) & 1;  // Extract bit i
                        if (send_block) {
                            uint32_t offset = tile_block_size * n_block;
                            dst_noc_addr = get_noc_addr(dst_core_x[i], dst_core_y[i], l1_write_addr_local + offset);
                            uint32_t tiles_to_send = 0;
                            do {  // Send contiguous blocks
                                tiles_to_send++;
                                n_block++;
                                if (n_block < total_nodes) {
                                    send_block = (recv_block_indexes[i] >> n_block) & 1;
                                } else {
                                    send_block = 0;  // Prevent reading past the mask
                                }
                            } while (send_block && n_block < total_nodes);
                            noc_async_write(l1_write_addr_local + offset, dst_noc_addr, tile_block_size * tiles_to_send);
                        } else {
                            n_block++;
                        }
                    }
                    noc_async_write_barrier();
                    noc_semaphore_inc(dst_noc_semaphore_1, 1);
                    uint32_t sem_value = ((algo_steps - i)+1)/2;
                    noc_semaphore_wait_min(semaphore_1_ptr[i%2], sem_value);
                }
            }
        }
        }
    }
    sync_NOC(cb_id_this, cb_id_that);
    if (this_core_SE == direction_SE && this_core_i == print_core) {
        uint32_t offset = tile_block_size * total_nodes * this_core_i;
        uint64_t dst0_noc_addr = get_noc_addr_from_bank_id<true>(dst0_bank_id, dst0_addr);
        noc_async_write(l1_write_addr_local, dst0_noc_addr, total_vector_size);
        noc_async_write_barrier();
        DPRINT << "NOC SE finished" << ENDL();
    } else {
        DPRINT << "NOC NW finished" << ENDL();
    }
}
