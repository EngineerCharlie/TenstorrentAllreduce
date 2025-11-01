// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "third_party/tracy/public/tracy/Tracy.hpp"

//Function to synchronize two NOC cores using their circular buffers
void sync_NOC(int cb_id_this, int cb_id_that) {
    cb_reserve_back(cb_id_that, 1);
    cb_push_back(cb_id_that, 1);
    cb_wait_front(cb_id_this, 1);
    cb_pop_front(cb_id_this, 1);
}

// Returns true if this core should send its block in this iteration
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
    uint32_t src0_addr = get_arg_val<uint32_t>(0); // Where to read from shared mem
    uint32_t dst0_addr = get_arg_val<uint32_t>(1); // Where to write to shared mem
    uint32_t src0_bank_id = get_arg_val<uint32_t>(2); // Bank ID to read from shared mem
    uint32_t print_core = get_arg_val<uint32_t>(3); // Which core will write data to shared mem
    uint32_t dst0_bank_id = get_arg_val<uint32_t>(4); // Bank ID to write to shared mem
    uint32_t bandwidth_optimal = (bool) get_arg_val<uint32_t>(5); // Which algorithm to use

    uint32_t algo_steps = get_arg_val<uint32_t>(6); // Number of communication steps
    uint32_t num_tiles = get_arg_val<uint32_t>(12); //  Total number of tiles involved in the allreduce
    uint32_t num_tiles_per_node = get_arg_val<uint32_t>(13); // Number of tiles per NOC node (== tiles/block)
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
    
    uint32_t this_core_i = get_arg_val<uint32_t>(9); //Core's linear index
    bool this_core_SE = (bool)get_arg_val<uint32_t>(10); //If the NoC is SE (true) or NW (false)
    uint32_t packed_direction_bools = get_arg_val<uint32_t>(11); //Which core will send in each step

    uint64_t src0_noc_addr = get_noc_addr_from_bank_id<true>(src0_bank_id, src0_addr);

    // setup circular buffers
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

    // Size of the different data structures
    uint32_t tile_size_bytes = get_tile_size(cb_id_local);
    uint32_t block_size_bytes = tile_size_bytes * num_tiles_per_node;
    uint32_t total_vector_size_bytes  = tile_size_bytes * num_tiles;

    // Pointers for circular buffers
    uint32_t l1_write_addr_recv = get_write_ptr(cb_id_recv);
    uint32_t l1_write_addr_local = get_write_ptr(cb_id_local);

    // Communication partner at each step
    uint32_t dst_core_x[algo_steps];
    uint32_t dst_core_y[algo_steps];
    
    //Which blocks need to be sent/received in each step
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

    // Number of each set of semaphores (max total sems = 8)
    const uint32_t num_sem_0 = 6;
    const uint32_t num_sem_1 = 8 - num_sem_0;

    // Read and setup semaphores

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

    // read data from shared DRAM to local SRAM
    if (!this_core_SE) {
        noc_async_read(src0_noc_addr, l1_write_addr_local, tile_size_bytes * num_tiles);
        noc_async_read_barrier();
        uint64_t dst0_noc_addr = get_noc_addr_from_bank_id<true>(dst0_bank_id, dst0_addr );
    }

    uint64_t dst_noc_semaphore_0, dst_noc_semaphore_1, dst_noc_addr;
    bool direction_SE, send_block;
    uint32_t num_syncs = 32;  // Number of synchronizations performed between the different nodes
    uint32_t sync_stride = total_nodes / num_syncs;
    if (num_tiles < 64 && num_tiles > 2) {
        num_syncs = num_tiles / 2;
    } else if (num_tiles == 2) {
        num_syncs = 2;
        sync_stride = 1;
    } else if (num_tiles <= 1) {
        num_syncs = 1;
        sync_stride = 1;
    }

    for (uint32_t j = 0; j < 1; j++) { // # repeats of algorithm to get accurate timings
        DeviceZoneScopedN("ALL_RED_LOOP");
        {
        sync_NOC(cb_id_this, cb_id_that);
        noc_semaphore_set(semaphore_1_ptr[0], 0); // reset semaphores
        // if bandwidth optimal -> reduce scatter else latency optimal -> allreduce
        for (uint32_t i = 0; i < algo_steps; i++) {
            direction_SE = (packed_direction_bools >> i) & 1;  //Get the communication direction for this step
            sync_NOC(cb_id_this, cb_id_that);

            uint32_t n_block_sync = sync_stride;
            if (this_core_SE == direction_SE) { //  This core is sending data

                // Get the addresses of the remote semaphores
                dst_noc_semaphore_0 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_0[i % num_sem_0]);
                dst_noc_semaphore_1 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_1[0]);

                // Reserves the entire circular buffer, can only be reserved once computation core is finished
                cb_reserve_back(cb_id_local, num_tiles);

                // await first sem from comm partner
                noc_semaphore_inc(dst_noc_semaphore_0, 1);
                int semaphore_wait_count = bandwidth_optimal ? 2 * j + 1 : j + 1;
                noc_semaphore_wait_min(semaphore_0_ptr[i % num_sem_0], semaphore_wait_count);

                // Iterate through the blocks of tiles and send the appropriate ones
                for (uint32_t n_block = 0; n_block < total_nodes; ) {
                    send_block = shouldSendBlock(bandwidth_optimal, send_block_indexes[i],
                        n_block, num_tiles);
                    uint32_t blocks_to_send = 0;
                    if (send_block) { // true, send all the tiles in this block
                        uint32_t offset = block_size_bytes * n_block;

                        //address to write to
                        dst_noc_addr = get_noc_addr(dst_core_x[i], dst_core_y[i], l1_write_addr_recv + offset);
                        //  Loop to calculate how many contiguous blocks to send
                        while (send_block && n_block < total_nodes && n_block < n_block_sync) { 
                            blocks_to_send++;
                            n_block++;
                            send_block = shouldSendBlock(bandwidth_optimal, send_block_indexes[i],
                                n_block, num_tiles);
                        }
                        //write to remote node
                        noc_async_write(l1_write_addr_local + offset, dst_noc_addr, block_size_bytes * blocks_to_send);
                    } else {
                        n_block++;
                    }
                    if (n_block >= n_block_sync) {
                        // Periodically (every num_tiles/num_sync blocks) synchronize the nodes and
                        // increment the the circular buffers, allowing computation to proceed
                        noc_async_write_barrier();
                        noc_semaphore_inc(dst_noc_semaphore_1, 1);
                        cb_push_back(cb_id_local, num_tiles / num_syncs);
                        n_block_sync = n_block_sync + sync_stride;
                        if (n_block > num_tiles) {
                            n_block += total_nodes;
                        }
                    }
                }
            } else { // This core is monitoring the semaphores and passing data to compute asap
                cb_reserve_back(cb_id_recv, num_tiles);
                // idle core monitors semaphore and pushes data to compute for greater parallelism
                for (uint32_t n_block = 0; n_block < num_syncs; n_block++) {
                    noc_semaphore_wait_min(semaphore_1_ptr[0], i * num_syncs + n_block + 1);
                    cb_push_back(cb_id_recv, num_tiles / num_syncs);
                }
            }
        }
        if (this_core_SE){ // Reserves full buffer to ensure compute has finished
            cb_reserve_back(cb_id_recv, num_tiles);
        }

        //This second allgather loop is only performed for the bandwidth optimal algorithm
        if (bandwidth_optimal){
            sync_NOC(cb_id_this, cb_id_that); // Synchronize before all gather
            noc_semaphore_set(semaphore_1_ptr[0], 0);
            noc_semaphore_set(semaphore_1_ptr[1], 0);
            //all gather
            for (uint32_t i = algo_steps; i-- > 0; ) {
                direction_SE = (packed_direction_bools >> i) & 1;  // Extract bit i
                sync_NOC(cb_id_this, cb_id_that);
                
                // If this core is sending/recieving, note the other core is fully idle here.
                if (this_core_SE == direction_SE) {
                    dst_noc_semaphore_0 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_0[i % num_sem_0]);
                    dst_noc_semaphore_1 = get_noc_addr(dst_core_x[i], dst_core_y[i], semaphore_1[i % num_sem_1]);

                    // await first sem from comm partner
                    noc_semaphore_inc(dst_noc_semaphore_0, 1);
                    noc_semaphore_wait_min(semaphore_0_ptr[i % num_sem_0], 2 * j + 2);

                    // More or less the same as the scatter loop but in reverse
                    for (uint32_t n_block = 0; n_block < total_nodes; ) {
                        //determine if it's necessary to send this block)
                        send_block = (recv_block_indexes[i] >> n_block) & 1;  
                        if (send_block) {
                            uint32_t offset = block_size_bytes * n_block;
                            dst_noc_addr = get_noc_addr(dst_core_x[i], dst_core_y[i], l1_write_addr_local + offset);
                            uint32_t tiles_to_send = 0;
                            // Checks if next block(s) also need to be sent, to reduce number of remote writes
                            do {
                                tiles_to_send++;
                                n_block++;
                                if (n_block < total_nodes) {
                                    send_block = (recv_block_indexes[i] >> n_block) & 1;
                                } else {
                                    send_block = 0;  // Prevent reading past the mask
                                }
                            } while (send_block && n_block < total_nodes);
                            noc_async_write(l1_write_addr_local + offset, dst_noc_addr, block_size_bytes * tiles_to_send);
                        } else {
                            n_block++;
                        }
                    }
                    // Once write has finished, signal the remote core and await signal.
                    noc_async_write_barrier();
                    noc_semaphore_inc(dst_noc_semaphore_1, 1);
                    uint32_t sem_value = ((algo_steps - i)+1)/2;
                    noc_semaphore_wait_min(semaphore_1_ptr[i%2], sem_value);
                }
            }
        }
        }
    }
    //Sync, then write data back to shared DRAM
    sync_NOC(cb_id_this, cb_id_that);
    if (this_core_SE == direction_SE && this_core_i == print_core) {
        uint32_t offset = block_size_bytes * total_nodes * this_core_i;
        uint64_t dst0_noc_addr = get_noc_addr_from_bank_id<true>(dst0_bank_id, dst0_addr);
        noc_async_write(l1_write_addr_local, dst0_noc_addr, total_vector_size_bytes);
        noc_async_write_barrier();
        DPRINT << "NOC SE finished" << ENDL();
    } else {
        DPRINT << "NOC NW finished" << ENDL();
    }
}
