// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "debug/dprint.h"  // required in all kernels using DPRINT
#ifdef TRISC_PACK
#include "llk_io_pack.h"
#endif
#ifdef TRISC_UNPACK
#include "llk_io_unpack.h"
#endif

namespace NAMESPACE {
void MAIN {
    uint32_t algo_steps = get_arg_val<uint32_t>(0);
    uint32_t this_core_x = get_arg_val<uint32_t>(1);
    uint32_t this_core_y = get_arg_val<uint32_t>(2);
    uint32_t packed_bools = get_arg_val<uint32_t>(3);
    uint32_t num_tiles = get_arg_val<uint32_t>(4);
    uint32_t num_tiles_per_node = get_arg_val<uint32_t>(5);
    uint32_t total_nodes = num_tiles / num_tiles_per_node;
    constexpr uint32_t cb_id_compute = tt::CBIndex::c_0;
    constexpr uint32_t cb_id_NW = tt::CBIndex::c_1;
    constexpr uint32_t cb_id_SE = tt::CBIndex::c_2;
    constexpr uint32_t cb_id_recv = tt::CBIndex::c_3;
    constexpr uint32_t cb_id_local = tt::CBIndex::c_16;

    uint64_t block_indexes[algo_steps]; // indexes of blocks to be exchanged

    for (uint32_t i = 0; i < algo_steps; i++) {
        uint64_t low_bits = get_arg_val<uint32_t>(6 + 2 * i);
        uint64_t high_bits = get_arg_val<uint32_t>(7 + 2 * i);
        block_indexes[i] = (high_bits << 32) | low_bits;
    }
    
    DPRINT_UNPACK(DPRINT << "Waiting on local" << ENDL());

    cb_wait_front(cb_id_local, num_tiles); // Wait for dataflow to read data from dram

    // Initialize the compute cores
    binary_op_init_common(cb_id_local, cb_id_recv, cb_id_local);
    add_tiles_init(cb_id_local, cb_id_recv, true);
    cb_pop_front(cb_id_local, num_tiles);
    // DPRINT_UNPACK(DPRINT << "Initialized buffers" << ENDL());

    algo_steps = 2;

    bool SE, recv_block;
    uint32_t recv_offset = 0;

    for (uint32_t j = 0; j < 1; j++) { // # repeats of algorithm to get accurate timings
        for (uint32_t i = 0; i < algo_steps; i++) {
            // DPRINT_MATH(DPRINT << "Compute starting " << i << ENDL());
            // Signal appropriate NOC core to exchange data with other core
            SE = (packed_bools >> i) & 1;  // Extract bit i

            cb_push_back(cb_id_SE, 1);
            cb_push_back(cb_id_NW, 1);

            uint32_t reg_index = 0;
            uint32_t tiles_consumed = 0;
            // cb_wait_front(cb_id_local, num_tiles); // Await local data to be ready
            for (uint32_t n_block = 0; n_block < total_nodes; n_block++) {
                recv_block = (block_indexes[i] >> n_block) & 1;  // Extract bit i
                for (uint32_t tile_num = n_block * num_tiles_per_node; tile_num < (n_block + 1) * num_tiles_per_node;
                      tile_num++) {
                    cb_wait_front(cb_id_local, tile_num); // Await local data to be ready
                    // DPRINT_UNPACK(DPRINT << "Waiting on " << tile_num << ENDL());
                    if (recv_block){
                        // DPRINT_MATH(DPRINT << "Waiting recv " << tile_num << ENDL());
                        tiles_consumed++;
                        cb_wait_front(cb_id_recv, tiles_consumed); // Await blocks to be exchanged
                        
                        tile_regs_acquire();
                        add_tiles(cb_id_local, cb_id_recv, tile_num, recv_offset, reg_index);
                        tile_regs_commit();

                        tile_regs_wait();
                        pack_tile(reg_index, cb_id_local);
                        tile_regs_release();
                        reg_index = reg_index < 7 ? reg_index + 1 : 0; // Increment reg index
                        recv_offset = recv_offset< num_tiles ? recv_offset + 1 : 0;
                    }
                    // if (recv_block){
                    //     cb_wait_front(cb_id_recv, 1); // Await blocks to be exchanged
                    //     tiles_consumed++;
                        
                    //     tile_regs_acquire();
                    //     copy_tile_to_dst_init_short(cb_id_local);
                    //     copy_tile(cb_id_local, tile_num, reg_index);
                    //     binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(cb_id_recv);
                    //     binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD, EltwiseBinaryReuseDestType::DEST_TO_SRCB>(
                    //         cb_id_recv, 0, reg_index);
                    //     tile_regs_commit();

                    //     tile_regs_wait();
                    //     pack_tile(reg_index, cb_id_local);
                    //     cb_pop_front(cb_id_recv, 1);
                    //     tile_regs_release();
                    //     reg_index = reg_index < 7 ? reg_index + 1 : 0; // Increment reg index
                    // }
                    // cb_pop_front(cb_id_local, 1); // Pop local tile
                    
                }
            }
            cb_pop_front(cb_id_local, num_tiles); // Pop local tile
            cb_pop_front(cb_id_recv, tiles_consumed);
            // DPRINT_UNPACK(DPRINT << "Unpack on " << i << ENDL());
            // DPRINT_MATH(DPRINT << "Compute pushed " << tiles_consumed << num_tiles << ENDL());
        }
        cb_push_back(cb_id_SE, 1);
        cb_push_back(cb_id_NW, 1);
    }
    DPRINT_UNPACK(DPRINT << "Unpack done " << ENDL());
    // DPRINT_MATH(DPRINT << "Compute done " << ENDL());
    // DPRINT_PACK(DPRINT << "Pack done " << ENDL());
}
}  // namespace NAMESPACE
