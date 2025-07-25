#include <tt-metalium/device.hpp>
#include "allred_helper.hpp"

int main(int argc, char** argv) {

    IDevice* device = CreateDevice(0);

    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    int SIDE_LENGTH = (argc >= 4) ? highest_power_of_two(std::stoi(argv[3])) : 1;
    CoreRange cores({0, 0}, {SIDE_LENGTH - 1, SIDE_LENGTH - 1});

    // Initialize the allreduce  setup
    AllredConfig arCfg(argc, argv, device, cq, program, cores, SIDE_LENGTH, true);


    tt_metal::InterleavedBufferConfig common_dram_config{
        .device = device,
        .size = arCfg.single_tile_size * arCfg.NUM_TILES * arCfg.TOTAL_NODES,
        .page_size = arCfg.single_tile_size * arCfg.NUM_TILES * arCfg.TOTAL_NODES,
        .buffer_type = tt_metal::BufferType::DRAM};
    std::shared_ptr<tt::tt_metal::Buffer> common_dram_buffer = CreateBuffer(common_dram_config);
    uint32_t common_bank_id = 0;     // common_dram_noc_coord.x;

    /*NOC kernel arg initialization*/
    std::vector<uint32_t> dataflow_args(17 + 2 * arCfg.SWING_ALGO_STEPS + 8 + 2 * arCfg.SWING_ALGO_STEPS);
    /*args:
    0-5 : src + dst dram
    6-8: common dram
    9: num steps
    10-11: core x, y
    12: core i (x+ y*side length)
    13: is_SE
    14: step_directions
    15: num_tiles
    16: tiles_per_node
    17-28: core x, y for each step
    29-36: semaphores for each step
    37-48: block indexes to send at each step
    */
    dataflow_args[1] = arCfg.dst_dram_buffer->address();
    dataflow_args[4] = arCfg.dst_bank_id;
    dataflow_args[6] = common_dram_buffer->address();
    dataflow_args[7] = common_bank_id;
    dataflow_args[9] = arCfg.SWING_ALGO_STEPS;
    dataflow_args[15] = arCfg.NUM_TILES;
    dataflow_args[16] = arCfg.NUM_TILES / arCfg.TOTAL_NODES;  // tiles per node
    for (int i = 0; i < 8; i++) {
        dataflow_args[17 + 2 * arCfg.SWING_ALGO_STEPS + i] = (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);
    }

    /*Compute kernel arg initialization*/
    std::vector<uint32_t> compute_args(7 + 2 * arCfg.SWING_ALGO_STEPS);
    compute_args[0] = arCfg.SWING_ALGO_STEPS;
    compute_args[5] = arCfg.NUM_TILES;
    compute_args[6] = arCfg.NUM_TILES / arCfg.TOTAL_NODES;  // tiles per node

    /*reused variable initialization*/
    KernelHandle dataflow_0_kernel, dataflow_1_kernel, compute_kernel;
    CoreCoord logical_core, physical_core;
    bool horizontal_step,sending_SE;
    uint32_t step_directions = 0b00000;
    uint32_t dummy_step_directions = 0b00000;
    int node_position, node_other_position, message_pass_depth, recv_node, comm_partner_idx;

    /*create kernels for each core*/
    for (int core_i = 0; core_i < arCfg.core_array.size(); core_i++) {
        physical_core = device->worker_core_from_logical_core(arCfg.core_array[core_i]);
        dataflow_args[10] = (uint32_t)physical_core.x;
        dataflow_args[11] = (uint32_t)physical_core.y;
        dataflow_args[12] = (uint32_t)core_i;  // Added core_i
        compute_args[1] = (uint32_t)physical_core.x;
        compute_args[2] = (uint32_t)physical_core.y;
        compute_args[3] = (uint32_t)core_i;
        if (arCfg.core_array[core_i].x % 2 == 0) {
            dataflow_args[0] = arCfg.src_1_dram_buffer->address();
            dataflow_args[2] = arCfg.src_1_bank_id;
        } else {
            dataflow_args[0] = arCfg.src_0_dram_buffer->address();
            dataflow_args[2] = arCfg.src_0_bank_id;
        }

        /* set block indexes to 0 */
        for (int i = 0; i < 2 * arCfg.SWING_ALGO_STEPS; i++) {
            dataflow_args[25 + 2 * arCfg.SWING_ALGO_STEPS + i] = 0;
        }
        for (int i = 0; i < 2 * arCfg.SWING_ALGO_STEPS; i++) {
            compute_args[7 + i] = 0;
        }

        horizontal_step = true;  // Start calcs on hrz step
        if (!arCfg.SWING_VERSION) {
            /*Recursive doubling algo partner node calculations*/
            message_pass_depth = 1;
            for (int algo_step = 0; algo_step < arCfg.SWING_ALGO_STEPS; algo_step++) {
                comm_partner_idx = get_comm_partner_recdub_2D(
                    core_i, algo_step, horizontal_step, message_pass_depth, step_directions, SIDE_LENGTH);

                logical_core = arCfg.core_array[comm_partner_idx];
                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[17 + 2 * algo_step] = (uint32_t)physical_core.x;
                dataflow_args[18 + 2 * algo_step] = (uint32_t)physical_core.y;

                uint32_t* blocks_to_send = &dataflow_args[25 + 2 * arCfg.SWING_ALGO_STEPS + 2 * algo_step];
                if (comm_partner_idx < 32) {
                    *blocks_to_send = *blocks_to_send | (1 << comm_partner_idx);
                } else {
                    *(blocks_to_send + 1) = *(blocks_to_send + 1) | (1 << (comm_partner_idx - 32));
                }

                uint32_t* blocks_to_recv = &compute_args[7 + 2 * algo_step];
                if (core_i < 32) {
                    *blocks_to_recv = *blocks_to_recv | (1 << core_i);
                } else {
                    *(blocks_to_recv + 1) = *(blocks_to_recv + 1) | (1 << (core_i - 32));
                }

                message_pass_depth = horizontal_step ? message_pass_depth : 2 * message_pass_depth;
                horizontal_step = !horizontal_step;
            }
        } else {
            /*Swing communication partner calculations*/
            for (int algo_step = 0; algo_step < arCfg.SWING_ALGO_STEPS; algo_step++) {
                comm_partner_idx =
                    get_comm_partner_swing_2D(core_i, algo_step, horizontal_step, SIDE_LENGTH, arCfg.TOTAL_NODES);

                logical_core = arCfg.core_array[comm_partner_idx];
                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[17 + 2 * algo_step] = (uint32_t)physical_core.x;
                dataflow_args[18 + 2 * algo_step] = (uint32_t)physical_core.y;

                uint32_t* blocks_to_send = &dataflow_args[25 + 2 * arCfg.SWING_ALGO_STEPS + 2 * algo_step];
                if (comm_partner_idx < 32) {
                    *blocks_to_send = *blocks_to_send | (1 << comm_partner_idx);
                } else {
                    *(blocks_to_send + 1) = *(blocks_to_send + 1) | (1 << (comm_partner_idx - 32));
                }

                uint32_t* blocks_to_recv = &compute_args[7 + 2 * algo_step];
                if (core_i < 32) {
                    *blocks_to_recv = *blocks_to_recv | (1 << core_i);
                } else {
                    *(blocks_to_recv + 1) = *(blocks_to_recv + 1) | (1 << (core_i - 32));
                }

                horizontal_step = !horizontal_step;
            }
            step_directions = get_SE(arCfg.core_array[core_i].x, arCfg.core_array[core_i].y);
        }

        dataflow_args[14] = step_directions;
        compute_args[4] = step_directions;

        /*SE Kernel*/
        dataflow_args[13] = (uint32_t)true;
        CreateDataflowKernel(program, arCfg.core_array[core_i], dataflow_args, true,"allred_mem_2D");  // SE kernel
        /*NW Kernel*/
        dataflow_args[13] = (uint32_t)false;
        CreateDataflowKernel(program, arCfg.core_array[core_i], dataflow_args, false,"allred_mem_2D"); // NW kernel
        CreateComputeKernel(program, arCfg.core_array[core_i], compute_args,"allred_mem_2D");
    }

    arCfg.RunProgram(cq, program, device);

    /* Read in result into a host vector */
    EnqueueReadBuffer(cq, arCfg.dst_dram_buffer, arCfg.result_vec, true);

    validate_result_vector(arCfg.result_vec, arCfg.src_vec_0, arCfg.src_vec_1, arCfg.num_els, arCfg.ERROR, arCfg.TOTAL_NODES);
    CloseDevice(device);
}
