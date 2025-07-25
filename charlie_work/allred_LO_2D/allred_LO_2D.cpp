#include <tt-metalium/device.hpp>
#include "allred_helper.hpp"

int main(int argc, char** argv) {
    IDevice* device = CreateDevice(0);

    CommandQueue& cq = device->command_queue();
    Program program = CreateProgram();

    int SIDE_LENGTH = (argc >= 4) ? highest_power_of_two(std::stoi(argv[3])) : 1;
    CoreRange cores({0, 0}, {SIDE_LENGTH - 1, SIDE_LENGTH - 1});

    // Initialize the allreduce  setup
    AllredConfig arCfg(argc, argv, device, cq, program, cores, SIDE_LENGTH, false);

    /*NOC kernel arg initialization*/
    std::vector<uint32_t> dataflow_args(12 + 8 + 2 * arCfg.SWING_ALGO_STEPS);
    dataflow_args[1] = arCfg.dst_dram_buffer->address();
    dataflow_args[4] = arCfg.dst_bank_id;
    dataflow_args[6] = arCfg.SWING_ALGO_STEPS;
    dataflow_args[11] = arCfg.NUM_TILES;
    for (int i = 0; i < 8; i++) {
        dataflow_args[12 + 2 * arCfg.SWING_ALGO_STEPS + i] =
            (uint32_t)tt_metal::CreateSemaphore(program, cores, INVALID);
    }

    /*Compute kernel arg initialization*/
    std::vector<uint32_t> compute_args(5);
    compute_args[0] = arCfg.SWING_ALGO_STEPS;
    compute_args[4] = arCfg.NUM_TILES;

    /*reused variable initialization*/
    KernelHandle dataflow_0_kernel, dataflow_1_kernel, compute_kernel;
    CoreCoord logical_core, physical_core;
    bool horizontal_step,sending_SE;
    uint32_t step_directions = 0b00000;
    int node_position, node_other_position, message_pass_depth, recv_node;


    /*create kernels for each core*/
    for (int core_i = 0; core_i < arCfg.core_array.size(); core_i++) {
        physical_core = device->worker_core_from_logical_core(arCfg.core_array[core_i]);
        dataflow_args[7] = (uint32_t)physical_core.x;
        dataflow_args[8] = (uint32_t)physical_core.y;
        compute_args[1] = (uint32_t)physical_core.x;
        compute_args[2] = (uint32_t)physical_core.y;
        if (arCfg.core_array[core_i].x % 2 == 0) {
            dataflow_args[0] = arCfg.src_1_dram_buffer->address();
            dataflow_args[2] = arCfg.src_1_bank_id;
        } else {
            dataflow_args[0] = arCfg.src_0_dram_buffer->address();
            dataflow_args[2] = arCfg.src_0_bank_id;
        }

        horizontal_step = true;  // Start calcs on hrz step
        if (!arCfg.SWING_VERSION) {
            /*Recursive doubling algo partner node calculations*/
            message_pass_depth = 1;
            for (int recdub_step = 0; recdub_step < arCfg.SWING_ALGO_STEPS; recdub_step++) {
                int comm_partner_id = get_comm_partner_recdub_2D(
                    core_i, recdub_step, horizontal_step, message_pass_depth, step_directions, SIDE_LENGTH);

                logical_core = {comm_partner_id % SIDE_LENGTH, comm_partner_id / SIDE_LENGTH};

                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[12 + 2 * recdub_step] = (uint32_t)physical_core.x;
                dataflow_args[13 + 2 * recdub_step] = (uint32_t)physical_core.y;

                message_pass_depth = horizontal_step ? message_pass_depth : 2 * message_pass_depth;
                horizontal_step = !horizontal_step;
            }
        } else {
            /*Swing communication partner calculations*/
            int comm_partner_idx;
            for (int swing_step = 0; swing_step < arCfg.SWING_ALGO_STEPS; swing_step++) {
                comm_partner_idx =
                    get_comm_partner_swing_2D(core_i, swing_step, horizontal_step, SIDE_LENGTH, arCfg.TOTAL_NODES);
                logical_core = arCfg.core_array[comm_partner_idx];

                physical_core = device->worker_core_from_logical_core(logical_core);
                dataflow_args[12 + 2 * swing_step] = (uint32_t)physical_core.x;
                dataflow_args[13 + 2 * swing_step] = (uint32_t)physical_core.y;
                horizontal_step = !horizontal_step;
            }
            step_directions = get_SE(arCfg.core_array[core_i].x, arCfg.core_array[core_i].y);
        }

        dataflow_args[10] = step_directions;
        compute_args[3] = step_directions;

        /*SE Kernel*/
        dataflow_args[9] = (uint32_t)true;
        dataflow_0_kernel = CreateDataflowKernel(program, arCfg.core_array[core_i], dataflow_args, true,"allred_LO_2D");  // SE kernel
        /*NW Kernel*/
        dataflow_args[9] = (uint32_t)false;
        dataflow_1_kernel = CreateDataflowKernel(program, arCfg.core_array[core_i], dataflow_args, false,"allred_LO_2D"); // NW kernel
        compute_kernel = CreateComputeKernel(program, arCfg.core_array[core_i], compute_args,"allred_LO_2D");
    }

    arCfg.RunProgram(cq, program, device);

    /* Read in result into a host vector */
    EnqueueReadBuffer(cq, arCfg.dst_dram_buffer, arCfg.result_vec, true);
    validate_result_vector(arCfg.result_vec, arCfg.src_vec_0, arCfg.src_vec_1, arCfg.num_els, arCfg.ERROR, arCfg.TOTAL_NODES);
    CloseDevice(device);
}
