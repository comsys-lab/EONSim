from helper_modules.Helper import Helper, print_styled_box
from ReqGenerator import ReqGenerator
from CoreOnmem_multicore import CoreOnmem
# from MemProfile import MemProfile
# from helper_modules.EnergyEstimator import EnergyEstimator
# from RuntimeModel import RuntimeModel
from MemoryModel import MemoryModel
from helper_modules.ConfigBuilder import build_sim_config
from matrix_ops_sim.matrix_simulation import run_matrix_simulation
import argparse
import numpy as np
import os


def print_general_config(nbatches, n_format_byte, bsz, table_config, emb_dim, lookups_per_sample, fname):
    emb_config = np.fromstring(table_config, dtype=int, sep="-")
    emb_config = np.asarray(emb_config, dtype=np.int32)
    
    content = [
        f"Dataset: {fname}",
        f"Numeric format: {str(n_format_byte*8)} bits",
        f"Num batches: {str(nbatches)}",
        f"Num tables: {str(len(emb_config))}",
        f"Batch Size (samples per batch): {str(bsz)}",
        f"Vectors per table: {str(emb_config[0])}",
        f"Lookups per sample: {str(lookups_per_sample)}",
        f"Embedding Dimension {str(emb_dim)}"
    ]
    
    print_styled_box("General Simulation Configuration", content)


def parse_args():
    parser = argparse.ArgumentParser(description="EONSim")

    # memory config
    parser.add_argument("--memory-config", type=str, default="spad_naive")

    # workload config
    parser.add_argument("--workload-config", type=str, required=True, help="Path to workload config (without extension)")

    # execution and dataset related parameters
    parser.add_argument("--data-generation", type=str, default="./datasets/reuse_high/table_1M.txt")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--output-name", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    # `profiling-period` is the canonical name; keep multiplier alias for backward compatibility.
    parser.add_argument("--profiling-period", "--profiling-multiplier", dest="profiling_period", type=int, default=1)
    parser.add_argument("--output-filename", type=str, default=None, help="Filename for simulation results (without extension)")
    parser.add_argument("--this-output-dir-file", type=str, default=None, help="Per-run file path to write resolved output directory")

    # Output base directory
    parser.add_argument("--output-base-dir", type=str, default="results", help="Base directory for output results")

    # Matrix ops. config
    parser.add_argument("--matrix-config", type=str, default="tpuv6e.cfg", help="Matrix configuration file name")

    # mNPUsim related parameters
    parser.add_argument("--offchip-memory-config", type=str, default="dram_config/total_dram_config/single_hbm3_819gbs.cfg")
    parser.add_argument("--npumem-config", type=str, default="npumem_config/npumem_architecture_list/tpuv6e.txt")

    # debug flag
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug prints")

    return parser.parse_args()


if __name__ == "__main__":
    #-------------------------------------------------------------------
    
    #######################
    ### parse arguments ###
    #######################
    
    args = parse_args()
    sim_cfg = build_sim_config(args)
    
    helper = Helper()
    
    #-------------------------------------------------------------------
    
    ################################
    ### Create request generator ###
    ################################

    helper.set_timer()
        
    reqgen = ReqGenerator(
        sim_cfg.nbatches,
        sim_cfg.n_format_byte,
        sim_cfg.embsize,
        sim_cfg.emb_dim,
        sim_cfg.bsz,
        sim_cfg.fname,
        sim_cfg.num_indices_per_lookup,
        sim_cfg.mem_gran,
        debug=sim_cfg.debug,
    )
    reqgen.data_gen()
    
    print_general_config(reqgen.nbatches, reqgen.n_format_byte, reqgen.bsz, reqgen.embsize, reqgen.emb_dim, reqgen.num_indices_per_lookup, reqgen.fname)

    helper.end_timer("model and data gen")
    
    #-------------------------------------------------------------------
    
    ######################################
    ### Convert indices to memory addr ###
    ######################################
    
    helper.set_timer()
    reqgen.index_to_addr()
    
    emb_dataset = reqgen.addr_trace
    helper.end_timer("address generation")

    #-------------------------------------------------------------------
    
    ###############################
    ### Create memory structure ###
    ###############################
    
    helper.set_timer()    
    
    # Create core on-memory object with the parsed configuration parameters
    core_onmem_obj = CoreOnmem(
        sim_cfg.mem_size,
        sim_cfg.mem_type,
        sim_cfg.cache_config,
        sim_cfg.emb_dim,
        emb_dataset,
        sim_cfg.n_format_byte,
        vectors_per_table=sim_cfg.vectors_per_table,
        mem_gran=sim_cfg.mem_gran,
        prof_period=sim_cfg.prof_period,
        mem_latency=sim_cfg.mem_latency,
        num_cores=sim_cfg.num_cores,
        onchip_structure=sim_cfg.onchip_structure,
        local_onmem_config={
            "mem_size": sim_cfg.local_onmem_size,
            "mem_type": sim_cfg.local_onmem_type,
            "mem_policy": sim_cfg.local_onmem_policy,
            "mem_latency": sim_cfg.local_onmem_latency,
        },
        global_onmem_config={
            "mem_size": sim_cfg.global_onmem_size,
            "mem_type": sim_cfg.global_onmem_type,
            "mem_policy": sim_cfg.global_onmem_policy,
            "mem_latency": sim_cfg.global_onmem_latency,
        },
        index_trace=reqgen.lS_i,
        debug=sim_cfg.debug,
    )
    core_onmem_obj.set_policy(sim_cfg.mem_policy)
    core_onmem_obj.print_config()
    # print("on mem data structure size: {:.2f} KB".format(sys.getsizeof(core_onmem_obj.on_mem)/1024))
    
    helper.end_timer("create memory structure")

    #-------------------------------------------------------------------
    
    ##########################
    ### Run Simulation ###
    ##########################
    
    helper.set_timer()
    core_onmem_obj.do_simulation()
    helper.end_timer("do simulation")
    
    #-------------------------------------------------------------------
    
    ####################################################
    ### Off-chip Memory Simulation using mNPUsim ###
    ####################################################
    
    # helper.set_timer()
    # memory_model = MemoryModel(
    #     sim_cfg.script_dir,
    #     core_onmem_obj.offmem_trace,
    #     sim_cfg.mnpusim_path,
    #     sim_cfg.mnpusim_config_path,
    #     sim_cfg.offchip_memory_config,
    #     sim_cfg.npumem_config,
    #     debug=sim_cfg.debug,
    # )
    # memory_model.do_memory_simulation()
    # helper.end_timer("off-chip memory simulation")
    
    #-------------------------------------------------------------------
    
    ####################################
    ### Computation Time Calculation ### (TODO: integrate memory simulation results to calculate the total runtime more accurately)
    ####################################
    
    # helper.set_timer()
    # compute_time = RuntimeModel(
    #     workload_type,
    #     emb_dim,
    #     num_tables,
    #     bsz,
    #     num_indices_per_lookup,
    #     vector_lanes,
    #     vector_sublanes,
    #     vector_alus_per_sublanes,
    #     mxu_dimension,
    #     num_mxus,
    #     onchip_config={
    #         "onchip_structure": sim_cfg.onchip_structure,
    #         "local_onmem_size": sim_cfg.local_onmem_size,
    #         "local_onmem_latency": sim_cfg.local_onmem_latency,
    #         "global_onmem_size": sim_cfg.global_onmem_size,
    #         "global_onmem_latency": sim_cfg.global_onmem_latency,
    #         "onmem_type": sim_cfg.mem_type,
    #         "onmem_policy": sim_cfg.mem_policy,
    #     },
    # )
    # compute_time.do_runtime_calculation()
    # helper.end_timer("do execution time calculation")
    
    #-------------------------------------------------------------------
    
    ###########################################
    ## Run Simulation for Matrix Operations ###
    ###########################################
    
    if sim_cfg.matrix_ops_csv_path and os.path.exists(sim_cfg.matrix_ops_csv_path):
        helper.set_timer()
        print("\n[Matrix Operations Simulation]")
        
        run_matrix_simulation(
            sim_cfg.matrix_ops_csv_path,
            sim_cfg.matrix_config_path,
            mnk_flag="gemm", 
            output_dir=sim_cfg.output_dir,
            output_filename=sim_cfg.output_filename,
            debug=sim_cfg.debug
        )
        helper.end_timer("matrix operations simulation")
    else:
        print("[WARNING] Matrix operations CSV config not found or not provided. Skipping matrix simulation.")


