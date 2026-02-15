from Helper import Helper, print_styled_box
from ReqGenerator import ReqGenerator
from CoreOnmem import CoreOnmem
from MemProfile import MemProfile
from EnergyEstimator import EnergyEstimator
from RuntimeModel import RuntimeModel
from MemoryModel import MemoryModel
from ConfigLoader import ConfigLoader
from matrix_simulation import run_matrix_simulation
import argparse
import sys
import numpy as np
import os
import yaml
import subprocess
import shutil

## Credit: Original code from Rishabh
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

if __name__ == "__main__":
    #-------------------------------------------------------------------
    
    #######################
    ### parse arguments ###
    #######################
    
    parser = argparse.ArgumentParser(description="EONSim")
    # memory config
    parser.add_argument("--memory-config", type=str, default="spad_naive")
    
    # workload config (New)
    parser.add_argument("--workload-config", type=str, required=True, help="Path to workload config (without extension)")

    # execution and dataset related parameters
    parser.add_argument("--data-generation", type=str, default="./datasets/reuse_high/table_1M.txt")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--output-name", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--profiling-multiplier", type=int, default=1)
    parser.add_argument("--output-filename", type=str, default=None, help="Filename for simulation results (without extension)")
    
    # Output base directory (New)
    parser.add_argument("--output-base-dir", type=str, default="results", help="Base directory for output results")
    
    # Matrix config (New)
    parser.add_argument("--matrix-config", type=str, default="tpuv6e.cfg", help="Matrix configuration file name")

    # mNPUsim related parameters
    parser.add_argument("--offchip-memory-config", type=str, default="dram_config/total_dram_config/single_hbm3_819gbs.cfg")
    parser.add_argument("--npumem-config", type=str, default="npumem_config/npumem_architecture_list/single.txt")
    
    # argparses
    args = parser.parse_args()

    # Load workload config
    print(f"[DEBUG] Loading workload config from base path: {args.workload_config}")
    cfg_loader = ConfigLoader(args.workload_config)
    
    # Extract parameters from ConfigLoader
    emb_conf = cfg_loader.get_embedding_config()
    gen_conf = cfg_loader.get_general_config()
    matrix_ops_csv_path = cfg_loader.get_matrix_ops_config_path()
    
    # Set simulation parameters directly from config
    emb_dim = emb_conf['embedding_dim']
    embsize = emb_conf['emb_size_str']
    num_indices_per_lookup = emb_conf['pooling_factor']
    
    # Extract additional params for directory naming
    vectors_per_table = emb_conf['vectors_per_table']
    num_tables = emb_conf['num_tables']
    pooling_factor = emb_conf['pooling_factor']
    
    # Set numeric format from config
    args.numeric_format_bits = gen_conf['num_format']
    
    workload_type = gen_conf['workload_type']
    
    print(f"[DEBUG] Matrix Ops CSV Config Path: {matrix_ops_csv_path}")
    print(f"[DEBUG] Generated Embedding Size String: {embsize[:50]}...")

    mem_config_file = args.memory_config
    n_format_bits = args.numeric_format_bits
    n_format_byte = int(np.ceil(n_format_bits / 8))
    nbatches = args.num_batches
    # embsize and emb_dim are already set above
    bsz = args.batch_size # batch size
    fname = args.data_generation
    # num_indices_per_lookup is already set above
    
    # Generate output directory path based on workload parameters
    # Rule: "vector_dimension"_"rows_per_table"_"num_tables"_"pooling_factor"_"batch_size"
    output_dir_name = f"{emb_dim}_{vectors_per_table}_{num_tables}_{pooling_factor}_{bsz}"
    output_dir = os.path.join(args.output_base_dir, output_dir_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"[DEBUG] Created output directory: {output_dir}")

    # Write output_dir to a temp file for the shell script to move the log file
    with open(".last_output_dir", "w") as f:
        f.write(output_dir)
    
    prof_multiplier = args.profiling_multiplier
    # workload_type is already set above
    
    # Script dir setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[DEBUG] script_dir: {script_dir}")
    
    # mNPUsim related configurations
    offchip_memory_config = args.offchip_memory_config
    npumem_config = args.npumem_config
    # mnpusim_path = "/home/choi/simulators/mNPUsim"
    mnpusim_path = os.path.join(os.path.dirname(script_dir), 'tools', 'mNPUsim')
    print(f"[DEBUG] mnpusim_path: {mnpusim_path}")
    
    # Set up config paths
    config_path = os.path.join(os.path.dirname(script_dir), 'configs', f'{mem_config_file}.yaml')
    mnpusim_config_path = os.path.join(os.path.dirname(script_dir), 'configs', 'mNPUsim_related')
    
    # Matrix config path construction
    matrix_config_path = os.path.join(os.path.dirname(script_dir), 'configs', 'scalesim_config', args.matrix_config)
    
    print(f"[DEBUG] memory_config_path: {config_path}")
    print(f"[DEBUG] mnpusim_config_path: {mnpusim_config_path}")
    print(f"[DEBUG] matrix_config_path: {matrix_config_path}")
    
    # Load memory configuration using ConfigLoader
    mem_config = ConfigLoader.load_memory_config(config_path)
    
    # Extract local buffer configuration
    local_buf = mem_config['local_buffer']
    mem_size = local_buf['mem_size']
    mem_type = local_buf['mem_type']
    mem_policy = local_buf['mem_policy']
    mem_gran = local_buf['mem_gran']
    mem_latency = local_buf['mem_latency']
    cache_config = mem_config['cache_config']
    
    # Extract global buffer configuration
    global_buf = mem_config['global_buffer']
    global_mem_size = global_buf['mem_size']
    global_mem_type = global_buf['mem_type']
    global_mem_policy = global_buf['mem_policy']
    global_mem_latency = global_buf['mem_latency']
    global_cache_config = mem_config['global_cache_config']
    
    # Extract core dimension
    core_dim = mem_config['core_dim']
    core_row = core_dim['row']
    core_col = core_dim['col']
    
    # Extract vector unit configuration
    vector_unit = mem_config['vector_unit']
    vector_lanes = vector_unit['lanes']
    vector_sublanes = vector_unit['sublanes']
    vector_alus_per_sublanes = vector_unit['alus_per_sublanes']
    
    # Extract matrix unit configuration
    matrix_unit = mem_config['matrix_unit']
    mxu_dimension = matrix_unit['mxu_dimension']
    num_mxus = matrix_unit['num_mxus']
    
    # Print the parsed configuration for debugging
    print(f"[DEBUG] Core Dimension - Row: {core_row}, Col: {core_col}")
    print(f"[DEBUG] Vector Unit - Lanes: {vector_lanes}, Sublanes: {vector_sublanes}, ALUs per sublanes: {vector_alus_per_sublanes}")
    print(f"[DEBUG] Matrix Unit - MXU dimension: {mxu_dimension}, Number of MXUs: {num_mxus}")
    print(f"[DEBUG] Local Buffer - Type: {mem_type}, Size: {mem_size} KB, Policy: {mem_policy}, Latency: {mem_latency} cycles")
    if global_mem_size > 0:
        print(f"[DEBUG] Global Buffer - Type: {global_mem_type}, Size: {global_mem_size} KB, Policy: {global_mem_policy}, Latency: {global_mem_latency} cycles")

    # these are for convenience...
    emb_config = np.fromstring(embsize, dtype=int, sep="-")
    emb_config = np.asarray(emb_config, dtype=np.int32)
    num_tables = len(emb_config)
    vectors_per_table = emb_config[0]
    
    helper = Helper()
    
    #-------------------------------------------------------------------
    
    ################################
    ### Create request generator ###
    ################################

    helper.set_timer()
        
    reqgen = ReqGenerator(nbatches, n_format_byte, embsize, emb_dim, bsz, fname, num_indices_per_lookup, mem_gran)
    reqgen.data_gen()
    
    # # temporal test: store reqgen.ls_i np array in a txt file, each element in each row in the txt file.
    # with open("ls_i.txt", "w") as f:
    #     for i in range(len(reqgen.lS_i)):
    #         for j in range(len(reqgen.lS_i[i])):
    #             for k in range(len(reqgen.lS_i[i][j])):
    #                 f.write(str(reqgen.lS_i[i][j][k]) + "\n")
    #             # f.write("\n")
    # f.close()
    
    # exit()
    
    
    
    print_general_config(reqgen.nbatches, reqgen.n_format_byte, reqgen.bsz, reqgen.embsize, reqgen.emb_dim, reqgen.num_indices_per_lookup, reqgen.fname)

    helper.end_timer("model and data gen")
    
    #-------------------------------------------------------------------
    
    ######################################
    ### Convert indices to memory addr ###
    ######################################
    
    helper.set_timer()
    reqgen.index_to_addr()
    
    # reqgen.do_batch_access_pattern_analysis() # DEBUG
    # exit()
    
    emb_dataset = reqgen.addr_trace
    # print("len(emb_dataset): {}".format(len(emb_dataset)))
    # print("len(emb_dataset[0]): {}".format(len(emb_dataset[0])))
    # print("emb_dataset[0][0].shape: {}".format(emb_dataset[0][0].shape))
    helper.end_timer("address generation")
    
    # temporal test: store reqgen.addr_trace np array in a txt file, each element in each row in the txt file.
    # with open("rand_.txt", "w") as f:
    #     for i in range(len(reqgen.addr_trace)):
    #         for j in range(len(reqgen.addr_trace[i])):
    #             for k in range(len(reqgen.addr_trace[i][j])):
    #                 f.write(str(reqgen.addr_trace[i][j][k]) + "\n")
    #                 # f.write(str(reqgen.addr_trace[i][j][k]) + ",")
    #             # f.write("\n")
    # f.close()
    
    # exit()

    #-------------------------------------------------------------------
    
    ###############################
    ### Create memory structure ###
    ###############################
    
    helper.set_timer()    
    
    # Create core on-memory object
    if mem_type == "spad" or mem_type == "cache":
        core_onmem_obj = CoreOnmem(mem_size, mem_type, cache_config, emb_dim, emb_dataset, n_format_byte, vectors_per_table=vectors_per_table, mem_gran=mem_gran, prof_multiplier=prof_multiplier, mem_latency=mem_latency)
    elif mem_type == "profile":
        # generate the profiled dataset path by replacing the folder name with 'profiled_datasets'
        last_slash = fname.rfind('/')
        second_last_slash = fname[:last_slash].rfind('/')
        file_name = fname[last_slash:]
        profiled_path = fname[:second_last_slash+1] + 'profiled_datasets' + file_name
        # print("[DEBUG] profiled_path: {}".format(profiled_path))
        # print("[DEBUG] argument of core_onmem_obj: {}, {}, {}, {}, {}, {}, {}, {}".format(mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran, n_format_byte, profiled_path))
        
        core_onmem_obj = MemProfile(mem_size, mem_type, cache_config, emb_dim, emb_dataset, vectors_per_table, mem_gran, n_format_byte, profiled_path, prof_multiplier)        
        
        # if mem_policy == "profile_dynamic_count":
        core_onmem_obj.set_index_trace(reqgen.lS_i)
        
    core_onmem_obj.set_policy(mem_policy)
    core_onmem_obj.print_config()
    # print("on_mem: {}, data structure size: {:.2f} KB".format(core_onmem_obj.on_mem, sys.getsizeof(core_onmem_obj.on_mem)/1024))
    print("on mem data structure size: {:.2f} KB".format(sys.getsizeof(core_onmem_obj.on_mem)/1024))
    
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
    # memory_model = MemoryModel(script_dir, core_onmem_obj.offmem_trace, mnpusim_path, mnpusim_config_path, offchip_memory_config, npumem_config)
    # memory_model.do_memory_simulation()
    # helper.end_timer("off-chip memory simulation")
    
    #-------------------------------------------------------------------
    
    ##################################
    ### Execution Time Calculation ###
    ##################################
    
    # helper.set_timer()
    # compute_time = RuntimeModel(workload_type, emb_dim, num_tables, bsz, num_indices_per_lookup, vector_lanes, vector_sublanes, vector_alus_per_sublanes, mxu_dimension, num_mxus)
    # compute_time.do_runtime_calculation()
    # helper.end_timer("do execution time calculation")
    
    #-------------------------------------------------------------------
    
    #################################
    ### Run Energy estimation ###
    #################################
    
    # helper.set_timer()
    
    # # set the parameters for energy estimation
    # workload_type = fname.split('/')[-2]
    
    # print("[DEBUG] workload_type: {}".format(workload_type))

    # workload_config_path = os.path.join(os.path.dirname(script_dir), 'configs', 'workload_config.yaml')
    # energy_table_path = os.path.join(os.path.dirname(script_dir), 'configs', 'energy_estimation_table.yaml')
    # # access_per_batch = num_tables * num_indices_per_lookup * bsz
    # access_per_batch = num_tables * len(reqgen.addr_trace[0][0])
    # tech_node = 45
    # if n_format_byte == 4: # currently only support fp32 and int8
    #     energy_n_format = "fp32"
    # elif n_format_byte == 1:
    #     energy_n_format = "int8"
    
    # energy_est = EnergyEstimator(workload_type, workload_config_path, tech_node, energy_table_path, energy_n_format, core_onmem_obj.access_results, access_per_batch, mem_gran)
    # # energy_est.print_all_config()
    # energy_est.do_energy_estimation()
    
    # helper.end_timer("energy estimation")
    
    #-------------------------------------------------------------------
    
    ###########################################
    ## Run Simulation for Matrix Operations ###
    ###########################################
    
    if matrix_ops_csv_path and os.path.exists(matrix_ops_csv_path):
        helper.set_timer()
        print("\n[Matrix Operations Simulation]")
        
        run_matrix_simulation(
            matrix_ops_csv_path, 
            matrix_config_path, 
            mnk_flag="gemm", 
            output_dir=output_dir, 
            output_filename=args.output_filename,
            debug=False
        )
        helper.end_timer("matrix operations simulation")
    else:
        print("[WARNING] Matrix operations CSV config not found or not provided. Skipping matrix simulation.")


