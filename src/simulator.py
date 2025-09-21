from Helper import Helper, print_styled_box
from ReqGenerator import ReqGenerator
from ReqGenerator_temp_criteo import ReqGenerator_temp_criteo # This is for temporal test
from MemSpad import MemSpad
from MemCache import MemCache
from MemProfile import MemProfile
from EnergyEstimator import EnergyEstimator
from RuntimeModel import RuntimeModel
import argparse
import sys
import numpy as np
import os
import yaml

## Credit: Original code from Rishabh; Assisting the args parser
def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value

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
    
    # emb related parameters
    parser.add_argument("--arch-sparse-feature-size", type=int, default=128)
    parser.add_argument("--arch-embedding-size", type=dash_separated_ints, default="500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000-500000")

    # execution and dataset related parameters
    parser.add_argument("--data-generation", type=str, default="./datasets/reuse_high/table_1M.txt")
    parser.add_argument("--numeric-format-bits", type=int, default=8)
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--output-name", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lookups-per-sample", type=int, default=150)
    parser.add_argument("--profiling-multiplier", type=int, default=1)
    parser.add_argument("--workload-type", type=str, default="dlrm")
    
    # argparses
    args = parser.parse_args()
    mem_config_file = args.memory_config
    n_format_bits = args.numeric_format_bits
    n_format_byte = int(np.ceil(n_format_bits / 8))
    nbatches = args.num_batches
    embsize = args.arch_embedding_size
    emb_dim = args.arch_sparse_feature_size #embedding dim
    bsz = args.batch_size # batch size
    fname = args.data_generation
    num_indices_per_lookup = args.lookups_per_sample # pooling factor or lookups per sample
    
    prof_multiplier = args.profiling_multiplier
    workload_type = args.workload_type
    
    # Parse the memory config file - YAML format
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[DEBUG] script_dir: {script_dir}")
    
    # Try YAML format first
    yaml_config_path = os.path.join(os.path.dirname(script_dir), 
                                  'configs', 
                                  f'{mem_config_file}.yaml')
    print(f"[DEBUG] yaml_config_path: {yaml_config_path}")
    
    # Fallback to .config format
    config_path = os.path.join(os.path.dirname(script_dir), 
                                  'configs', 
                                  f'{mem_config_file}.config')
    
    mem_type = None
    mem_policy = None
    cache_way = 0
    cache_line_size = 0    
    rrpv_bits = 0
    rrip_insert = 0
    
    # Try to load YAML config first
    if os.path.exists(yaml_config_path):
        with open(yaml_config_path, 'r') as yaml_cfg:
            config_data = yaml.safe_load(yaml_cfg)
            
            # Parse memory configuration
            memory_config = config_data.get('memory', {})
            mem_size = memory_config.get('mem_size', 0)  # KB
            mem_type = memory_config.get('mem_type', '')
            mem_policy = mem_type + '_' + memory_config.get('policy', '')
            mem_gran = memory_config.get('access_granularity', 0)  # B
            
            # Parse vector unit configuration
            vector_unit_config = config_data.get('vector_unit', {})
            vector_lanes = vector_unit_config.get('lanes', 128)
            vector_sublanes = vector_unit_config.get('sublanes', 8)
            vector_alus_per_sublanes = vector_unit_config.get('ALUs_per_sublanes', 4)
            
            # Parse matrix unit configuration
            matrix_unit_config = config_data.get('matrix_unit', {})
            mxu_dimension = matrix_unit_config.get('mxu_dimension', 128)
            num_mxus = matrix_unit_config.get('num_mxus', 4)
            
            if mem_type == "cache":
                cache_way = memory_config.get('cache_way', 0)
                cache_line_size = mem_gran  # Use access_granularity as cache_line_size
                
            if mem_policy == 'cache_SRRIP' or mem_policy == 'profile_dynamic_SRRIP':
                rrpv_bits = memory_config.get('RRPV_bits', 0)
                rrip_insert = memory_config.get('RRPV_insertion', 0)
                
    # Fallback to old .config format (commented out but kept for reference)
    # elif os.path.exists(config_path):
    #     with open(config_path, 'r') as mem_cfg:
    #         for cfg_line in mem_cfg:
    #             key, value = cfg_line.split(':')
    #             if key.strip() == 'mem_size':
    #                 mem_size = int(value.strip()) # KB
    #             elif key.strip() == 'mem_type':
    #                 mem_type = str(value.strip())
    #             elif key.strip() == 'policy':
    #                 mem_policy = mem_type+'_'+str(value.strip())
    #             elif key.strip() == 'access_granularity':
    #                 mem_gran = int(value.strip()) # B
    #             if mem_type == "cache":
    #                 if key.strip() == 'cache_way':
    #                     cache_way = int(value.strip())
    #                 elif key.strip() == 'cache_line_size':
    #                     # cache_line_size = int(value.strip())
    #                     cache_line_size = mem_gran
    #                 
    #             if mem_policy == 'cache_SRRIP' or mem_policy == 'profile_dynamic_SRRIP':
    #                 if key.strip() == 'RRPV_bits':
    #                     rrpv_bits = int(value.strip())
    #                 elif key.strip() == 'RRPV_insertion':
    #                     rrip_insert = int(value.strip())
    else:
        raise FileNotFoundError(f"Config file not found: {yaml_config_path} or {config_path}")
        
    cache_config = [cache_way, cache_line_size, rrpv_bits, rrip_insert]
    
    # Print the parsed configuration for debugging
    print(f"[DEBUG] Vector Unit - Lanes: {vector_lanes}, Sublanes: {vector_sublanes}, ALUs per sublanes: {vector_alus_per_sublanes}")
    print(f"[DEBUG] Matrix Unit - MXU dimension: {mxu_dimension}, Number of MXUs: {num_mxus}")
    print(f"[DEBUG] Memory - Type: {mem_type}, Size: {mem_size} KB, Policy: {mem_policy}")

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
    
    if "criteo" in fname.lower():
        reqgen = ReqGenerator_temp_criteo(nbatches, n_format_byte, embsize, emb_dim, bsz, fname, num_indices_per_lookup, mem_gran)
    else:
        reqgen = ReqGenerator(nbatches, n_format_byte, embsize, emb_dim, bsz, fname, num_indices_per_lookup, mem_gran)
    # reqgen = ReqGenerator_temp_criteo(nbatches, n_format_byte, embsize, emb_dim, bsz, fname, num_indices_per_lookup, mem_gran)
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
    # with open("addr_trace.txt", "w") as f:
    #     for i in range(len(reqgen.addr_trace)):
    #         for j in range(len(reqgen.addr_trace[i])):
    #             for k in range(len(reqgen.addr_trace[i][j])):
    #                 # f.write(str(reqgen.addr_trace[i][j][k]) + "\n")
    #                 f.write(str(reqgen.addr_trace[i][j][k]) + ",")
    #             # f.write("\n")
    # f.close()
    
    # exit()

    #-------------------------------------------------------------------
    
    ###############################
    ### Create memory structure ###
    ###############################
    
    helper.set_timer()    
    
    # Create mem_struct
    if mem_type == "spad":
        mem_struct = MemSpad(mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran, n_format_byte, prof_multiplier)
    elif mem_type == "cache":
        mem_struct = MemCache(mem_size, mem_type, cache_config, emb_dim, emb_dataset, n_format_byte)
    elif mem_type == "profile":
        # generate the profiled dataset path by replacing the folder name with 'profiled_datasets'
        last_slash = fname.rfind('/')
        second_last_slash = fname[:last_slash].rfind('/')
        file_name = fname[last_slash:]
        profiled_path = fname[:second_last_slash+1] + 'profiled_datasets' + file_name
        print("[DEBUG] profiled_path: {}".format(profiled_path))
        # print("[DEBUG] argument of mem_struct: {}, {}, {}, {}, {}, {}, {}, {}".format(mem_size, mem_type, emb_dim, emb_dataset, vectors_per_table, mem_gran, n_format_byte, profiled_path))
        
        mem_struct = MemProfile(mem_size, mem_type, cache_config, emb_dim, emb_dataset, vectors_per_table, mem_gran, n_format_byte, profiled_path, prof_multiplier)        
        
        # if mem_policy == "profile_dynamic_count":
        mem_struct.set_index_trace(reqgen.lS_i)
        
    mem_struct.set_policy(mem_policy)
    mem_struct.print_config()
    mem_struct.create_on_mem() # num_tables, num_rows_per_table
    # print("on_mem: {}, data structure size: {:.2f} KB".format(mem_struct.on_mem, sys.getsizeof(mem_struct.on_mem)/1024))
    print("on mem data structure size: {:.2f} KB".format(sys.getsizeof(mem_struct.on_mem)/1024))
    
    helper.end_timer("create memory structure")

    #-------------------------------------------------------------------
    
    ##########################
    ### Run Simulation ###
    ##########################
    
    helper.set_timer()
    mem_struct.do_simulation()
    helper.end_timer("do simulation")
    
    #-------------------------------------------------------------------
    
    ##################################
    ### Execution Time Calculation ###
    ##################################
    
    helper.set_timer()
    compute_time = RuntimeModel(workload_type, emb_dim, num_tables, bsz, num_indices_per_lookup, vector_lanes, vector_sublanes, vector_alus_per_sublanes, mxu_dimension, num_mxus)
    compute_time.do_runtime_calculation()
    helper.end_timer("do execution time calculation")
    
    #-------------------------------------------------------------------
    
    #################################
    ### Run Energy estimation ###
    #################################
    
    helper.set_timer()
    
    # set the parameters for energy estimation
    workload_type = fname.split('/')[-2]
    
    print("[DEBUG] workload_type: {}".format(workload_type))

    workload_config_path = os.path.join(os.path.dirname(script_dir), 'configs', 'workload_config.yaml')
    energy_table_path = os.path.join(os.path.dirname(script_dir), 'configs', 'energy_estimation_table.yaml')
    # access_per_batch = num_tables * num_indices_per_lookup * bsz
    access_per_batch = num_tables * len(reqgen.addr_trace[0][0])
    tech_node = 45
    if n_format_byte == 4: # currently only support fp32 and int8
        energy_n_format = "fp32"
    elif n_format_byte == 1:
        energy_n_format = "int8"
    
    energy_est = EnergyEstimator(workload_type, workload_config_path, tech_node, energy_table_path, energy_n_format, mem_struct.access_results, access_per_batch, mem_gran)
    # energy_est.print_all_config()
    energy_est.do_energy_estimation()
    
    helper.end_timer("energy estimation")
    
    #-------------------------------------------------------------------