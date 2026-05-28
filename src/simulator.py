from helper_modules.helper import Helper, print_styled_box
from req_generator import ReqGenerator
from core_onmem_model import CoreOnmem
from runtime_model import RuntimeModel
from memory_model import MemoryModel
from helper_modules.config_builder import build_sim_config
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
    parser.add_argument("--memory-config", type=str, default="spm_naive")

    # workload config
    parser.add_argument("--workload-config", type=str, required=True, help="Path to workload config (without extension)")

    # execution and dataset related parameters
    parser.add_argument("--data-generation", type=str, default="./datasets/reuse_high/table_1M.txt")
    parser.add_argument("--num-batches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--profiling-period", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=0, help="Number of leading batches excluded from memory/runtime simulation")
    parser.add_argument("--output-filename", type=str, default=None, help="Filename for simulation results (without extension)")

    # Output base directory
    parser.add_argument("--output-base-dir", type=str, default="results", help="Base directory for output results")

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

    # Per-batch memory simulation; warmup batches are skipped entirely.
    offmem_trace = core_onmem_obj.offmem_trace
    warmup_batches = sim_cfg.warmup_batches
    nbatches = sim_cfg.nbatches

    print("\n\n\n START OFF-CHIP MEMORY SIMULATION \n")
    helper.set_timer()

    per_batch_memory_models = []   # (batch_idx, MemoryModel) for non-warmup batches
    for nb in range(nbatches):
        if nb < warmup_batches:
            print(f"[INFO] Batch {nb}: warmup — skipping memory simulation.")
            continue

        memory_model = MemoryModel(
            sim_cfg.script_dir,
            offmem_trace[nb],          # single-batch trace slice
            sim_cfg.mnpusim_path,
            sim_cfg.dram_ini_dir,
            mnpusim_params=sim_cfg.mnpusim_params,
            global_bw_bytes_per_cycle=sim_cfg.global_buf_bw_bytes_per_cycle,
            global_latency_cycles=sim_cfg.global_onmem_latency,
            onchip_structure=sim_cfg.onchip_structure,
            local_onmem_size_kb=sim_cfg.local_onmem_size,
            mem_gran=sim_cfg.mem_gran,
            emb_dim=sim_cfg.emb_dim,
            n_format_byte=sim_cfg.n_format_byte,
            debug=sim_cfg.debug,
        )
        memory_model.do_memory_simulation()
        per_batch_memory_models.append((nb, memory_model))

    MemoryModel.print_aggregate_stats(per_batch_memory_models)

    helper.end_timer("off-chip memory simulation")
    
    #-------------------------------------------------------------------
    
    ####################################
    ### Computation Time Calculation ### (TODO: integrate memory simulation results to calculate the total runtime more accurately)
    ####################################

    helper.set_timer()
    compute_time = RuntimeModel(
        sim_cfg.workload_type,
        sim_cfg.emb_dim,
        sim_cfg.num_tables,
        sim_cfg.bsz,
        sim_cfg.num_indices_per_lookup,
        sim_cfg.n_format_byte,
        sim_cfg.vector_lanes,
        sim_cfg.vector_sublanes,
        sim_cfg.vector_alus_per_sublanes,
        sim_cfg.mxu_sa_row,
        sim_cfg.mxu_sa_col,
        sim_cfg.num_mxus,
        onchip_config={
            "onchip_structure": sim_cfg.onchip_structure,
            "local_onmem_size": sim_cfg.local_onmem_size,
            "local_onmem_latency": sim_cfg.local_onmem_latency,
            "global_onmem_size": sim_cfg.global_onmem_size,
            "global_onmem_latency": sim_cfg.global_onmem_latency,
            "onmem_type": sim_cfg.mem_type,
            "onmem_policy": sim_cfg.mem_policy,
        },
        debug=sim_cfg.debug,
    )
    compute_time.do_runtime_calculation()
    helper.end_timer("do execution time calculation")
    
    #-------------------------------------------------------------------

    ##############################################################
    ### Saving Embedding Vector Operation Simulation Results ###
    ##############################################################

    emb_csv_path = os.path.join(sim_cfg.output_dir, f"emb_results{sim_cfg.output_filename}.csv")
    try:
        with open(emb_csv_path, "w") as f:
            f.write("batch_idx,memory_cycles,compute_cycles,total_accesses,offchip_accesses,onchip_accesses,on_hit_ratio\n")
            for nb, model in per_batch_memory_models:
                if not model.execution_successful:
                    continue

                # Fetch memory and compute cycles
                mem_cycles = model.offmem_cycles
                comp_cycles = compute_time.total_compute_time_cycles

                # Fetch cache/buffer hit and miss from core_onmem_obj (handles global or local context correctly)
                batch_hit, batch_miss = core_onmem_obj.access_results[nb]
                total_accesses = batch_hit + batch_miss
                offchip_accesses = batch_miss
                onchip_accesses = batch_hit
                on_hit_ratio = batch_hit / total_accesses if total_accesses > 0 else 0.0

                f.write(f"{nb},{mem_cycles},{comp_cycles},{total_accesses},{offchip_accesses},{onchip_accesses},{on_hit_ratio:.4f}\n")
        print(f"[INFO] Saved primary embedding metrics to {emb_csv_path}")
    except Exception as e:
        print(f"[WARNING] Could not write {emb_csv_path}: {e}")
    
    #-------------------------------------------------------------------
    
    ###########################################
    ## Run Simulation for Matrix Operations ###
    ###########################################
    
    if sim_cfg.matrix_ops:
        helper.set_timer()
        print("\n[Matrix Operations Simulation]")
        topology_name = os.path.basename(args.workload_config)

        run_matrix_simulation(
            sim_cfg.matrix_ops['layers_text'],
            sim_cfg.scalesim_hw_config,
            mnk_flag=sim_cfg.matrix_ops['format'],
            topology_name=topology_name,
            output_dir=sim_cfg.output_dir,
            output_filename=sim_cfg.output_filename,
            debug=sim_cfg.debug
        )
        helper.end_timer("matrix operations simulation")
    else:
        print("[INFO] No matrix_operation section in workload config. Skipping matrix simulation.")


