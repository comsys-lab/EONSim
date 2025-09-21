from Helper import Helper, print_styled_box
from EnergyEstimator import EnergyEstimator
import argparse
import sys
import numpy as np
import os

if __name__ == "__main__":
    #################################
    ### Parse command line args   ###
    #################################
    if len(sys.argv) != 9:
        print("Usage: python3 simulator_only_energy.py <workload_type> <num_tables> <num_indices_per_lookup> <bsz> <mem_gran> <n_format_byte> <on_mem_access> <off_mem_access>")
        sys.exit(1)
        
    # Get command line arguments
    workload_type = sys.argv[1]
    num_tables = int(sys.argv[2])
    num_indices_per_lookup = int(sys.argv[3])
    bsz = int(sys.argv[4])
    mem_gran = int(sys.argv[5])
    n_format_byte = int(sys.argv[6])
    on_mem_access = int(sys.argv[7])
    off_mem_access = int(sys.argv[8])
    
    helper = Helper()
    
    #################################
    ### Run the energy estimation ###
    #################################
    helper.set_timer()
    
    # Set configuration paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workload_config_path = os.path.join(os.path.dirname(script_dir), 'configs', 'workload_config.yaml')
    energy_table_path = os.path.join(os.path.dirname(script_dir), 'configs', 'energy_estimation_table.yaml')
    
    # Prepare access results and other parameters
    access_results = [[on_mem_access, off_mem_access]]  # List of [on_mem_access, off_mem_access] for each batch
    # access_per_batch = num_tables * num_indices_per_lookup * bsz
    access_per_batch = num_indices_per_lookup
    tech_node = 45
    energy_n_format = "fp32" if n_format_byte == 4 else "int8"
    
    # Create and run energy estimator
    energy_est = EnergyEstimator(workload_type, workload_config_path, tech_node, energy_table_path, 
                                energy_n_format, access_results, access_per_batch, mem_gran)
    energy_est.print_all_config()
    energy_est.do_energy_estimation()
    
    helper.end_timer("energy estimation")