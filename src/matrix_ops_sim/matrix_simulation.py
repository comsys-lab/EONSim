"Python 3.10.8"
import argparse
from . import matrix_ops_setup
import time

def run_matrix_simulation(topology_text, hw_config, mnk_flag="gemm", topology_name="workload", output_dir=None, output_filename=None, debug=False):
    start_time = time.time()

    accelerator = matrix_ops_setup.Accelerator(topology_text, hw_config, mnk_flag, topology_name, output_dir, output_filename, debug)
    accelerator.do_simulation()

    end_time = time.time()

    elapsed_time = end_time - start_time  
    print(f"Simulation completed in {elapsed_time:.5f} seconds.")