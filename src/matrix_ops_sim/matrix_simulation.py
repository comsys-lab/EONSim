"Python 3.10.8"
import argparse
from . import matrix_ops_setup
import time

def run_matrix_simulation(topology_path, hw_config, mnk_flag="gemm", output_dir=None, output_filename=None, debug=False):
    start_time = time.time()

    accelerator = matrix_ops_setup.accelerator(topology_path, hw_config, mnk_flag, output_dir, output_filename, debug)
    accelerator.do_simulation()

    end_time = time.time()

    elapsed_time = end_time - start_time  
    print(f"Simulation completed in {elapsed_time:.5f} seconds.")