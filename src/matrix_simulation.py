"Python 3.10.8"
import argparse
import matrix_ops_setup
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', required=True, help='Enter topology file path')
    parser.add_argument('-c', required=True, help='Enter configuration file path')
    parser.add_argument('-i', metavar='input type', type=str, default="gemm", help="Type of input topology, gemm: MNK, conv: conv")

    args = parser.parse_args()

    topology_path = args.t
    configuration_path = args.c
    mnk_flag = args.i

    start_time = time.time()

    accelerator = accelerator_level_setup.accelerator(topology_path, configuration_path, mnk_flag)
    accelerator.do_simulation()

    end_time = time.time()

    elapsed_time = end_time - start_time  
    print(f"Simulation completed in {elapsed_time:.5f} seconds.")