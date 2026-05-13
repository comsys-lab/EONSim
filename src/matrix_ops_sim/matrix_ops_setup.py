import csv
import numpy as np
import os
from . import matrix_single_core_sim
from .matrix_single_core_sim import scale_up_runtime, scale_up_buf_access, scale_up_off_access

class accelerator:
    def __init__(self, topology_path, hw_config, mnk_flag, output_dir=None, output_filename=None, debug=False):
        self.topology_path = ""
        self.mnk_flag = "mnk"
        self.output_dir = output_dir
        self.output_filename = output_filename
        self.layer_result_table = {}  # Cache for previously simulated layers
        self.pod_result_table = {}    # Cache for previously simulated pod MNKs

        self.debug = debug
        self.dprint = print if debug else lambda *a, **k: None
        matrix_single_core_sim.DEBUG = debug

        self.setup_params(topology_path, hw_config, mnk_flag)

    def setup_params(self, topology_path, hw_config, mnk_flag):
        self.topology_path = topology_path
        self.mnk_flag = mnk_flag

        self.setup_topo()
        self.setup_hw(hw_config)
    
    def conv_to_mnk(self, this_line):
        ''' 
        Topology of CNNs is composed of [layer_name, Input_W, Input_H, Filter_W, Filter_H, Channel, Num_filter, Stride]
        Converting such topology to MNK format
        '''
        # Convert string values to integers for calculations
        input_w = int(this_line[1])
        input_h = int(this_line[2])
        filter_w = int(this_line[3])
        filter_h = int(this_line[4])
        channel = int(this_line[5])
        num_filter = int(this_line[6])
        stride = int(this_line[7])

        output_row = int(np.ceil((input_w - filter_w + stride) / stride))
        output_col = int(np.ceil((input_h - filter_h + stride) / stride))

        os_input_row = output_row * output_col
        os_input_col = filter_w * filter_h * channel
        os_filter_col = num_filter
        
        # Return the MNK values
        mnk = ["none", os_input_row, os_filter_col, os_input_col]

        return mnk
    
    def setup_topo(self):
        self.mnk_topo = []
        
        try:
            with open(self.topology_path, 'r') as topo_file:
                csv_reader = csv.reader(topo_file)
                
                for row in csv_reader:
                    # Skip empty lines
                    if not row or all(cell.strip() == "" for cell in row):
                        continue
                    
                    if self.mnk_flag == "conv":
                        # Ensure we have at least 8 fields for conv
                        this_line = (row + [""] * 8)[:8]
                        # Call conv_to_mnk to convert conv params to MNK format
                        this_line = self.conv_to_mnk(this_line)
                    elif self.mnk_flag == "gemm" or self.mnk_flag == "mnk":
                        # Ensure we have at least 4 fields for gemm/mnk
                        this_line = (row + [""] * 4)[:4]
                    
                    self.dprint(f"Processing line: {this_line}")
                    # Extract M, N, K values (indices 1, 2, 3) and convert them to integers
                    this_line = [int(this_line[1]), int(this_line[2]), int(this_line[3])]
                    self.mnk_topo.append(this_line)
        except Exception as e:
            print(f"Error reading topology file: {e}")
            
        
        self.num_layers = len(self.mnk_topo)
        self.dprint(f"Number of layers: {self.num_layers}")
        self.dprint(f"MNK Topology: {self.mnk_topo}")
    
    def setup_hw(self, hw_config):
        self.hw_config = dict(hw_config) if hw_config else {}

        self.dprint("Hardware Configuration:")
        for key, value in self.hw_config.items():
            self.dprint(f"  {key}: {value}")

        required_keys = [
            'pod_row', 'pod_col', 'freq', 'bw', 'latency', 'dataflow',
            'sa_row', 'sa_col', 'input_buf_size', 'weight_buf_size', 'output_buf_size',
            'global_buf_size',
        ]

        missing_keys = [key for key in required_keys if key not in self.hw_config]
        if missing_keys:
            raise ValueError(f"Missing hw_config parameters: {', '.join(missing_keys)}")

    def skip_redundant_layer(self, this_layer):
        # Convert the MNK values of the current layer to a tuple of integers to use as a hashable key
        key = tuple(map(int, this_layer))

        # Check if the result for this exact MNK configuration has already been computed and stored
        # If yes, return the cached result to skip redundant simulation
        # If not, return None, which means simulation should proceed for this layer
        return self.layer_result_table.get(key)

    def skip_redundant_pod(self, this_part_mnk):
        # Convert the MNK values of the current pod partition to a tuple of integers
        key = tuple(map(int, this_part_mnk))

        # Check if this exact MNK configuration has already been simulated across any pod
        # If cached result exists, return it to avoid re-running the simulation for this pod
        # Otherwise, return None indicating simulation is needed
        return self.pod_result_table.get(key)

    def do_simulation(self):
        """
        Processes each layer in mnk_topo by:
        1. Getting the layer data
        2. Partitioning it using scale_out_partitioning
        3. Calling do_scale_up_simulation for each partition
        """
        if not hasattr(self, 'mnk_topo') or not self.mnk_topo:
            print("No topology data available for simulation")
            return
        
        if not hasattr(self, 'hw_config') or not self.hw_config:
            print("No hardware configuration available for simulation")
            return
        
        
        self.total_results = []
        
        print(f"Starting simulation for {len(self.mnk_topo)} layers...")
        
        for layer_idx, this_layer in enumerate(self.mnk_topo):
            print(f"Processing layer {layer_idx + 1}: {this_layer}")
            
            layer_key = tuple(map(int, this_layer))
            cached_result = self.skip_redundant_layer(this_layer)
            if cached_result:
                self.dprint(f"  Layer {layer_idx + 1} is redundant. Using cached result: {cached_result}")
                self.total_results.append(np.array(cached_result))
                continue

            # results_this_layer = [runtime, input_buf_access, weight_buf_access, output_buf_access, input_off_access, weight_off_access, output_off_access]
            # results_this_layer is used to accumulate results for this layer across all pods
            # Initialize results for this layer
            results_this_layer = np.zeros(7, dtype=int)
            
            # Partition the layer across multiple pods
            partitioned_this_layer = self.scale_out_partitioning(this_layer)
            
            # Process each partition with do_scale_up_simulation
            for row_idx in range(self.hw_config['pod_row']):
                for col_idx in range(self.hw_config['pod_col']):
                    this_part_mnk = partitioned_this_layer[row_idx][col_idx]
                    
                    # Skip empty partitions
                    if this_part_mnk is None:
                        self.dprint(f"  Skipping empty partition [{row_idx}, {col_idx}]")
                        continue

                    cached_pod_result = self.skip_redundant_pod(this_part_mnk)
                    if cached_pod_result:
                        self.dprint(f"  Partition [{row_idx}, {col_idx}] is redundant. Using cached result: {cached_pod_result}")
                        # For runtime (index 0), take the maximum value
                        results_this_layer[0] = max(results_this_layer[0], cached_pod_result[0])
                        # For other metrics, accumulate the values
                        results_this_layer[1:] += np.array(cached_pod_result[1:])
                        continue
                        
                    self.dprint(f"  Simulating partition [{row_idx}, {col_idx}]: {this_part_mnk}")
                    results_this_part = self.do_scale_up_simulation(this_part_mnk)
                    self.pod_result_table[tuple(map(int, this_part_mnk))] = results_this_part

                    # For runtime (index 0), take the maximum value
                    results_this_layer[0] = max(results_this_layer[0], results_this_part[0])
                    
                    # For other metrics, accumulate the values
                    results_this_layer[1:] += np.array(results_this_part[1:])
                    
                    # Apply memory model to runtime
                    results_this_layer[0] = self.this_layer_memory_model(results_this_layer)
                    
                    self.dprint(f"  Results for partition [{row_idx}, {col_idx}]: {results_this_part}")
            # Store the results for this layer
            self.total_results.append(results_this_layer)
            self.layer_result_table[layer_key] = results_this_layer.tolist()
            self.dprint(f"Results for layer {layer_idx + 1}: {results_this_layer}")
        
        print("Simulation complete")
        
        self.save_results()
    
    def scale_out_partitioning(self, this_layer):
        """
        Partitions a layer across multiple pods.
        
        Args:
            this_layer: List containing M, N, K values
            
        Returns:
            A 2D array of shape [pod_row, pod_col] where each element contains MNK data
        """
        self.num_pods = self.hw_config['pod_row'] * self.hw_config['pod_col']
        pod_row = self.hw_config['pod_row']
        pod_col = self.hw_config['pod_col']
        
        # Create a 2D array of objects instead of zeros
        partitioned_this_layer = np.empty((self.hw_config['pod_row'], self.hw_config['pod_col']), dtype=object)
        
        # Initialize all positions with None
        for r in range(self.hw_config['pod_row']):
            for c in range(self.hw_config['pod_col']):
                partitioned_this_layer[r][c] = None
        
        # Ensure all values in this_layer are integers
        this_layer = [int(val) for val in this_layer]
        
        if self.hw_config['dataflow'] == 'OS':
            row = this_layer[0] # M
            col = this_layer[1] # N
            matrix_rows_per_part = int(np.ceil(row / pod_row)) # Height of each tile
            matrix_cols_per_part = int(np.ceil(col / pod_col)) # Width of each tile
            
            for r in range(pod_row):
                for c in range(pod_col):
                    # Create a new partitioned layer for this pod
                    this_part_mnk = [this_layer[0], this_layer[1], this_layer[2]]
                    
                    # Calculate the start and end indices for the current pod
                    start_row = r * matrix_rows_per_part
                    end_row = min((r + 1) * matrix_rows_per_part, int(this_layer[0]))
                    
                    start_col = c * matrix_cols_per_part
                    end_col = min((c + 1) * matrix_cols_per_part, int(this_layer[1]))               
                    
                    # Set the partitioned values
                    this_part_mnk[0] = end_row - start_row
                    this_part_mnk[1] = end_col - start_col
                    if (this_part_mnk[0] <= 0) or (this_part_mnk[1] <= 0):
                        self.dprint("Containing empty partition")
                        this_part_mnk = None
                        
                    # Assign the partitioned layer to the appropriate position in the array
                    partitioned_this_layer[r][c] = this_part_mnk
                    self.dprint(f"Partitioned layer [{r}, {c}]: {this_part_mnk}")
        
        elif self.hw_config['dataflow'] == 'WS':
            row = this_layer[2] # K
            col = this_layer[1] # N
            matrix_rows_per_part = int(np.ceil(row / pod_row)) # Height of each tile
            matrix_cols_per_part = int(np.ceil(col / pod_col)) # Width of each tile
            
            for r in range(pod_row):
                for c in range(pod_col):
                    # Create a new partitioned layer for this pod
                    this_part_mnk = [this_layer[0], this_layer[1], this_layer[2]]
                    
                    # Calculate the start and end indices for the current pod
                    start_row = r * matrix_rows_per_part
                    end_row = min((r + 1) * matrix_rows_per_part, int(this_layer[2]))
                    
                    start_col = c * matrix_cols_per_part
                    end_col = min((c + 1) * matrix_cols_per_part, int(this_layer[1]))               
                    
                    # Set the partitioned values
                    this_part_mnk[2] = end_row - start_row
                    this_part_mnk[1] = end_col - start_col
                    if (this_part_mnk[2] <= 0) or (this_part_mnk[1] <= 0):
                        self.dprint("Containing empty partition")
                        this_part_mnk = None
                
                    # Assign the partitioned layer to the appropriate position in the array
                    partitioned_this_layer[r][c] = this_part_mnk
                    self.dprint(f"Partitioned layer [{r}, {c}]: {this_part_mnk}")
                    
        elif self.hw_config['dataflow'] == 'IS':
            row = this_layer[2] # K
            col = this_layer[0] # M
            matrix_rows_per_part = int(np.ceil(row / pod_row)) # Height of each tile
            matrix_cols_per_part = int(np.ceil(col / pod_col)) # Width of each tile
            
            for r in range(pod_row):
                for c in range(pod_col):
                    # Create a new partitioned layer for this pod
                    this_part_mnk = [this_layer[0], this_layer[1], this_layer[2]]
                    
                    # Calculate the start and end indices for the current pod
                    start_row = r * matrix_rows_per_part
                    end_row = min((r + 1) * matrix_rows_per_part, int(this_layer[2]))
                    
                    start_col = c * matrix_cols_per_part
                    end_col = min((c + 1) * matrix_cols_per_part, int(this_layer[0]))               
                    
                    # Set the partitioned values
                    this_part_mnk[2] = end_row - start_row
                    this_part_mnk[0] = end_col - start_col
                    if (this_part_mnk[2] <= 0) or (this_part_mnk[0] <= 0):
                        self.dprint("Containing empty partition")
                        this_part_mnk = None
                
                    # Assign the partitioned layer to the appropriate position in the array
                    partitioned_this_layer[r][c] = this_part_mnk
                    self.dprint(f"Partitioned layer [{r}, {c}]: {this_part_mnk}")
        
        return partitioned_this_layer
    
    def do_scale_up_simulation(self, this_part_mnk):
        this_runtime = 0
        this_input_buf_access = 0
        this_weight_buf_access = 0
        this_output_buf_access = 0
        this_input_off_access = 0
        this_weight_off_access = 0
        this_output_off_access = 0
        
        # Calculate results using the analytical model
        this_runtime = scale_up_runtime(this_part_mnk, self.hw_config)
        this_buf_access = scale_up_buf_access(this_part_mnk, self.hw_config)
        this_off_access = scale_up_off_access(this_part_mnk, self.hw_config)
        
        this_input_buf_access = this_buf_access[0]
        this_weight_buf_access = this_buf_access[1]
        this_output_buf_access = this_buf_access[2]
        this_input_off_access = this_off_access[0]
        this_weight_off_access = this_off_access[1]
        this_output_off_access = this_off_access[2]
        
        results_this_part = (
            this_runtime, this_input_buf_access, this_weight_buf_access,
            this_output_buf_access, this_input_off_access, this_weight_off_access,
            this_output_off_access
        )
        
        return results_this_part
        
    def this_layer_memory_model(self, results_this_layer):
        runtime_this_layer = results_this_layer[0]
        
        # 1. Calculate data transfer sizes (assuming 1 byte per element)
        input_off_access = results_this_layer[4]
        weight_off_access = results_this_layer[5]
        
        # Assuming int8 (1 byte)
        input_transfer_size = input_off_access * 1
        weight_transfer_size = weight_off_access * 1
        
        # 2. Divide by pod dimensions to remove redundant fetches
        # Input is shared across columns (broadcast to row), Weight is shared across rows (broadcast to col)
        input_transfer_size = input_transfer_size / self.hw_config['pod_col']
        weight_transfer_size = weight_transfer_size / self.hw_config['pod_row']
        
        # 3. Calculate number of transfers based on global buffer size
        gbuf_size = self.hw_config['global_buf_size']
        inactive_gbuf_size = gbuf_size / 2
        total_transfer_data = input_transfer_size + weight_transfer_size
        
        if inactive_gbuf_size > 0:
            num_transfer = np.ceil(total_transfer_data / inactive_gbuf_size)
        else:
            num_transfer = 1
            
        # 4. Calculate data transfer time
        # data transfer time = Num_transfer * L + (input+weight)/BW
        latency = self.hw_config['latency']
        
        # Calculate BW in Bytes/Cycle
        # bw (GB/s) -> Bytes/s: bw * 1e9
        # freq (MHz) -> Cycles/s: freq * 1e6
        # Bytes/Cycle = (bw * 1e9) / (freq * 1e6)
        bw_gbps = self.hw_config['bw']
        freq_mhz = self.hw_config['freq']
        
        if freq_mhz > 0:
            bw_bytes_per_cycle = (bw_gbps * 1e9) / (freq_mhz * 1e6)
        else:
            bw_bytes_per_cycle = 0
        
        # Data transfer part in cycles
        if bw_bytes_per_cycle > 0:
            data_time_cycles = total_transfer_data / bw_bytes_per_cycle
        else:
            data_time_cycles = 0
            
        transfer_cycles = (num_transfer * latency) + data_time_cycles
        
        # 5. Compare with compute time
        if transfer_cycles > runtime_this_layer:
            self.dprint(f"    Memory Bound: Transfer {int(transfer_cycles)} > Compute {int(runtime_this_layer)}")
            runtime_this_layer = transfer_cycles
        else:
            self.dprint(f"    Compute Bound: Compute {int(runtime_this_layer)} >= Transfer {int(transfer_cycles)}")
        
        return runtime_this_layer

    def save_results(self):
        topology_name = os.path.splitext(os.path.basename(self.topology_path))[0]

        if self.output_dir:
            output_dir = self.output_dir
        else:
            output_dir = os.path.join('results', topology_name)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        suffix = self.output_filename if self.output_filename else ""
        output_file = os.path.join(output_dir, f"matrix_results{suffix}.csv")
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Layer', 'Runtime', 'Input Buffer Access', 'Weight Buffer Access', 'Output Buffer Access', 'Input Off-chip Access', 'Weight Off-chip Access', 'Output Off-chip Access'])
            
            for layer_idx, results in enumerate(self.total_results):
                writer.writerow([layer_idx] + list(results))
        
        print(f"Results saved to {output_file}")