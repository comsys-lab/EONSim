import csv
import numpy as np
import os
from scale_up_sim import scale_up_runtime, scale_up_buf_access, scale_up_off_access

class accelerator:
    def __init__(self, topology_path, configuration_path, mnk_flag):
        self.topology_path = ""
        self.configuration_path = ""
        self.mnk_flag = "mnk"
        self.layer_result_table = {}  # Cache for previously simulated layers
        self.pod_result_table = {}    # Cache for previously simulated pod MNKs
        
        self.setup_params(topology_path, configuration_path, mnk_flag)
    
    def setup_params(self, topology_path, configuration_path, mnk_flag):
        self.topology_path = topology_path
        self.configuration_path = configuration_path
        self.mnk_flag = mnk_flag
        
        self.setup_topo()
        self.setup_hw()
    
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
                    
                    print(f"Processing line: {this_line}")
                    # Extract M, N, K values (indices 1, 2, 3) and convert them to integers
                    this_line = [int(this_line[1]), int(this_line[2]), int(this_line[3])]
                    self.mnk_topo.append(this_line)
        except Exception as e:
            print(f"Error reading topology file: {e}")
            
        
        self.num_layers = len(self.mnk_topo)
        print(f"Number of layers: {self.num_layers}")
        print(f"MNK Topology: {self.mnk_topo}")
    
    def setup_hw(self):
        """
        Reads hardware configuration from the specified file and stores parameters in a dictionary.
        The configuration file should have sections for NPU_others and NPU_systolic.
        """
        self.hw_config = {}
        
        try:
            with open(self.configuration_path, 'r') as config_file:
                current_section = None
                
                for line in config_file:
                    line = line.strip()
                    
                    # Skip empty lines and comments
                    if not line or line.startswith('//'):
                        continue
                    
                    # Check for section headers
                    if line.startswith('[') and line.endswith(']'):
                        current_section = line[1:-1]
                        continue
                    
                    # Process key-value pairs
                    if ':' in line:
                        key, value = [item.strip() for item in line.split(':', 1)]
                        
                        if current_section == 'NPU_others':
                            if key == 'pod_dimension_row':
                                self.hw_config['pod_row'] = int(value)
                            elif key == 'pod_dimension_col':
                                self.hw_config['pod_col'] = int(value)
                            elif key == 'clock_frequency':
                                self.hw_config['freq'] = int(value)
                            elif key == 'bandwidth':
                                self.hw_config['bw'] = int(value)
                            elif key == 'latency':
                                self.hw_config['latency'] = int(value)
                            elif key == 'dataflow':
                                self.hw_config['dataflow'] = value
                        
                        elif current_section == 'NPU_systolic':
                            if key == 'row':
                                self.hw_config['sa_row'] = int(value)
                            elif key == 'col':
                                self.hw_config['sa_col'] = int(value)
                            elif key == 'input_buffer':
                                self.hw_config['input_buf_size'] = int(value) * 1024 # Convert to unit of bytes
                            elif key == 'weight_buffer':
                                self.hw_config['weight_buf_size'] = int(value) * 1024 # Convert to unit of bytes
                            elif key == 'output_buffer':
                                self.hw_config['output_buf_size'] = int(value) * 1024 # Convert to unit of bytes
            
            # Print the configuration values for testing
            print("Hardware Configuration:")
            for key, value in self.hw_config.items():
                print(f"  {key}: {value}")
                
            # Verify that all required keys are present
            required_keys = [
                'pod_row', 'pod_col', 'freq', 'bw', 'latency', 'dataflow',
                'sa_row', 'sa_col', 'input_buf_size', 'weight_buf_size', 'output_buf_size'
            ]
            
            missing_keys = [key for key in required_keys if key not in self.hw_config]
            if missing_keys:
                print(f"Warning: Missing configuration parameters: {', '.join(missing_keys)}")
                
        except Exception as e:
            print(f"Error reading configuration file: {e}")
            self.hw_config = {}  # Reset configuration if there's an error

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
                print(f"  Layer {layer_idx + 1} is redundant. Using cached result: {cached_result}")
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
                        print(f"  Skipping empty partition [{row_idx}, {col_idx}]")
                        continue

                    cached_pod_result = self.skip_redundant_pod(this_part_mnk)
                    if cached_pod_result:
                        print(f"  Partition [{row_idx}, {col_idx}] is redundant. Using cached result: {cached_pod_result}")
                        # For runtime (index 0), take the maximum value
                        results_this_layer[0] = max(results_this_layer[0], cached_pod_result[0])
                        # For other metrics, accumulate the values
                        results_this_layer[1:] += np.array(cached_pod_result[1:])
                        continue
                        
                    print(f"  Simulating partition [{row_idx}, {col_idx}]: {this_part_mnk}")
                    results_this_part = self.do_scale_up_simulation(this_part_mnk)
                    self.pod_result_table[tuple(map(int, this_part_mnk))] = results_this_part

                    # For runtime (index 0), take the maximum value
                    results_this_layer[0] = max(results_this_layer[0], results_this_part[0])
                    # For other metrics, accumulate the values
                    results_this_layer[1:] += np.array(results_this_part[1:])
                    print(f"  Results for partition [{row_idx}, {col_idx}]: {results_this_part}")
            # Store the results for this layer
            self.total_results.append(results_this_layer)
            self.layer_result_table[layer_key] = results_this_layer.tolist()
            print(f"Results for layer {layer_idx + 1}: {results_this_layer}")
        
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
                        print("Containing empty partition")
                        this_part_mnk = None
                        
                    # Assign the partitioned layer to the appropriate position in the array
                    partitioned_this_layer[r][c] = this_part_mnk
                    print(f"Partitioned layer [{r}, {c}]: {this_part_mnk}")
        
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
                        print("Containing empty partition")
                        this_part_mnk = None
                
                    # Assign the partitioned layer to the appropriate position in the array
                    partitioned_this_layer[r][c] = this_part_mnk
                    print(f"Partitioned layer [{r}, {c}]: {this_part_mnk}")
                    
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
                        print("Containing empty partition")
                        this_part_mnk = None
                
                    # Assign the partitioned layer to the appropriate position in the array
                    partitioned_this_layer[r][c] = this_part_mnk
                    print(f"Partitioned layer [{r}, {c}]: {this_part_mnk}")
        
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
        
    def save_results(self):
        """
        Save the simulation results to a CSV file in a subdirectory named after the topology.
        """
        # Get the topology name from the path (without extension)
        topology_name = os.path.splitext(os.path.basename(self.topology_path))[0]
        
        # Set output directory to 'results/topology_name' in the parent directory of the config file
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(self.configuration_path)), 
            'results', 
            topology_name
        )
        
        # Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # output file = this_path/results/topology_name/this_configuration_name_results.csv
        output_file = os.path.join(
            output_dir, 
            self.configuration_path.split('/')[-1].replace('.cfg', '_results.csv')
        )
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['Layer', 'Runtime', 'Input Buffer Access', 'Weight Buffer Access', 'Output Buffer Access', 'Input Off-chip Access', 'Weight Off-chip Access', 'Output Off-chip Access'])
            
            for layer_idx, results in enumerate(self.total_results):
                writer.writerow([layer_idx] + list(results))
        
        print(f"Results saved to {output_file}")