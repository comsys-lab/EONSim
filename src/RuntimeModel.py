import yaml
import numpy as np
from Helper import print_styled_header, print_styled_box

class RuntimeModel:
    def __init__(self, workload_type, emb_dim, num_tables, bsz, num_indices_per_lookup, vector_lanes, vector_sublanes, vector_alus_per_sublanes, mxu_dimension, num_mxus):
        
        print("\n\n\n START RUNTIME CALCULATION \n")
        
        self.workload_type = None
        self.emb_dim = 0
        self.num_tables = 0
        self.bsz = 0
        self.num_indices_per_lookup = 0
        self.vector_lanes = 0
        self.vector_sublanes = 0
        self.vector_alus_per_sublanes = 0
        self.mxu_dimension = 0
        self.num_mxus = 0
        
        # Runtime calculation results
        self.runtime_results = []
        self.total_runtime = 0
        
        self.set_params(workload_type, emb_dim, num_tables, bsz, num_indices_per_lookup, vector_lanes, vector_sublanes, vector_alus_per_sublanes, mxu_dimension, num_mxus)
    
    def set_params(self, workload_type, emb_dim, num_tables, bsz, num_indices_per_lookup, vector_lanes, vector_sublanes, vector_alus_per_sublanes, mxu_dimension, num_mxus):
        print(f"Setting parameters for runtime model...")
        self.workload_type = workload_type
        self.emb_dim = emb_dim
        self.num_tables = num_tables
        self.bsz = bsz
        self.num_indices_per_lookup = num_indices_per_lookup
        self.vector_lanes = vector_lanes
        self.vector_sublanes = vector_sublanes
        self.vector_alus_per_sublanes = vector_alus_per_sublanes
        self.mxu_dimension = mxu_dimension
        self.num_mxus = num_mxus
    
    def do_runtime_calculation(self):
        print(f"Calculating runtime model...")
        
        # Number of operations based on workload characteristics
        if self.workload_type == "dlrm":
            # vadd
            num_vops = ((self.bsz * self.num_indices_per_lookup - 1) * np.ceil(self.emb_dim / self.vector_lanes)) # each table
            num_vops = num_vops * self.num_tables
        
        # Runtime calculation logic (assuming fully parallelized)
        self.total_runtime = np.ceil(num_vops / (self.vector_lanes * self.vector_sublanes * self.vector_alus_per_sublanes))
        
        # Store runtime results (can be extended for per-batch calculations)
        self.runtime_results.append({
            'total_runtime': self.total_runtime,
            'vector_unit_utilization': 0.0,  # Placeholder
            'matrix_unit_utilization': 0.0,  # Placeholder
            'memory_stall_cycles': 0.0       # Placeholder
        })
        
        self.print_stats()
        
    def print_stats(self):
        # print_styled_header("Runtime Model Results")
        
        # Prepare content as a list of strings
        content_lines = []
        
        # Basic configuration
        # content_lines.append(f"Workload Type: {self.workload_type}")
        # content_lines.append(f"Embedding Dimension: {self.emb_dim}")
        # content_lines.append(f"Number of Tables: {self.num_tables}")
        # content_lines.append(f"Batch Size: {self.bsz}")
        # content_lines.append(f"Number of Indices per Lookup: {self.num_indices_per_lookup}")
        # content_lines.append("")  # Empty line for spacing
        
        # Hardware configuration
        # content_lines.append("Hardware Configuration:")
        # content_lines.append(f"  Vector Unit - Lanes: {self.vector_lanes}, Sublanes: {self.vector_sublanes}, ALUs per Sublane: {self.vector_alus_per_sublanes}")
        # content_lines.append(f"  Matrix Unit - MXU Dimension: {self.mxu_dimension}, Number of MXUs: {self.num_mxus}")
        # content_lines.append("")  # Empty line for spacing
        
        # Runtime results
        content_lines.append(f"Total Runtime: {self.total_runtime} cycles")
        
        # Additional runtime metrics (placeholders for future implementation)
        # for i, result in enumerate(self.runtime_results):
        #     if len(self.runtime_results) > 1:
        #         content_lines.append(f"Batch {i} Runtime Details:")
        #     else:
        #         content_lines.append("Runtime Details:")
        #     content_lines.append(f"  Vector Unit Utilization: {result['vector_unit_utilization']:.2f}%")
        #     content_lines.append(f"  Matrix Unit Utilization: {result['matrix_unit_utilization']:.2f}%")
        #     content_lines.append(f"  Memory Stall Cycles: {result['memory_stall_cycles']:.0f}")
        
        print_styled_box("Runtime Model Results", content_lines)
    
    def print_all_config(self):
        print("\n============= Runtime Model Configuration =============")
        print(f"Workload Type: {self.workload_type}")
        print(f"Embedding Dimension: {self.emb_dim}")
        print(f"Number of Tables: {self.num_tables}")
        print(f"Batch Size: {self.bsz}")
        print(f"Number of Indices per Lookup: {self.num_indices_per_lookup}")
        
        print("\n[Hardware Configuration]")
        print("Vector Unit:")
        print(f"- Lanes: {self.vector_lanes}")
        print(f"- Sublanes: {self.vector_sublanes}")
        print(f"- ALUs per Sublane: {self.vector_alus_per_sublanes}")
        
        print("\nMatrix Unit:")
        print(f"- MXU Dimension: {self.mxu_dimension}")
        print(f"- Number of MXUs: {self.num_mxus}")
        print("=============================================\n")