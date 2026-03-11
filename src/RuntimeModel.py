import yaml
import numpy as np
from helper_modules.Helper import print_styled_header, print_styled_box

class RuntimeModel:
    def __init__(self, workload_type, emb_dim, num_tables, bsz, num_indices_per_lookup, n_format_byte, vector_lanes, vector_sublanes, vector_alus_per_sublanes, mxu_dimension, num_mxus, onchip_config=None, debug=False):
        
        print("\n\n\n START COMPUTATION TIME CALCULATION \n")
        
        self.workload_type = None
        self.emb_dim = 0
        self.num_tables = 0
        self.bsz = 0
        self.num_indices_per_lookup = 0
        self.n_format_byte = 4
        self.vector_lanes = 0
        self.vector_sublanes = 0
        self.vector_alus_per_sublanes = 0
        self.mxu_dimension = 0
        self.num_mxus = 0
        self.debug = bool(debug)

        # On-chip config metadata (interface only)
        self.onchip_config = {}
        
        # Computation time calculation results
        self.compute_time_results = []
        self.total_compute_time_cycles = 0
        self.vload_cycles = 0
        self.vadd_cycles = 0
        
        self.set_params(workload_type, emb_dim, num_tables, bsz, num_indices_per_lookup, n_format_byte, vector_lanes, vector_sublanes, vector_alus_per_sublanes, mxu_dimension, num_mxus, onchip_config)
    
    def set_params(self, workload_type, emb_dim, num_tables, bsz, num_indices_per_lookup, n_format_byte, vector_lanes, vector_sublanes, vector_alus_per_sublanes, mxu_dimension, num_mxus, onchip_config=None):
        print(f"Setting parameters for computation time model...")
        self.workload_type = workload_type
        self.emb_dim = emb_dim
        self.num_tables = num_tables
        self.bsz = bsz
        self.num_indices_per_lookup = num_indices_per_lookup
        self.n_format_byte = n_format_byte
        self.vector_lanes = vector_lanes
        self.vector_sublanes = vector_sublanes
        self.vector_alus_per_sublanes = vector_alus_per_sublanes
        self.mxu_dimension = mxu_dimension
        self.num_mxus = num_mxus
        self.onchip_config = dict(onchip_config) if onchip_config else {}
    
    def do_runtime_calculation(self):
        print(f"Calculating computation time model...")
        
        # Number of vector additions for pooled embedding reduction.
        if self.workload_type == "dlrm":
            vectors_per_sample_per_table = self.num_indices_per_lookup
            num_vops = max(0, (vectors_per_sample_per_table - 1) * self.bsz * self.num_tables)
            num_vectors_loaded = vectors_per_sample_per_table * self.bsz * self.num_tables
        else:
            num_vops = 0
            num_vectors_loaded = 0

        vector_bytes = self.emb_dim * self.n_format_byte

        # vld from local or global buffer.
        # We assume one buffer access can fill one VREG chunk of (lanes * sublanes * 4B).
        onchip_structure = self.onchip_config.get('onchip_structure', 'local_only')
        if onchip_structure == "global_only":
            buffer_access_latency = self.onchip_config.get('global_onmem_latency', 0)
        else:
            buffer_access_latency = self.onchip_config.get('local_onmem_latency', 0)

        vreg_bytes = self.vector_lanes * self.vector_sublanes * 4
        total_vector_bytes = num_vectors_loaded * vector_bytes
        if vreg_bytes > 0 and buffer_access_latency > 0:
            self.vload_cycles = int(np.ceil((buffer_access_latency * total_vector_bytes) / vreg_bytes))
        else:
            self.vload_cycles = 0

        # num_vops is vector-count based; convert to FP32-op count before ALU throughput scaling.
        num_ops = num_vops * (vector_bytes / 4.0)
        fp32_parallel_ops = self.vector_lanes * self.vector_sublanes * self.vector_alus_per_sublanes
        self.vadd_cycles = int(np.ceil(num_ops / fp32_parallel_ops)) if fp32_parallel_ops > 0 else 0

        self.total_compute_time_cycles = self.vload_cycles + self.vadd_cycles
        
        # Store computation time results (can be extended for per-batch calculations)
        self.compute_time_results.append({
            'total_compute_time_cycles': self.total_compute_time_cycles,
            'vload_cycles': self.vload_cycles,
            'vadd_cycles': self.vadd_cycles,
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
        
        # Computation time results
        content_lines.append(f"Total Computation Time: {self.total_compute_time_cycles} cycles")
        if self.debug:
            content_lines.append(f"Vector Load Cycles: {self.vload_cycles} cycles")
            content_lines.append(f"Vector Add Cycles: {self.vadd_cycles} cycles")
        
        # Additional runtime metrics (placeholders for future implementation)
        # for i, result in enumerate(self.runtime_results):
        #     if len(self.runtime_results) > 1:
        #         content_lines.append(f"Batch {i} Runtime Details:")
        #     else:
        #         content_lines.append("Runtime Details:")
        #     content_lines.append(f"  Vector Unit Utilization: {result['vector_unit_utilization']:.2f}%")
        #     content_lines.append(f"  Matrix Unit Utilization: {result['matrix_unit_utilization']:.2f}%")
        #     content_lines.append(f"  Memory Stall Cycles: {result['memory_stall_cycles']:.0f}")
        
        print_styled_box("Computation Time Model Results", content_lines)
    
    def print_all_config(self):
        print("\n============= Computation Time Model Configuration =============")
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