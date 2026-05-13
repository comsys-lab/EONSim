import yaml
import numpy as np
from helper_modules.Helper import print_styled_header, print_styled_box

class RuntimeModel:
    def __init__(self, workload_type, emb_dim, num_tables, bsz, num_indices_per_lookup, n_format_byte, vector_lanes, vector_sublanes, vector_alus_per_sublanes, mxu_sa_row, mxu_sa_col, num_mxus, onchip_config=None, debug=False):
        print("\n\n\n START COMPUTATION TIME CALCULATION \n")

        self.debug = bool(debug)
        self.workload_type = workload_type
        self.emb_dim = emb_dim
        self.num_tables = num_tables
        self.bsz = bsz
        self.num_indices_per_lookup = num_indices_per_lookup
        self.n_format_byte = n_format_byte
        self.vector_lanes = vector_lanes
        self.vector_sublanes = vector_sublanes
        self.vector_alus_per_sublanes = vector_alus_per_sublanes
        self.mxu_sa_row = mxu_sa_row
        self.mxu_sa_col = mxu_sa_col
        self.num_mxus = num_mxus
        self.onchip_config = dict(onchip_config) if onchip_config else {}

        self.compute_time_results = []
        self.total_compute_time_cycles = 0
        self.vload_cycles = 0
        self.vadd_cycles = 0
        self.vector_compute_cycles = 0
        self.num_candidate_vectors = 0
        self.l2_sub_ops = 0
        self.l2_mul_ops = 0
        self.l2_acc_ops = 0
    
    def do_runtime_calculation(self):
        print(f"Calculating computation time model...")

        # Reset per-run derived stats.
        self.vload_cycles = 0
        self.vadd_cycles = 0
        self.vector_compute_cycles = 0
        self.num_candidate_vectors = 0
        self.l2_sub_ops = 0
        self.l2_mul_ops = 0
        self.l2_acc_ops = 0

        vector_bytes = self.emb_dim * self.n_format_byte

        # vld from local or global buffer.
        # We assume one buffer access can fill one VREG chunk of (lanes * sublanes * 4B).
        onchip_structure = self.onchip_config.get('onchip_structure', 'local_only')
        if onchip_structure == "global_only":
            buffer_access_latency = self.onchip_config.get('global_onmem_latency', 0)
        else:
            buffer_access_latency = self.onchip_config.get('local_onmem_latency', 0)

        vreg_bytes = self.vector_lanes * self.vector_sublanes * 4
        fp32_parallel_ops = self.vector_lanes * self.vector_sublanes * self.vector_alus_per_sublanes

        # Number of vector additions for pooled embedding reduction.
        if self.workload_type == "dlrm":
            vectors_per_sample_per_table = self.num_indices_per_lookup
            num_vops = max(0, (vectors_per_sample_per_table - 1) * self.bsz * self.num_tables)
            num_vectors_loaded = vectors_per_sample_per_table * self.bsz * self.num_tables

            total_vector_bytes = num_vectors_loaded * vector_bytes
            if vreg_bytes > 0 and buffer_access_latency > 0:
                self.vload_cycles = int(np.ceil((buffer_access_latency * total_vector_bytes) / vreg_bytes))
            else:
                self.vload_cycles = 0

            # num_vops is vector-count based; convert to FP32-op count before ALU throughput scaling.
            num_ops = num_vops * (vector_bytes / 4.0)
            self.vadd_cycles = int(np.ceil(num_ops / fp32_parallel_ops)) if fp32_parallel_ops > 0 else 0
            self.vector_compute_cycles = self.vadd_cycles

        elif self.workload_type == "vectordb":
            # Candidate vectors compared against query vectors.
            self.num_candidate_vectors = self.bsz * self.num_tables * self.num_indices_per_lookup

            # Query vector load is modeled with reuse per (sample, table), while candidates are loaded per comparison.
            candidate_bytes = self.num_candidate_vectors * vector_bytes
            query_bytes = self.bsz * self.num_tables * vector_bytes
            total_vector_bytes = candidate_bytes + query_bytes

            if vreg_bytes > 0 and buffer_access_latency > 0:
                self.vload_cycles = int(np.ceil((buffer_access_latency * total_vector_bytes) / vreg_bytes))
            else:
                self.vload_cycles = 0

            # L2 distance per element: sub + mul + accumulation.
            self.l2_sub_ops = self.num_candidate_vectors * self.emb_dim
            self.l2_mul_ops = self.num_candidate_vectors * self.emb_dim
            self.l2_acc_ops = self.num_candidate_vectors * max(0, self.emb_dim - 1)
            num_ops = self.l2_sub_ops + self.l2_mul_ops + self.l2_acc_ops
            self.vector_compute_cycles = int(np.ceil(num_ops / fp32_parallel_ops)) if fp32_parallel_ops > 0 else 0

        else:
            self.vload_cycles = 0
            self.vadd_cycles = 0
            self.vector_compute_cycles = 0

        self.total_compute_time_cycles = self.vload_cycles + self.vector_compute_cycles
        
        # Store computation time results (can be extended for per-batch calculations)
        self.compute_time_results.append({
            'total_compute_time_cycles': self.total_compute_time_cycles,
            'vload_cycles': self.vload_cycles,
            'vadd_cycles': self.vadd_cycles,
            'vector_compute_cycles': self.vector_compute_cycles,
            'num_candidate_vectors': self.num_candidate_vectors,
            'l2_sub_ops': self.l2_sub_ops,
            'l2_mul_ops': self.l2_mul_ops,
            'l2_acc_ops': self.l2_acc_ops
        })
        
        self.print_stats()
        
    def print_stats(self):
        
        # Prepare content as a list of strings
        content_lines = []
        
        # Computation time results
        content_lines.append(f"Total Computation Time: {self.total_compute_time_cycles} cycles")
        if self.debug:
            content_lines.append(f"Vector Load Cycles: {self.vload_cycles} cycles")
            if self.workload_type == "vectordb":
                content_lines.append(f"L2 Compute Cycles: {self.vector_compute_cycles} cycles")
                content_lines.append(f"Candidate Vectors: {self.num_candidate_vectors}")
                content_lines.append(f"L2 Sub Ops: {self.l2_sub_ops}")
                content_lines.append(f"L2 Mul Ops: {self.l2_mul_ops}")
                content_lines.append(f"L2 Acc Ops: {self.l2_acc_ops}")
            else:
                content_lines.append(f"Vector Add Cycles: {self.vadd_cycles} cycles")
        
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
        print(f"- MXU Dimension: {self.mxu_sa_row} x {self.mxu_sa_col}")
        print(f"- Number of MXUs: {self.num_mxus}")
        print("=============================================\n")