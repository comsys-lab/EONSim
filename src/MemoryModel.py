import os
import subprocess
import shutil
import math
import tempfile
from helper_modules.Helper import print_styled_box

class MemoryModel:
    def __init__(
        self,
        script_dir,
        offmem_trace,
        mnpusim_path,
        dram_ini_dir,
        mnpusim_params,
        global_bw_bytes_per_cycle=0,
        global_latency_cycles=0,
        onchip_structure="global_only",
        local_onmem_size_kb=0,
        mem_gran=0,
        emb_dim=0,
        n_format_byte=0,
        debug=False,
    ):
        # offmem_trace: single-batch trace, i.e. list of per-table address arrays.
        self.script_dir = script_dir
        self.offmem_trace = offmem_trace
        self.intermediate_dir = None
        self.mnpusim_path = mnpusim_path
        self.dram_ini_dir = dram_ini_dir
        self.mnpusim_params = dict(mnpusim_params) if mnpusim_params else {}
        self.eonsim_results_dir = None
        self.eonsim_results_dir_name = None
        self.eonsim_config_dir = None
        self.eonsim_dir_name = None
        # Relative paths (from mNPUsim root) to the cfg files generated per run.
        self.generated_dram_config_relpath = None
        self.generated_npumem_list_relpath = None
        self.debug = debug

        # Analytical model parameters.
        self.offchip_issue_width = 1   # off-chip DMA issue width; increase to model wider DMA engines
        self.global_issue_width = 1    # global-to-local DMA issue width
        self.global_bw_bytes_per_cycle = float(global_bw_bytes_per_cycle) if global_bw_bytes_per_cycle else 0.0
        self.global_latency_cycles = int(global_latency_cycles) if global_latency_cycles else 0
        self.onchip_structure = onchip_structure
        self.local_onmem_size_kb = int(local_onmem_size_kb) if local_onmem_size_kb else 0
        self.mem_gran = int(mem_gran) if mem_gran else 0
        self.emb_dim = int(emb_dim) if emb_dim else 0
        self.n_format_byte = int(n_format_byte) if n_format_byte else 0

        # Results
        self.offmem_cycles = 0
        self.offmem_cycles_raw = 0
        self.analytical_results = {}
        self.execution_successful = False

    def _flatten_batch_trace(self):
        # Flatten single-batch trace (list of per-table arrays) into a single address list.
        if not self.offmem_trace:
            return []
        return [addr for table_trace in self.offmem_trace for addr in table_trace]

    def _calc_access_per_vector(self):
        if self.mem_gran <= 0 or self.emb_dim <= 0 or self.n_format_byte <= 0:
            return 1
        return max(1, int(math.ceil((self.emb_dim * self.n_format_byte) / self.mem_gran)))

    def _count_trace_types(self):
        flat_trace = self._flatten_batch_trace()
        # `-1` entries represent accesses satisfied by on-chip memory.
        # For analytical memory-cycle modeling, global stage is only modeled in `two_level`.
        onchip_hit_accesses = sum(1 for addr in flat_trace if addr == -1)
        offchip_accesses = len(flat_trace) - onchip_hit_accesses

        if self.onchip_structure == "two_level":
            global_to_local_accesses = onchip_hit_accesses
        else:
            global_to_local_accesses = 0

        return len(flat_trace), global_to_local_accesses, offchip_accesses

    def _calc_vector_count(self, total_access_count):
        access_per_vector = self._calc_access_per_vector()
        if total_access_count <= 0:
            return 0
        return int(math.ceil(total_access_count / access_per_vector))

    def _calc_dma_requests(self, access_count):
        access_per_vector = self._calc_access_per_vector()
        return int(math.ceil(access_count / access_per_vector)) if access_count > 0 else 0

    def _calc_issue_cycles(self, request_count, issue_width):
        if request_count <= 0:
            return 0
        return int(math.ceil(request_count / max(1, issue_width)))

    def _resolve_chunk_accesses(self):
        if self.mem_gran <= 0:
            return 0

        if self.local_onmem_size_kb > 0:
            half_local_bytes = (self.local_onmem_size_kb * 1024) // 2
            if half_local_bytes > 0:
                return max(1, half_local_bytes // self.mem_gran)

        return 0

    def _calc_global_transfer_cycles(self):
        if self.global_bw_bytes_per_cycle <= 0 or self.mem_gran <= 0:
            return 0

        flat_trace = self._flatten_batch_trace()
        if not flat_trace:
            return 0

        chunk_accesses = self._resolve_chunk_accesses()
        if chunk_accesses <= 0:
            chunk_accesses = len(flat_trace)

        total_cycles = 0
        for start in range(0, len(flat_trace), chunk_accesses):
            chunk = flat_trace[start:start + chunk_accesses]
            chunk_global_to_local_accesses = sum(1 for addr in chunk if addr == -1)
            if chunk_global_to_local_accesses == 0:
                continue
            chunk_bytes = chunk_global_to_local_accesses * self.mem_gran
            transfer_cycles = self.global_latency_cycles + int(math.ceil(chunk_bytes / self.global_bw_bytes_per_cycle))
            total_cycles += transfer_cycles

        return total_cycles

    def _apply_analytical_model(self):
        total_access_count, global_to_local_accesses, offchip_accesses = self._count_trace_types()
        total_vector_count = self._calc_vector_count(total_access_count)

        offchip_requests = self._calc_dma_requests(offchip_accesses)
        global_to_local_requests = self._calc_dma_requests(global_to_local_accesses)

        offchip_issue_cycles = self._calc_issue_cycles(offchip_requests, self.offchip_issue_width)
        global_to_local_issue_cycles = self._calc_issue_cycles(global_to_local_requests, self.global_issue_width)

        offchip_total_cycles = max(self.offmem_cycles_raw, offchip_issue_cycles)

        # Analytical Local Buffer Access calculation
        local_buffer_accesses = 0
        if self.onchip_structure == "two_level":
            local_buffer_accesses = total_vector_count * self._calc_access_per_vector()

        global_to_local_transfer_cycles = 0
        global_to_local_total_cycles = 0
        if self.onchip_structure == "two_level" and global_to_local_accesses > 0:
            global_to_local_transfer_cycles = self._calc_global_transfer_cycles()
            global_to_local_total_cycles = max(global_to_local_issue_cycles, global_to_local_transfer_cycles)
            # two_level: memory bottleneck is max(off-chip path, global-to-local path)
            memory_cycles_final = max(offchip_total_cycles, global_to_local_total_cycles)
        else:
            # local_only/global_only: memory cycle is off-chip path only.
            memory_cycles_final = offchip_total_cycles

        self.analytical_results = {
            "offchip_accesses": offchip_accesses,
            "total_vector_count": total_vector_count,
            "local_buffer_accesses": local_buffer_accesses,
            "global_to_local_accesses": global_to_local_accesses,
            "offchip_requests": offchip_requests,
            "global_to_local_requests": global_to_local_requests,
            "offchip_issue_cycles": offchip_issue_cycles,
            "global_to_local_issue_cycles": global_to_local_issue_cycles,
            "offchip_cycles_raw": self.offmem_cycles_raw,
            "offchip_total_cycles": offchip_total_cycles,
            "global_to_local_transfer_cycles": global_to_local_transfer_cycles,
            "global_to_local_total_cycles": global_to_local_total_cycles,
            "memory_cycles_final": memory_cycles_final,
        }
        self.offmem_cycles = memory_cycles_final
        
    def setup_intermediate_directory(self):
        """Create a unique intermediate directory for this run."""
        self.intermediate_dir = tempfile.mkdtemp(prefix="intermediate_")
        if self.debug: print(f"[DEBUG] Created intermediate directory: {self.intermediate_dir}")
        
    def setup_eonsim_results_directory(self):
        """Create a unique mNPUsim results directory for this run."""
        self.eonsim_results_dir = tempfile.mkdtemp(prefix="eonsim_results_", dir=self.mnpusim_path)
        self.eonsim_results_dir_name = os.path.basename(self.eonsim_results_dir)
        
        if self.debug: print(f"[DEBUG] Created eonsim_results directory name: {self.eonsim_results_dir_name}")
        
    def generate_trace_file(self):
        """Generate flattened trace file for mNPUsim"""
        flat_offmem_trace = self._flatten_batch_trace()
        offmem_trace_path = os.path.join(self.intermediate_dir, "offmem_trace_flat.txt")
        
        access_per_vector = self._calc_access_per_vector()
        
        with open(offmem_trace_path, "w") as f:
            f.write("0,")  # Initial dummy value for mNPUsim
            
            # Process the trace
            for i in range(0, len(flat_offmem_trace), access_per_vector):
                f.write("-1,") # index lookup is pipelined, immediatly fetch the target vector in next cycle
                                
                # If the first element is -1, this request goes to the global buffer
                vector_for_check = flat_offmem_trace[i : i + access_per_vector]
                if vector_for_check[0] == -1:
                    f.write("-1,")
                else:
                    # If miss, write all addresses for off-chip memory requests
                    for addr in vector_for_check:
                        # Fallback for potential partial hits (if applicable)
                        if addr == -1:
                            f.write("-1,")
                        else:
                            f.write(f"{addr},")
                            
        if self.debug: print(f"[DEBUG] Generated trace file: {offmem_trace_path}")
        
        return offmem_trace_path
        
    def setup_eonsim_config_directory(self):
        """Create per-run config dir. Symlink the DRAMsim3 .ini tree; generate everything else."""
        self.eonsim_config_dir = tempfile.mkdtemp(prefix="eonsim_config_", dir=self.mnpusim_path)
        self.eonsim_dir_name = os.path.basename(self.eonsim_config_dir)

        # Symlink the DRAMsim3 .ini directory into the per-run config tree.
        if not os.path.exists(self.dram_ini_dir):
            raise FileNotFoundError(f"DRAM .ini directory not found: {self.dram_ini_dir}")
        dram_config_dir = os.path.join(self.eonsim_config_dir, "dram_config")
        os.makedirs(dram_config_dir, exist_ok=True)
        os.symlink(
            os.path.abspath(self.dram_ini_dir),
            os.path.join(dram_config_dir, "single_dram_config"),
        )
        if self.debug:
            print(f"[DEBUG] Symlinked single_dram_config -> {self.dram_ini_dir}")

        # Empty dirs for generated cfg/txt files.
        os.makedirs(os.path.join(dram_config_dir, "total_dram_config"), exist_ok=True)
        npumem_dir = os.path.join(self.eonsim_config_dir, "npumem_config")
        os.makedirs(os.path.join(npumem_dir, "npumem_architecture"), exist_ok=True)
        os.makedirs(os.path.join(npumem_dir, "npumem_architecture_list"), exist_ok=True)

        self.generated_dram_config_relpath = self._generate_dram_config()
        self.generated_npumem_list_relpath = self._generate_npumem_config()

    def _generate_dram_config(self):
        """Write total_dram_config/generated.cfg from mnpusim_params; return mNPUsim-relative path."""
        rel_dir = os.path.join("dram_config", "total_dram_config")
        cfg_path = os.path.join(self.eonsim_config_dir, rel_dir, "generated.cfg")

        ini_rel = os.path.join(self.eonsim_dir_name, "dram_config", "single_dram_config",
                               self.mnpusim_params.get("dram_config", ""))
        p = self.mnpusim_params

        lines = [
            f"dramconfig_name     {ini_rel}",
            f"spm_latency         {p.get('spm_latency', 1)}",
            f"pagebits            {p.get('pagebits', 12)}",
            f"npu_num             {p.get('npu_num', 1)}",
            f"dramoutdir_name     {p.get('dramoutdir_name', 'dramsim_output')}",
            f"dram_unit           {p.get('dram_unit', 128)}",
            f"dram_log            {p.get('dram_log', 0)}",
            f"dram_capacity       {p.get('dram_capacity_per_module', 0)}",
            f"module_num          {p.get('module_num', 1)}",
        ]
        with open(cfg_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        if self.debug:
            print(f"[DEBUG] Generated DRAM config: {cfg_path}")
        return os.path.join(self.eonsim_dir_name, rel_dir, "generated.cfg")

    def _generate_npumem_config(self):
        """Write npumem_architecture/generated.cfg and list .txt; return list-file mNPUsim-relative path."""
        arch_rel_dir = os.path.join("npumem_config", "npumem_architecture")
        list_rel_dir = os.path.join("npumem_config", "npumem_architecture_list")
        arch_cfg_path = os.path.join(self.eonsim_config_dir, arch_rel_dir, "generated.cfg")
        list_txt_path = os.path.join(self.eonsim_config_dir, list_rel_dir, "generated.txt")

        p = self.mnpusim_params
        arch_lines = [
            f"template            {p.get('template', 'arch_tpu_small.csv')}",
            f"spm_size            {p.get('spm_size', 0)}",
            f"cacheline_size      {p.get('cacheline_size', 128)}",
            f"tlb_hit_latency     {p.get('tlb_hit_latency', 0)}",
            f"tlb_miss_latency    {p.get('tlb_miss_latency', 0)}",
            f"tlb_assoc           {p.get('tlb_assoc', 8)}",
            f"tlb_entrynum        {p.get('tlb_entrynum', 16)}",
            f"tlb_portnum         {p.get('tlb_portnum', 0)}",
            f"spm_latency         {p.get('spm_latency', 1)}",
            f"double_buffer       {p.get('double_buffer', 1)}",
            f"npu_clockspeed      {p.get('npu_clockspeed', 1)}",
            f"dram_clockspeed     {p.get('dram_clockspeed', 1)}",
            f"tlb_pref_mode       {p.get('tlb_pref_mode', 0)}",
            f"ptw_num             {p.get('ptw_num', 8)}",
            f"pt_step_num         {p.get('pt_step_num', 1)}",
        ]
        with open(arch_cfg_path, "w") as f:
            f.write("\n".join(arch_lines) + "\n")

        # List file points at the generated architecture cfg via the mNPUsim-relative path.
        arch_relpath = os.path.join(self.eonsim_dir_name, arch_rel_dir, "generated.cfg")
        with open(list_txt_path, "w") as f:
            f.write(arch_relpath + "\n")

        if self.debug:
            print(f"[DEBUG] Generated npumem arch cfg: {arch_cfg_path}")
            print(f"[DEBUG] Generated npumem list txt: {list_txt_path}")
        return os.path.join(self.eonsim_dir_name, list_rel_dir, "generated.txt")
        
    def execute_mnpusim(self, trace_file_path):
        """Execute mNPUsim with proper environment setup"""
        # Runtime config paths (already relative to mNPUsim root) come from the cfg generators.
        dram_config_path = self.generated_dram_config_relpath
        npumem_config_path = self.generated_npumem_list_relpath

        if self.debug: print(f"[DEBUG] Using DRAM config: {dram_config_path}")
        if self.debug: print(f"[DEBUG] Using NPU memory config: {npumem_config_path}")

        full_dram_config_path = os.path.join(self.mnpusim_path, dram_config_path)
        full_npumem_config_path = os.path.join(self.mnpusim_path, npumem_config_path)
        
        if not os.path.exists(full_dram_config_path):
            print(f"[WARNING] DRAM config file not found: {full_dram_config_path}")
        else:
            if self.debug: print(f"[DEBUG] DRAM config file verified: {full_dram_config_path}")
            
        if not os.path.exists(full_npumem_config_path):
            print(f"[WARNING] NPU memory config file not found: {full_npumem_config_path}")
        else:
            if self.debug: print(f"[DEBUG] NPU memory config file verified: {full_npumem_config_path}")
        
        mnpusim_cmd = [
            self.mnpusim_path + "/mnpusim",
            "arch_config/core_architecture_list/tpu.txt",
            "network_config/netconfig_list/single/test1_network.txt",
            dram_config_path,
            npumem_config_path,
            self.eonsim_results_dir_name,  # Use the dynamic directory name
            "misc_config/single.cfg",
            trace_file_path
        ]
        
        # Set up environment variables for mnpusim execution
        env = os.environ.copy()
        dramsim3_path = self.mnpusim_path + "/DRAMsim3"
        if 'LD_LIBRARY_PATH' in env:
            env['LD_LIBRARY_PATH'] = f"{dramsim3_path}:{env['LD_LIBRARY_PATH']}"
        else:
            env['LD_LIBRARY_PATH'] = dramsim3_path
        
        try:
            if self.debug: print("[DEBUG] Executing mnpusim...")
            if self.debug: print(f"[DEBUG] LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
            if self.debug: print(f"[DEBUG] mnpusim command: {' '.join(mnpusim_cmd)}")
            result = subprocess.run(mnpusim_cmd, capture_output=True, text=True, check=True, env=env, cwd=self.mnpusim_path)
            
            # Save output to intermediate directory
            output_path = os.path.join(self.intermediate_dir, "output.txt")
            with open(output_path, "w") as f:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            if self.debug: print(f"[DEBUG] mnpusim execution completed. Output saved to {output_path}")
            self.execution_successful = True
            
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] mnpusim execution failed with return code {e.returncode}")
            error_output_path = os.path.join(self.intermediate_dir, "output.txt")
            with open(error_output_path, "w") as f:
                f.write(f"ERROR: mnpusim execution failed with return code {e.returncode}\n")
                f.write("STDOUT:\n")
                f.write(e.stdout)
                f.write("\nSTDERR:\n")
                f.write(e.stderr)
            self.execution_successful = False
            
        except FileNotFoundError:
            print("[ERROR] mnpusim binary not found")
            error_output_path = os.path.join(self.intermediate_dir, "output.txt")
            with open(error_output_path, "w") as f:
                f.write("ERROR: mnpusim binary not found\n")
            self.execution_successful = False
            
    def extract_results(self):
        """Copy results and extract execution cycles"""
        if not self.execution_successful:
            print("[WARNING] mNPUsim execution was not successful, skipping result extraction")
            return
            
        # Copy eonsim_results/result directory to intermediate directory
        result_source_dir = os.path.join(self.eonsim_results_dir, "result")
        result_dest_dir = os.path.join(self.intermediate_dir, "result")
        
        if os.path.exists(result_source_dir):
            shutil.copytree(result_source_dir, result_dest_dir)
            if self.debug: print(f"[DEBUG] Copied result directory to {result_dest_dir}")
            
            # Find and read execution_cycle file
            for filename in os.listdir(result_dest_dir):
                if filename.startswith("execution_cycle"):
                    execution_cycle_file = os.path.join(result_dest_dir, filename)
                    try:
                        with open(execution_cycle_file, 'r') as f:
                            self.offmem_cycles_raw = int(f.readline().strip())
                            if self.debug: print(f"[DEBUG] mNPUsim cycles: {self.offmem_cycles_raw}")
                            break
                    except (ValueError, IOError) as e:
                        print(f"[ERROR] Failed to read execution cycles from {filename}: {e}")
            else:
                print("[WARNING] No execution_cycle file found in result directory")
        else:
            print(f"[WARNING] Result directory not found: {result_source_dir}")

        # Keep legacy loop-overhead correction on raw mNPUsim cycles.
        # self.offmem_cycles_raw += (len(self.offmem_trace_last_batch) // 1024) * 41
        self._apply_analytical_model()
    
    def cleanup_intermediate_directory(self):
        """Remove the intermediate directory after simulation"""
        if os.path.exists(self.intermediate_dir):
            shutil.rmtree(self.intermediate_dir)
            if self.debug: print(f"[DEBUG] Cleaned up intermediate directory: {self.intermediate_dir}")
            
    def cleanup_eonsim_results_directory(self):
        """Remove the eonsim_results directory after simulation"""
        if os.path.exists(self.eonsim_results_dir):
            shutil.rmtree(self.eonsim_results_dir)
            if self.debug: print(f"[DEBUG] Cleaned up eonsim_results directory: {self.eonsim_results_dir}")

    def cleanup_eonsim_config_directory(self):
        """Remove the per-run eonsim_config directory after simulation."""
        if self.eonsim_config_dir and os.path.exists(self.eonsim_config_dir):
            shutil.rmtree(self.eonsim_config_dir)
            if self.debug: print(f"[DEBUG] Cleaned up eonsim_config directory: {self.eonsim_config_dir}")
            
    def do_memory_simulation(self):
        """Main method to run the complete memory simulation pipeline"""
        if self.debug: print("[DEBUG] Starting off-chip memory simulation...")
        if self.debug: print(f"[DEBUG] Requested DRAM .ini: {self.mnpusim_params.get('dram_config', '')}")

        try:
            # Setup and run simulation pipeline.
            self.setup_intermediate_directory()
            self.setup_eonsim_results_directory()  # Setup results directory with random suffix
            self.setup_eonsim_config_directory()
            trace_file_path = self.generate_trace_file()

            # Execute mNPUsim
            self.execute_mnpusim(trace_file_path)

            # Extract results and apply analytical model.
            self.extract_results()
        finally:
            # Always cleanup temporary artifacts even when an exception occurs.
            self.cleanup_intermediate_directory()
            self.cleanup_eonsim_results_directory()  # Cleanup results directory
            self.cleanup_eonsim_config_directory()  # Cleanup per-run config directory
        
    def print_stats(self, batch_idx=None):
        """Print memory simulation results for one batch."""
        content_lines = []

        label = f"[Batch {batch_idx}] " if batch_idx is not None else ""

        if self.execution_successful:
            content_lines.append(f"{label}Memory Cycles: {self.offmem_cycles}")
            if self.debug and self.analytical_results:
                content_lines.append(f"Off-chip Issue Cycles: {self.analytical_results['offchip_issue_cycles']}")
                content_lines.append(f"Off-chip Transfer Cycles: {self.offmem_cycles_raw}")
                content_lines.append(f"Off-chip Total Cycles: {self.analytical_results['offchip_total_cycles']}")
                if self.onchip_structure == "two_level":
                    content_lines.append(f"Global-to-Local Issue Cycles: {self.analytical_results['global_to_local_issue_cycles']}")
                    content_lines.append(f"Global-to-Local Transfer Cycles: {self.analytical_results['global_to_local_transfer_cycles']}")
                    content_lines.append(f"Global-to-Local Total Cycles: {self.analytical_results['global_to_local_total_cycles']}")
        else:
            content_lines.append(f"{label}Memory Cycles: N/A")

        print_styled_box("Memory Simulation Results", content_lines)

    @staticmethod
    def print_aggregate_stats(batch_model_pairs):
        """Print average then per-batch memory cycle summary.

        Args:
            batch_model_pairs: list of (batch_idx, MemoryModel) for non-warmup batches.
        """
        valid = [(nb, model.offmem_cycles) for nb, model in batch_model_pairs if model.execution_successful]
        if not valid:
            print("[WARNING] No successful memory simulation results to aggregate.")
            return

        avg_cycles = sum(c for _, c in valid) / len(valid)
        content = [
            f"Average Memory Cycles: {avg_cycles:.1f}",
            "----------------------------------------",
        ]
        for nb, cycles in valid:
            content.append(f"[Batch {nb}] Memory Cycles: {cycles}")
        print_styled_box("Memory Simulation Summary", content)
