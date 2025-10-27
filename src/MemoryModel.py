import os
import subprocess
import shutil
import random
from Helper import print_styled_box

class MemoryModel:
    def __init__(self, script_dir, offmem_trace, mnpusim_path, mnpusim_config_path, offchip_memory_config, npumem_config):
        print("\n\n\n START OFF-CHIP MEMORY SIMULATION \n")
        
        self.script_dir = script_dir
        self.offmem_trace = offmem_trace
        self.intermediate_dir = os.path.join(os.path.dirname(script_dir), 'intermediate')
        self.mnpusim_path = mnpusim_path
        self.mnpusim_config_path = mnpusim_config_path
        self.offchip_memory_config = offchip_memory_config
        self.npumem_config = npumem_config
        self.eonsim_results_dir = None  # Will be set in setup_eonsim_results_directory()
        self.eonsim_results_dir_name = None  # Will store the directory name for mnpusim command
        self.eonsim_dir_name = "eonsim_config"
        self.eonsim_config_dir = os.path.join(self.mnpusim_path, self.eonsim_dir_name)

        # Results
        self.offmem_cycles = 0
        self.execution_successful = False
        
    def setup_intermediate_directory(self):
        """Create intermediate directory with unique random number suffix"""
        base_dir = os.path.dirname(self.intermediate_dir)
        
        # Find an unused random number for directory name
        while True:
            random_num = random.randint(1, 999999)
            candidate_dir = os.path.join(base_dir, f'intermediate_{random_num}')
            if not os.path.exists(candidate_dir):
                self.intermediate_dir = candidate_dir
                break
        
        os.makedirs(self.intermediate_dir)
        print(f"[DEBUG] Created intermediate directory: {self.intermediate_dir}")
        
    def setup_eonsim_results_directory(self):
        """Create eonsim_results directory with unique random number suffix"""
        # Find an unused random number for directory name
        while True:
            random_num = random.randint(1, 999999)
            candidate_dir_name = f'eonsim_results_{random_num}'
            candidate_dir = os.path.join(self.mnpusim_path, candidate_dir_name)
            if not os.path.exists(candidate_dir):
                self.eonsim_results_dir = candidate_dir
                self.eonsim_results_dir_name = candidate_dir_name
                break
        
        print(f"[DEBUG] Created eonsim_results directory name: {self.eonsim_results_dir_name}")
        
    def generate_trace_file(self):
        """Generate flattened trace file for mNPUsim"""
        # 25.10.09: Only use the last batch for now (offmem_trace[-1])
        self.offmem_trace = self.offmem_trace[-1]
        # Flatten the offmem_trace 3D array to 1D array
        # flat_offmem_trace = [addr for sublist in self.offmem_trace for tbl in sublist for addr in tbl]
        flat_offmem_trace = [addr for sublist in self.offmem_trace for addr in sublist]
        
        # Save flattened trace
        offmem_trace_path = os.path.join(self.intermediate_dir, "offmem_trace_flat.txt")
        with open(offmem_trace_path, "w") as f:
            f.write("0,")  # Initial dummy value for mNPUsim
            for addr in flat_offmem_trace:                
                if not addr == -1:  # Skip -1 entries
                    f.write(str(addr) + ",")            
        
        print(f"[DEBUG] Generated trace file: {offmem_trace_path}")
        
        # Print trace statistics
        total_elements = len(flat_offmem_trace)
        minus_one_count = sum(1 for addr in flat_offmem_trace if addr == -1)
        
        print(f"[DEBUG] Total elements in offmem_trace: {total_elements}")
        print(f"[DEBUG] -1 count in offmem_trace: {minus_one_count}")
        print(f"[DEBUG] -1 ratio in offmem_trace: {minus_one_count/total_elements:.4f}")
        
        return offmem_trace_path
        
    def cleanup_mnpusim_results(self):
        """Remove existing mNPUsim results directory"""
        if os.path.exists(self.eonsim_results_dir):
            shutil.rmtree(self.eonsim_results_dir)
            print(f"[DEBUG] Removed existing eonsim_results directory")
            
    def setup_eonsim_config_directory(self):
        """Create and setup eonsim_config directory with required config files"""
        # Create eonsim_config directory if it doesn't exist
        if not os.path.exists(self.eonsim_config_dir):
            os.makedirs(self.eonsim_config_dir)
            print(f"[DEBUG] Created eonsim_config directory: {self.eonsim_config_dir}")
        
        # Remove existing dram_config and npumem_config if they exist
        dram_config_dest = os.path.join(self.eonsim_config_dir, "dram_config")
        npumem_config_dest = os.path.join(self.eonsim_config_dir, "npumem_config")
        
        if os.path.exists(dram_config_dest):
            shutil.rmtree(dram_config_dest)
            print(f"[DEBUG] Removed existing dram_config directory")
            
        if os.path.exists(npumem_config_dest):
            shutil.rmtree(npumem_config_dest)
            print(f"[DEBUG] Removed existing npumem_config directory")
        
        # Copy dram_config and npumem_config from mnpusim_config_path
        dram_config_src = os.path.join(self.mnpusim_config_path, "dram_config")
        npumem_config_src = os.path.join(self.mnpusim_config_path, "npumem_config")
        
        if os.path.exists(dram_config_src):
            shutil.copytree(dram_config_src, dram_config_dest)
            print(f"[DEBUG] Copied dram_config from {dram_config_src} to {dram_config_dest}")
        else:
            print(f"[WARNING] dram_config source directory not found: {dram_config_src}")
            
        if os.path.exists(npumem_config_src):
            shutil.copytree(npumem_config_src, npumem_config_dest)
            print(f"[DEBUG] Copied npumem_config from {npumem_config_src} to {npumem_config_dest}")
        else:
            print(f"[WARNING] npumem_config source directory not found: {npumem_config_src}")
        
    def execute_mnpusim(self, trace_file_path):
        """Execute mNPUsim with proper environment setup"""
        # Construct the full paths for config files
        dram_config_path = os.path.join(self.eonsim_dir_name, self.offchip_memory_config)
        npumem_config_path = os.path.join(self.eonsim_dir_name, self.npumem_config)
        
        print(f"[DEBUG] DRAM config path: {dram_config_path}")
        print(f"[DEBUG] NPU memory config path: {npumem_config_path}")
        
        # Verify that the config files exist in the mNPUsim directory
        full_dram_config_path = os.path.join(self.mnpusim_path, dram_config_path)
        full_npumem_config_path = os.path.join(self.mnpusim_path, npumem_config_path)
        
        if not os.path.exists(full_dram_config_path):
            print(f"[WARNING] DRAM config file not found: {full_dram_config_path}")
        else:
            print(f"[DEBUG] DRAM config file verified: {full_dram_config_path}")
            
        if not os.path.exists(full_npumem_config_path):
            print(f"[WARNING] NPU memory config file not found: {full_npumem_config_path}")
        else:
            print(f"[DEBUG] NPU memory config file verified: {full_npumem_config_path}")
        
        mnpusim_cmd = [
            self.mnpusim_path + "/mnpusim",
            "arch_config/core_architecture_list/tpu.txt",
            "network_config/netconfig_list/single/test1_network.txt",
            "eonsim_config/dram_config/total_dram_config/single_hbm3_819gbs.cfg",
            "eonsim_config/npumem_config/npumem_architecture_list/single.txt",
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
            print("[DEBUG] Executing mnpusim...")
            print(f"[DEBUG] LD_LIBRARY_PATH: {env['LD_LIBRARY_PATH']}")
            result = subprocess.run(mnpusim_cmd, capture_output=True, text=True, check=True, env=env, cwd=self.mnpusim_path)
            
            # Save output to intermediate directory
            output_path = os.path.join(self.intermediate_dir, "output.txt")
            with open(output_path, "w") as f:
                f.write("STDOUT:\n")
                f.write(result.stdout)
                f.write("\nSTDERR:\n")
                f.write(result.stderr)
            
            print(f"[DEBUG] mnpusim execution completed. Output saved to {output_path}")
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
            print(f"[DEBUG] Copied result directory to {result_dest_dir}")
            
            # Find and read execution_cycle file
            for filename in os.listdir(result_dest_dir):
                if filename.startswith("execution_cycle"):
                    execution_cycle_file = os.path.join(result_dest_dir, filename)
                    try:
                        with open(execution_cycle_file, 'r') as f:
                            self.offmem_cycles = int(f.readline().strip())
                            print(f"[RESULT] mNPUsim cycles: {self.offmem_cycles}")
                            break
                    except (ValueError, IOError) as e:
                        print(f"[ERROR] Failed to read execution cycles from {filename}: {e}")
            else:
                print("[WARNING] No execution_cycle file found in result directory")
        else:
            print(f"[WARNING] Result directory not found: {result_source_dir}")
            
        # get the number of "-1" in offmem_trace for "onmem_elems"
        onmem_elems = sum(1 for sublist in self.offmem_trace for addr in sublist if addr == -1)
        # Add onmem_elems to offmem_cycles
        # self.offmem_cycles += onmem_elems
        # Add len(self.offmem_trace)/1024 to offmem_cycles (for loop overhead)
        self.offmem_cycles += (len(self.offmem_trace) // 1024) * 20  # Assuming 20 cycles latency
    
    def cleanup_intermediate_directory(self):
        """Remove the intermediate directory after simulation"""
        if os.path.exists(self.intermediate_dir):
            shutil.rmtree(self.intermediate_dir)
            print(f"[DEBUG] Cleaned up intermediate directory: {self.intermediate_dir}")
            
    def cleanup_eonsim_results_directory(self):
        """Remove the eonsim_results directory after simulation"""
        if os.path.exists(self.eonsim_results_dir):
            shutil.rmtree(self.eonsim_results_dir)
            print(f"[DEBUG] Cleaned up eonsim_results directory: {self.eonsim_results_dir}")
            
    def do_memory_simulation(self):
        """Main method to run the complete memory simulation pipeline"""
        print("[DEBUG] Starting off-chip memory simulation...")
        
        # Setup and cleanup
        self.setup_intermediate_directory()
        self.setup_eonsim_results_directory()  # Setup results directory with random suffix
        self.setup_eonsim_config_directory()
        trace_file_path = self.generate_trace_file()
        self.cleanup_mnpusim_results()
        
        # Execute mNPUsim
        self.execute_mnpusim(trace_file_path)
        
        # Extract results
        self.extract_results()
        
        self.print_stats()
        
        # Cleanup directories after simulation
        self.cleanup_intermediate_directory()
        self.cleanup_eonsim_results_directory()  # Cleanup results directory
        
    def print_stats(self):
        """Print memory simulation results"""
        content_lines = []
        
        if self.execution_successful:
            content_lines.append(f"Off-chip Memory Cycles: {self.offmem_cycles}")
            # content_lines.append(f"Simulation Status: Successful")
        else:
            # content_lines.append("Simulation Status: Failed")
            content_lines.append("Off-chip Memory Cycles: N/A")
            
        # content_lines.append(f"Results Directory: {self.intermediate_dir}")
        
        print_styled_box("Off-chip Memory Simulation Results", content_lines)
