import yaml
import numpy as np
from Helper import print_styled_header, print_styled_box

class DataTypeEnergy:
    def __init__(self, config_dict):
        self.add = float(config_dict['add'])
        self.mul = float(config_dict['mul'])
        self.cmp = float(config_dict['cmp'])

class EnergyTable:
    def __init__(self, config_dict):
        self.offchip = float(config_dict['offchip'])
        self.global_buffer = float(config_dict['global_buffer'])
        self.local_buffer = float(config_dict['local_buffer'])
        self.int8 = DataTypeEnergy(config_dict['int8'])
        self.fp32 = DataTypeEnergy(config_dict['fp32'])

class OpConfig:
    def __init__(self, config_dict):
        self.op_type = str(config_dict['op_type'])
        self.access_per_op = int(config_dict['access_per_op'])
        self.num_op = int(config_dict['num_op'])

class EnergyEstimator:
    def __init__(self, workload_type, workload_config_path, tech_node, energy_table_path, energy_n_format, access_results, access_per_batch, mem_gran):
        
        print("\n\n\n START ENERGY ESTIMATION \n")
        
        self.workload_config = None
        self.workload_config_path = None
        self.tech_node = 0
        self.energy_table_path = None
        self.energy_n_format = None
        self.access_results = None
        self.access_per_batch = 0
        self.mem_gran = 0
        
        self.set_params(workload_type, workload_config_path, tech_node, energy_table_path, energy_n_format, access_results, access_per_batch, mem_gran)
        self.set_workload_config()
        self.set_energy_table()

    def set_params(self, workload_type, workload_config_path, tech_node, energy_table_path, energy_n_format, access_results, access_per_batch, mem_gran):
        print(f"Setting parameters for energy estimation...")
        self.workload_type = workload_type
        self.workload_config_path = workload_config_path
        self.tech_node = tech_node
        self.energy_table_path = energy_table_path
        self.energy_n_format = energy_n_format
        self.access_results = access_results
        self.access_per_batch = access_per_batch
        self.mem_gran = mem_gran
    
    def set_workload_config(self):
        print(f"Setting workload configuration...")
        with open(self.workload_config_path, 'r') as file:
            self.workload_config = yaml.safe_load(file)
            
        # Get workload specific configuration
        workload_spec = self.workload_config[self.workload_type]
        self.num_op_types = int(workload_spec['num_op_types'])
        
        # Parse operation configurations
        for i in range(self.num_op_types):
            op_key = f'op{i}'
            op_config = workload_spec[op_key]
            setattr(self, op_key, OpConfig(op_config))

    def set_energy_table(self):
        print(f"Setting energy table configuration...")
        with open(self.energy_table_path, 'r') as file:
            energy_config = yaml.safe_load(file)
            
        # Get technology node specific energy values
        tech_key = f'energy_{self.tech_node}nm'
        energy_spec = energy_config[tech_key]
        
        # Create energy table object
        self.ET = EnergyTable(energy_spec)
        
    def do_energy_estimation(self):
        # Set the action count for each operation / per batch
        self.energy_results = []
        for nb in range(len(self.access_results)):
            # Global buffer energy
            num_global_buffer_access = self.access_results[nb][0] # TODO: add local buffer access
            global_buffer_energy = num_global_buffer_access * self.ET.global_buffer * self.mem_gran # pJ
            
            # Off-chip memory access energy
            num_offchip_access = self.access_results[nb][1]
            offchip_energy = num_offchip_access * self.ET.offchip * self.mem_gran # pJ            
            
            # For each operation type
            this_batch_ops_energy = []
            for i in range(self.num_op_types):
                op_attr = getattr(self, f'op{i}')
                this_action_type = op_attr.op_type
                this_action_count = np.ceil(self.access_per_batch / op_attr.access_per_op) * op_attr.num_op
                
                # print(" [DEBUG] action type and count for op {}: {} {}".format(i, this_action_type, this_action_count))
                    
                if this_action_type in ['vadd', 'vmul']:
                    this_action_count = this_action_count * self.mem_gran
                    if this_action_type == 'vadd':
                        this_action_type = 'add'
                    elif this_action_type == 'vmul':
                        this_action_type = 'mul'
                    
                    print(" [DEBUG] action type and count for op {}: {} {}".format(i, this_action_type, this_action_count))
                
                if self.energy_n_format == 'fp32':
                    this_action_count = np.ceil(this_action_count / 4).astype(int)
                    
                # print(" [DEBUG] action type and count for op {}: {} {}".format(i, this_action_type, this_action_count))
                    
                energy_object = getattr(self.ET, self.energy_n_format)
                this_action_energy = this_action_count * getattr(energy_object, this_action_type)
                this_batch_ops_energy.append(this_action_energy)
                
            this_batch_total_energy = np.sum(this_batch_ops_energy) + global_buffer_energy + offchip_energy
            self.energy_results.append([this_batch_total_energy, global_buffer_energy, offchip_energy, this_batch_ops_energy])
        
        self.print_stats()

    def print_stats(self):
        print_styled_header("Energy Estimation Results")
        
        # Calculate total energy across all batches
        total_batch_energy = 0
        total_global_buffer_energy = 0
        total_offchip_energy = 0
        total_ops_energy = np.zeros(self.num_op_types)
        
        for batch_result in self.energy_results:
            total_batch_energy += batch_result[0]
            total_global_buffer_energy += batch_result[1]
            total_offchip_energy += batch_result[2]
            total_ops_energy += np.array(batch_result[3])
        
        # Prepare content as a list of strings
        content_lines = []
        
        # Total energies
        content_lines.append(f"Total Energy: {total_batch_energy:.3f} pJ")
        content_lines.append(f"Off-chip Memory Energy: {total_offchip_energy:.3f} pJ ({total_offchip_energy/total_batch_energy*100:.2f}%)")
        content_lines.append(f"On-chip Memory Energy: {total_global_buffer_energy:.3f} pJ ({total_global_buffer_energy/total_batch_energy*100:.2f}%)")
        content_lines.append("Operations Energy:")
        for i in range(self.num_op_types):
            content_lines.append(f"  - Operation {i}: {total_ops_energy[i]:.3f} pJ ({total_ops_energy[i]/total_batch_energy*100:.2f}%)")
        
        # Per-batch energy details
        content_lines.append("")  # Empty line for spacing
        content_lines.append("Per-Batch Energy:")
        for i, batch_result in enumerate(self.energy_results):
            content_lines.append(f"  - Batch {i}:")
            content_lines.append(f"    Total Energy: {batch_result[0]:.3f} pJ")
            content_lines.append(f"    Off-chip Memory Energy: {batch_result[2]:.3f} pJ")
            content_lines.append(f"    On-chip Memory Energy: {batch_result[1]:.3f} pJ")
            content_lines.append("    Operations Energy:")
            for j, op_energy in enumerate(batch_result[3]):
                content_lines.append(f"      - Operation {j}: {op_energy:.3f} pJ")
        
        print_styled_box("Energy Estimation Results", content_lines)

    def print_all_config(self):
        print("\n============= Configuration Summary =============")
        print("\n[Workload Configuration]")
        print(f"Workload Type: {self.workload_type}")
        print(f"Number of Operation Types: {self.num_op_types}")
        
        print("\n[Operation Details]")
        for i in range(self.num_op_types):
            op = getattr(self, f'op{i}')
            print(f"- Operation {i}:")
            print(f"  Type: {op.op_type}")
            print(f"  Access per operation: {op.access_per_op}")
            print(f"  Number of operations: {op.num_op}")
        
        print("\n[Energy Table Configuration]")
        print(f"Technology Node: {self.tech_node}nm")
        print("\nMemory Access Energy (pJ/byte):")
        print(f"- Off-chip Memory: {self.ET.offchip:.3f}")
        print(f"- Global Buffer: {self.ET.global_buffer:.3f}")
        print(f"- Local Buffer: {self.ET.local_buffer:.3f}")
        
        print("\nComputation Energy (pJ):")
        print("INT8 Operations:")
        print(f"- Addition: {self.ET.int8.add:.3f}")
        print(f"- Multiplication: {self.ET.int8.mul:.3f}")
        print(f"- Compare: {self.ET.int8.cmp:.3f}")
        
        print("\nFP32 Operations:")
        print(f"- Addition: {self.ET.fp32.add:.3f}")
        print(f"- Multiplication: {self.ET.fp32.mul:.3f}")
        print(f"- Compare: {self.ET.fp32.cmp:.3f}")
        print("=============================================\n")