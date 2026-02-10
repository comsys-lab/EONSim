import yaml
import os

class ConfigLoader:
    def __init__(self, workload_config_base_path):
        # workload_config_base_path: path without extension (e.g., .../dlrm_rmc2_small)
        self.base_path = workload_config_base_path
        self.yaml_path = f"{self.base_path}.yaml"
        self.csv_path = f"{self.base_path}.csv"
        
        self.config = self._load_yaml()

    def _load_yaml(self):
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"Workload YAML config not found: {self.yaml_path}")
        with open(self.yaml_path, 'r') as f:
            return yaml.safe_load(f)
    
    @staticmethod
    def load_memory_config(memory_config_path):
        """
        Load memory configuration from YAML file.
        
        Args:
            memory_config_path: Full path to memory config YAML file
            
        Returns:
            Dictionary containing parsed memory configuration including:
            - core_dim: {row, col}
            - local_buffer: {mem_size, mem_type, mem_policy, mem_gran, cache_way, cache_line_size, rrpv_bits, rrip_insert}
            - global_buffer: {mem_size, mem_type, mem_policy, mem_gran, cache_way, cache_line_size, rrpv_bits, rrip_insert}
            - vector_unit: {lanes, sublanes, alus_per_sublanes}
            - matrix_unit: {mxu_dimension, num_mxus}
            - cache_config: list for local buffer
            - global_cache_config: list for global buffer
        """
        if not os.path.exists(memory_config_path):
            raise FileNotFoundError(f"Memory config file not found: {memory_config_path}")
        
        with open(memory_config_path, 'r') as yaml_cfg:
            config_data = yaml.safe_load(yaml_cfg)
        
        result = {}
        
        # Parse core dimension configuration
        core_dim_config = config_data.get('core_dim', {})
        result['core_dim'] = {
            'row': core_dim_config.get('row', 2),
            'col': core_dim_config.get('col', 2)
        }
        
        # Parse local buffer configuration (main memory for simulation)
        local_buffer_config = config_data.get('local_buffer', {})
        if not local_buffer_config:
            # Fallback to old 'memory' key for backward compatibility
            local_buffer_config = config_data.get('memory', {})
        
        mem_size = local_buffer_config.get('mem_size', 0)  # KB
        mem_type = local_buffer_config.get('mem_type', '')
        policy = local_buffer_config.get('policy', '')
        mem_policy = mem_type + '_' + policy if policy else mem_type
        mem_gran = local_buffer_config.get('access_granularity', 0)  # B
        mem_latency = local_buffer_config.get('access_latency', 1)  # cycles
        
        # Parse cache-specific parameters for local buffer
        cache_way = 0
        cache_line_size = 0
        rrpv_bits = 0
        rrip_insert = 0
        
        if mem_type == "cache":
            cache_way = local_buffer_config.get('cache_way', 0)
            cache_line_size = mem_gran  # Use access_granularity as cache_line_size
        
        if mem_policy in ['cache_SRRIP', 'profile_dynamic_SRRIP']:
            rrpv_bits = local_buffer_config.get('RRPV_bits', 0)
            rrip_insert = local_buffer_config.get('RRPV_insertion', 0)
        
        # Temporary fix for profile_dynamic_SRRIP
        if mem_policy == 'profile_dynamic_SRRIP':
            rrpv_bits = 4
            rrip_insert = 14
        
        result['local_buffer'] = {
            'mem_size': mem_size,
            'mem_type': mem_type,
            'mem_policy': mem_policy,
            'mem_gran': mem_gran,
            'mem_latency': mem_latency,
            'cache_way': cache_way,
            'cache_line_size': cache_line_size,
            'rrpv_bits': rrpv_bits,
            'rrip_insert': rrip_insert
        }
        result['cache_config'] = [cache_way, cache_line_size, rrpv_bits, rrip_insert]
        
        # Parse global buffer configuration
        global_buffer_config = config_data.get('global_buffer', {})
        global_mem_size = 0
        global_mem_type = None
        global_mem_policy = None
        global_mem_gran = 0
        global_mem_latency = 15  # Default latency for global buffer
        global_cache_way = 0
        global_cache_line_size = 0
        global_rrpv_bits = 0
        global_rrip_insert = 0
        
        if global_buffer_config:
            global_mem_size = global_buffer_config.get('mem_size', 0)  # KB
            global_mem_type = global_buffer_config.get('mem_type', '')
            global_policy = global_buffer_config.get('policy', '')
            global_mem_policy = global_mem_type + '_' + global_policy if global_policy else global_mem_type
            global_mem_gran = global_buffer_config.get('access_granularity', 0)  # B
            global_mem_latency = global_buffer_config.get('access_latency', 15)  # cycles
            
            # Parse cache-specific parameters for global buffer
            if global_mem_type == "cache":
                global_cache_way = global_buffer_config.get('cache_way', 0)
                global_cache_line_size = global_mem_gran
            
            if global_mem_policy in ['cache_SRRIP', 'profile_dynamic_SRRIP']:
                global_rrpv_bits = global_buffer_config.get('RRPV_bits', 0)
                global_rrip_insert = global_buffer_config.get('RRPV_insertion', 0)
        
        result['global_buffer'] = {
            'mem_size': global_mem_size,
            'mem_type': global_mem_type,
            'mem_policy': global_mem_policy,
            'mem_gran': global_mem_gran,
            'mem_latency': global_mem_latency,
            'cache_way': global_cache_way,
            'cache_line_size': global_cache_line_size,
            'rrpv_bits': global_rrpv_bits,
            'rrip_insert': global_rrip_insert
        }
        result['global_cache_config'] = [global_cache_way, global_cache_line_size, global_rrpv_bits, global_rrip_insert]
        
        # Parse vector unit configuration
        vector_unit_config = config_data.get('vector_unit', {})
        result['vector_unit'] = {
            'lanes': vector_unit_config.get('lanes', 128),
            'sublanes': vector_unit_config.get('sublanes', 8),
            'alus_per_sublanes': vector_unit_config.get('ALUs_per_sublanes', 4)
        }
        
        # Parse matrix unit configuration
        matrix_unit_config = config_data.get('matrix_unit', {})
        result['matrix_unit'] = {
            'mxu_dimension': matrix_unit_config.get('mxu_dimension', 128),
            'num_mxus': matrix_unit_config.get('num_mxus', 4)
        }
        
        return result

    def get_embedding_config(self):
        # Extract embedding layer config
        emb_config = self.config.get('embedding_layer', {})
        embedding_dim = emb_config.get('embedding_dim', 128)
        vectors_per_table = emb_config.get('vectors_per_table', 500000)
        num_tables = emb_config.get('num_tables', 1)
        pooling_factor = emb_config.get('pooling_factor', 1)
        
        # Generate embedding size string
        emb_size_str = "-".join([str(vectors_per_table)] * num_tables)
        
        return {
            'embedding_dim': embedding_dim,
            'vectors_per_table': vectors_per_table,
            'num_tables': num_tables,
            'pooling_factor': pooling_factor,
            'emb_size_str': emb_size_str
        }

    def get_matrix_ops_config_path(self):
        # Return the path to the CSV file for Matrix Ops config
        if not os.path.exists(self.csv_path):
            print(f"[WARNING] Matrix Ops CSV config not found: {self.csv_path}")
            return None
        return self.csv_path

    def get_general_config(self):
        return {
            'workload_type': self.config.get('workload_type', 'dlrm'),
            'num_format': self.config.get('num_format', 32)
        }
