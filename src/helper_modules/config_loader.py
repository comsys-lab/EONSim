import yaml
import os


# mNPUsim fixed parameters that are not exposed through YAML.
# These rarely change across experiments and would just clutter the user-facing config.
_MNPUSIM_FIXED = {
    "template": "arch_tpu_small.csv",
    "pagebits": 12,
    "npu_num": 1,
    "dramoutdir_name": "dramsim_output",
    "dram_log": 0,
    "double_buffer": 1,
    "npu_clockspeed": 1,
    "dram_clockspeed": 1,
    "tlb_assoc": 8,
    "tlb_entrynum": 16,
    "tlb_portnum": 0,
    "tlb_pref_mode": 0,
    "ptw_num": 8,
    "pt_step_num": 1,
}

# Matrix-unit local buffer split (input:weight:output = 3:3:2, TPU-like).
_MATRIX_BUFFER_SPLIT = (3, 3, 2)


class ConfigLoader:
    def __init__(self, workload_config_base_path):
        self.base_path = workload_config_base_path
        self.yaml_path = f"{self.base_path}.yaml"
        self.csv_path = f"{self.base_path}.csv"

        self.config = self._load_yaml()

    def _load_yaml(self):
        if not os.path.exists(self.yaml_path):
            raise FileNotFoundError(f"Workload YAML config not found: {self.yaml_path}")
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Malformed workload config: expected YAML object at root in {self.yaml_path}")

        return data

    @staticmethod
    def _buffer_dict(buf_cfg):
        mem_type = buf_cfg.get('mem_type', '') or ''
        policy = buf_cfg.get('policy', '') or ''
        mem_policy = f"{mem_type}_{policy}" if policy else mem_type

        result = {
            'mem_size': buf_cfg.get('mem_size', 0),
            'mem_type': mem_type,
            'mem_policy': mem_policy,
            'mem_gran': buf_cfg.get('access_granularity', 0),
            'mem_latency': buf_cfg.get('access_latency', 1),
            'bandwidth': buf_cfg.get('bandwidth', 0),
            'cache_way': 0,
            'cache_line_size': buf_cfg.get('access_granularity', 0),
            'rrpv_bits': 0,
            'rrpv_insert': 0,
            'lfu_counter_bits': 8,
            'lfu_aging_interval': 0,
        }

        if mem_type == "cache":
            result['cache_way'] = buf_cfg.get('cache_way', 0)

        if mem_policy == 'cache_SRRIP':
            result['rrpv_bits'] = buf_cfg.get('RRPV_bits', 0)
            result['rrpv_insert'] = buf_cfg.get('RRPV_insertion', 0)
        elif mem_policy == 'profile_dynamic_SRRIP':
            result['rrpv_bits'] = buf_cfg.get('RRPV_bits', 4)
            result['rrpv_insert'] = buf_cfg.get('RRPV_insertion', 14)

        if mem_policy == 'cache_LFU':
            result['lfu_counter_bits'] = buf_cfg.get('lfu_counter_bits', 8)
            result['lfu_aging_interval'] = buf_cfg.get('lfu_aging_interval', 0)

        return result

    @staticmethod
    def _cache_config_view(buf_result):
        return {
            'way': buf_result['cache_way'],
            'line_size': buf_result['cache_line_size'],
            'rrpv_bits': buf_result['rrpv_bits'],
            'rrpv_insert': buf_result['rrpv_insert'],
            'lfu_counter_bits': buf_result['lfu_counter_bits'],
            'lfu_aging_interval': buf_result['lfu_aging_interval'],
        }

    @staticmethod
    def _build_scalesim_hw_config(accel, matrix_unit, bandwidth_gbps, latency, matrix_total_buf_kb):
        pool_bytes = int(matrix_total_buf_kb) * 1024
        split_total = sum(_MATRIX_BUFFER_SPLIT)
        input_share, weight_share, output_share = _MATRIX_BUFFER_SPLIT
        return {
            'pod_row': int(accel.get('core_dim_row', 1)),
            'pod_col': int(accel.get('core_dim_col', 1)),
            'freq': int(accel.get('clock_frequency', 0)),
            'bw': int(bandwidth_gbps),
            'latency': int(latency),
            'dataflow': matrix_unit.get('dataflow', 'WS'),
            'sa_row': int(matrix_unit.get('sa_row', 0)),
            'sa_col': int(matrix_unit.get('sa_col', 0)),
            'input_buf_size': pool_bytes * input_share // split_total,
            'weight_buf_size': pool_bytes * weight_share // split_total,
            'output_buf_size': pool_bytes * output_share // split_total,
            'global_buf_size': pool_bytes,
        }

    @staticmethod
    def _build_mnpusim_params(local_buf, offchip):
        params = dict(_MNPUSIM_FIXED)
        params.update({
            'spm_size': int(local_buf.get('mem_size', 0)) * 1024,
            'cacheline_size': int(local_buf.get('access_granularity', 0)),
            'spm_latency': int(local_buf.get('access_latency', 1)),
            'tlb_hit_latency': int(offchip.get('latency', 0)),
            'tlb_miss_latency': int(offchip.get('latency', 0)),
            'dram_unit': int(local_buf.get('access_granularity', 0)),
            'dram_capacity_per_module': int(offchip.get('dram_capacity_per_module', 0)),
            'module_num': int(offchip.get('module_num', 1)),
            'dram_config': offchip.get('dram_config', ''),
        })
        return params

    @staticmethod
    def load_memory_config(memory_config_path):
        if not os.path.exists(memory_config_path):
            raise FileNotFoundError(f"Memory config file not found: {memory_config_path}")

        with open(memory_config_path, 'r') as yaml_cfg:
            data = yaml.safe_load(yaml_cfg)

        if not isinstance(data, dict):
            raise ValueError(f"Malformed memory config: expected YAML object at root in {memory_config_path}")

        accel = data.get('accelerator_config', {}) or {}
        per_core = data.get('per_core_config', {}) or {}
        global_buf_cfg = data.get('global_buffer_config', {}) or {}
        offchip = data.get('offchip_memory_config', {}) or {}

        matrix_unit = per_core.get('matrix_unit', {}) or {}
        vector_unit = per_core.get('vector_unit', {}) or {}
        local_buf_cfg = per_core.get('local_buffer', {}) or {}

        local_buffer = ConfigLoader._buffer_dict(local_buf_cfg)
        global_buffer = ConfigLoader._buffer_dict(global_buf_cfg)

        # Total on-chip buffer available to the matrix unit.
        # two_level: uses global_buffer size — TODO: analytical model for two_level matrix buf split needs revisiting
        matrix_total_buf_kb = global_buffer['mem_size'] if global_buffer['mem_size'] > 0 else local_buffer['mem_size']
        bandwidth_gbps = float(offchip.get('bandwidth', 0))
        latency = int(offchip.get('latency', 0))

        scalesim_hw_config = ConfigLoader._build_scalesim_hw_config(
            accel, matrix_unit, bandwidth_gbps, latency, matrix_total_buf_kb
        )
        mnpusim_params = ConfigLoader._build_mnpusim_params(local_buf_cfg, offchip)

        return {
            'accelerator': {
                'core_dim_row': int(accel.get('core_dim_row', 1)),
                'core_dim_col': int(accel.get('core_dim_col', 1)),
                'clock_frequency': int(accel.get('clock_frequency', 0)),
            },
            'core_dim': {
                'row': int(accel.get('core_dim_row', 1)),
                'col': int(accel.get('core_dim_col', 1)),
            },
            'matrix_unit': {
                'sa_row': int(matrix_unit.get('sa_row', 0)),
                'sa_col': int(matrix_unit.get('sa_col', 0)),
                'dataflow': matrix_unit.get('dataflow', 'WS'),
            },
            'vector_unit': {
                'lanes': int(vector_unit.get('lanes', 128)),
                'sublanes': int(vector_unit.get('sublanes', 8)),
                'alus_per_sublanes': int(vector_unit.get('ALUs_per_sublanes', 4)),
            },
            'local_buffer': local_buffer,
            'global_buffer': global_buffer,
            'cache_config': ConfigLoader._cache_config_view(local_buffer),
            'global_cache_config': ConfigLoader._cache_config_view(global_buffer),
            'offchip': {
                'latency': latency,
                'bandwidth_gbps': bandwidth_gbps,
                'dram_config': offchip.get('dram_config', ''),
                'dram_capacity_per_module': int(offchip.get('dram_capacity_per_module', 0)),
                'module_num': int(offchip.get('module_num', 1)),
            },
            'scalesim_hw_config': scalesim_hw_config,
            'mnpusim_params': mnpusim_params,
        }

    def get_embedding_config(self):
        emb_config = self.config.get('embedding_layer', {})
        embedding_dim = emb_config.get('embedding_dim', 128)
        vectors_per_table = emb_config.get('vectors_per_table', 500000)
        num_tables = emb_config.get('num_tables', 1)
        pooling_factor = emb_config.get('pooling_factor', 1)

        emb_size_str = "-".join([str(vectors_per_table)] * num_tables)

        return {
            'embedding_dim': embedding_dim,
            'vectors_per_table': vectors_per_table,
            'num_tables': num_tables,
            'pooling_factor': pooling_factor,
            'emb_size_str': emb_size_str
        }

    def get_matrix_ops_config_path(self):
        if not os.path.exists(self.csv_path):
            print(f"[WARNING] Matrix Ops CSV config not found: {self.csv_path}")
            return None
        return self.csv_path

    def get_general_config(self):
        return {
            'workload_type': self.config.get('workload_type', 'dlrm'),
            'num_format': self.config.get('num_format', 32)
        }
