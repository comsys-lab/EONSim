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
