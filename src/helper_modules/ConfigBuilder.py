from dataclasses import dataclass
import numpy as np
import os

from .Helper import Helper
from .ConfigLoader import ConfigLoader


@dataclass
class SimConfig:
    debug: bool
    matrix_ops_csv_path: str
    emb_dim: int
    embsize: str
    num_indices_per_lookup: int
    vectors_per_table: int
    num_tables: int
    pooling_factor: int
    n_format_byte: int
    nbatches: int
    bsz: int
    fname: str
    output_dir: str
    prof_multiplier: int
    script_dir: str
    offchip_memory_config: str
    npumem_config: str
    mnpusim_path: str
    mnpusim_config_path: str
    matrix_config_path: str
    mem_size: int
    mem_type: str
    mem_policy: str
    mem_gran: int
    mem_latency: int
    cache_config: list
    num_cores: int
    output_filename: str


def _ensure_positive_int(value, name):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid '{name}': expected positive integer, got {value}")


def _validate_memory_policy(mem_type, mem_policy):
    supported_policies = {
        "cache": {"cache_LRU", "cache_SRRIP", "cache_OPT"},
        "spad": {"spad_naive", "spad_random", "spad_oracle"},
    }

    if mem_type not in supported_policies:
        raise ValueError(
            f"Invalid memory type '{mem_type}'. Supported types: {sorted(supported_policies.keys())}"
        )
    if mem_policy not in supported_policies[mem_type]:
        raise ValueError(
            f"Invalid policy-memory combination: mem_type='{mem_type}', mem_policy='{mem_policy}'. "
            f"Supported policies for '{mem_type}': {sorted(supported_policies[mem_type])}"
        )


def _validate_config_values(
    emb_dim,
    vectors_per_table,
    num_tables,
    pooling_factor,
    n_format_bits,
    nbatches,
    bsz,
    mem_size,
    mem_type,
    mem_policy,
    mem_gran,
    mem_latency,
    cache_config,
    core_row,
    core_col,
):
    _ensure_positive_int(emb_dim, "embedding_dim")
    _ensure_positive_int(vectors_per_table, "vectors_per_table")
    _ensure_positive_int(num_tables, "num_tables")
    _ensure_positive_int(pooling_factor, "pooling_factor")
    _ensure_positive_int(n_format_bits, "num_format")
    _ensure_positive_int(nbatches, "num_batches")
    _ensure_positive_int(bsz, "batch_size")

    _ensure_positive_int(mem_size, "local_buffer.mem_size")
    _ensure_positive_int(mem_gran, "local_buffer.access_granularity")
    _ensure_positive_int(mem_latency, "local_buffer.access_latency")
    _ensure_positive_int(core_row, "core_dim.row")
    _ensure_positive_int(core_col, "core_dim.col")

    _validate_memory_policy(mem_type, mem_policy)

    cache_way = cache_config[0] if len(cache_config) > 0 else 0
    if mem_type == "cache":
        _ensure_positive_int(cache_way, "local_buffer.cache_way")


def build_sim_config(args):
    debug = args.debug

    if debug:
        print(f"[DEBUG] Loading workload config from base path: {args.workload_config}")
    cfg_loader = ConfigLoader(args.workload_config)

    emb_conf = cfg_loader.get_embedding_config()
    gen_conf = cfg_loader.get_general_config()
    matrix_ops_csv_path = cfg_loader.get_matrix_ops_config_path()

    emb_dim = emb_conf['embedding_dim']
    embsize = emb_conf['emb_size_str']
    num_indices_per_lookup = emb_conf['pooling_factor']
    vectors_per_table = emb_conf['vectors_per_table']
    num_tables = emb_conf['num_tables']
    pooling_factor = emb_conf['pooling_factor']

    n_format_bits = gen_conf['num_format']
    n_format_byte = int(np.ceil(n_format_bits / 8))

    nbatches = args.num_batches
    bsz = args.batch_size
    fname = args.data_generation

    if debug:
        print(f"[DEBUG] Matrix Ops CSV Config Path: {matrix_ops_csv_path}")
    if debug:
        print(f"[DEBUG] Generated Embedding Size String: {embsize[:50]}...")

    output_dir = Helper.build_output_dir(
        output_base_dir=args.output_base_dir,
        emb_dim=emb_dim,
        vectors_per_table=vectors_per_table,
        num_tables=num_tables,
        pooling_factor=pooling_factor,
        batch_size=bsz
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if debug:
            print(f"[DEBUG] Created output directory: {output_dir}")

    if args.this_output_dir_file:
        with open(args.this_output_dir_file, "w") as f:
            f.write(output_dir)
        if debug:
            print(f"[DEBUG] Wrote output directory handoff file: {args.this_output_dir_file}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.dirname(script_dir)
    if debug:
        print(f"[DEBUG] script_dir: {script_dir}")

    offchip_memory_config = args.offchip_memory_config
    npumem_config = args.npumem_config
    mnpusim_path = os.path.join(os.path.dirname(script_dir), 'tools', 'mNPUsim')
    if debug:
        print(f"[DEBUG] mnpusim_path: {mnpusim_path}")

    config_path = os.path.join(os.path.dirname(script_dir), 'configs', f'{args.memory_config}.yaml')
    mnpusim_config_path = os.path.join(os.path.dirname(script_dir), 'configs', 'mNPUsim_related')
    matrix_config_path = os.path.join(os.path.dirname(script_dir), 'configs', 'scalesim_config', args.matrix_config)

    if debug:
        print(f"[DEBUG] memory_config_path: {config_path}")
    if debug:
        print(f"[DEBUG] mnpusim_config_path: {mnpusim_config_path}")
    if debug:
        print(f"[DEBUG] matrix_config_path: {matrix_config_path}")

    mem_config = ConfigLoader.load_memory_config(config_path)

    local_buf = mem_config['local_buffer']
    mem_size = local_buf['mem_size']
    mem_type = local_buf['mem_type']
    mem_policy = local_buf['mem_policy']
    mem_gran = local_buf['mem_gran']
    mem_latency = local_buf['mem_latency']
    cache_config = mem_config['cache_config']

    global_buf = mem_config['global_buffer']
    global_mem_size = global_buf['mem_size']
    global_mem_type = global_buf['mem_type']
    global_mem_policy = global_buf['mem_policy']
    global_mem_latency = global_buf['mem_latency']

    core_dim = mem_config['core_dim']
    core_row = core_dim['row']
    core_col = core_dim['col']
    num_cores = core_row * core_col

    _validate_config_values(
        emb_dim=emb_dim,
        vectors_per_table=vectors_per_table,
        num_tables=num_tables,
        pooling_factor=pooling_factor,
        n_format_bits=n_format_bits,
        nbatches=nbatches,
        bsz=bsz,
        mem_size=mem_size,
        mem_type=mem_type,
        mem_policy=mem_policy,
        mem_gran=mem_gran,
        mem_latency=mem_latency,
        cache_config=cache_config,
        core_row=core_row,
        core_col=core_col,
    )

    vector_unit = mem_config['vector_unit']
    vector_lanes = vector_unit['lanes']
    vector_sublanes = vector_unit['sublanes']
    vector_alus_per_sublanes = vector_unit['alus_per_sublanes']

    matrix_unit = mem_config['matrix_unit']
    mxu_dimension = matrix_unit['mxu_dimension']
    num_mxus = matrix_unit['num_mxus']

    if debug:
        print(f"[DEBUG] Core Dimension - Row: {core_row}, Col: {core_col}")
    if debug:
        print(f"[DEBUG] Vector Unit - Lanes: {vector_lanes}, Sublanes: {vector_sublanes}, ALUs per sublanes: {vector_alus_per_sublanes}")
    if debug:
        print(f"[DEBUG] Matrix Unit - MXU dimension: {mxu_dimension}, Number of MXUs: {num_mxus}")
    if debug:
        print(f"[DEBUG] Local Buffer - Type: {mem_type}, Size: {mem_size} KB, Policy: {mem_policy}, Latency: {mem_latency} cycles")
    if debug and global_mem_size > 0:
        print(f"[DEBUG] Global Buffer - Type: {global_mem_type}, Size: {global_mem_size} KB, Policy: {global_mem_policy}, Latency: {global_mem_latency} cycles")

    return SimConfig(
        debug=debug,
        matrix_ops_csv_path=matrix_ops_csv_path,
        emb_dim=emb_dim,
        embsize=embsize,
        num_indices_per_lookup=num_indices_per_lookup,
        vectors_per_table=vectors_per_table,
        num_tables=num_tables,
        pooling_factor=pooling_factor,
        n_format_byte=n_format_byte,
        nbatches=nbatches,
        bsz=bsz,
        fname=fname,
        output_dir=output_dir,
        prof_multiplier=args.profiling_multiplier,
        script_dir=script_dir,
        offchip_memory_config=offchip_memory_config,
        npumem_config=npumem_config,
        mnpusim_path=mnpusim_path,
        mnpusim_config_path=mnpusim_config_path,
        matrix_config_path=matrix_config_path,
        mem_size=mem_size,
        mem_type=mem_type,
        mem_policy=mem_policy,
        mem_gran=mem_gran,
        mem_latency=mem_latency,
        cache_config=cache_config,
        num_cores=num_cores,
        output_filename=args.output_filename,
    )
