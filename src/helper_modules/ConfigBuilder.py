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
    prof_period: int
    script_dir: str
    offchip_memory_config: str
    npumem_config: str
    global_bandwidth_bytes_per_cycle: float
    global_access_latency_cycles: int
    mnpusim_path: str
    mnpusim_config_path: str
    matrix_config_path: str
    onchip_structure: str
    local_onmem_type: str
    local_onmem_policy: str
    local_onmem_size: int
    local_onmem_latency: int
    global_onmem_type: str
    global_onmem_policy: str
    global_onmem_size: int
    global_onmem_latency: int
    mem_size: int
    mem_type: str
    mem_policy: str
    mem_gran: int
    mem_latency: int
    cache_config: dict
    num_cores: int
    workload_type: str
    vector_lanes: int
    vector_sublanes: int
    vector_alus_per_sublanes: int
    mxu_dimension: int
    num_mxus: int
    output_filename: str
    warmup_batches: int


def _ensure_positive_int(value, name):
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"Invalid '{name}': expected positive integer, got {value}")


def _validate_memory_policy(mem_type, mem_policy):
    supported_policies = {
        "cache": {"cache_LRU", "cache_SRRIP", "cache_LFU", "cache_OPT"},
        "spad": {"spad_naive", "spad_random", "spad_oracle"},
        "profile": {"profile_dynamic_cache", "profile_dynamic_SRRIP", "profile_dynamic_count"},
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


def _determine_onchip_structure(local_onmem_size, global_onmem_size):
    if local_onmem_size <= 0 and global_onmem_size > 0:
        return "global_only"
    if local_onmem_size > 0 and global_onmem_size <= 0:
        return "local_only"
    if local_onmem_size > 0 and global_onmem_size > 0:
        return "two_level"
    raise ValueError(
        "Invalid on-chip memory configuration: both local_buffer.mem_size and global_buffer.mem_size are zero."
    )


def _validate_config_values(
    emb_dim,
    vectors_per_table,
    num_tables,
    pooling_factor,
    n_format_bits,
    nbatches,
    bsz,
    onchip_structure,
    onmem,
    onmem_cache_config,
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

    if onchip_structure not in {"global_only", "local_only", "two_level"}:
        raise ValueError(f"Invalid on-chip structure: {onchip_structure}")

    _ensure_positive_int(onmem['mem_size'], "onmem.mem_size")
    _ensure_positive_int(onmem['mem_gran'], "onmem.access_granularity")
    _ensure_positive_int(onmem['mem_latency'], "onmem.access_latency")
    _ensure_positive_int(core_row, "core_dim.row")
    _ensure_positive_int(core_col, "core_dim.col")

    _validate_memory_policy(onmem['mem_type'], onmem['mem_policy'])

    cache_way = onmem_cache_config.get('way', 0)
    if onmem['mem_type'] == "cache":
        _ensure_positive_int(cache_way, "local_buffer.cache_way")

    if onmem['mem_policy'] == "cache_LFU":
        lfu_counter_bits = onmem_cache_config.get('lfu_counter_bits', 0)
        lfu_aging_interval = onmem_cache_config.get('lfu_aging_interval', -1)
        _ensure_positive_int(lfu_counter_bits, "local_buffer.lfu_counter_bits")
        if not isinstance(lfu_aging_interval, int) or lfu_aging_interval < 0:
            raise ValueError(
                f"Invalid 'local_buffer.lfu_aging_interval': expected non-negative integer, got {lfu_aging_interval}"
            )

    num_cores = core_row * core_col
    if num_cores > num_tables:
        print(f"[WARNING] num_cores ({num_cores}) > num_tables ({num_tables}): "
              f"{num_cores - num_tables} core(s) will be idle.")


def build_sim_config(args):
    debug = args.debug

    if debug:
        print(f"[DEBUG] Loading workload config from base path: {args.workload_config}")
    cfg_loader = ConfigLoader(args.workload_config)

    emb_conf = cfg_loader.get_embedding_config()
    gen_conf = cfg_loader.get_general_config()
    matrix_ops_csv_path = cfg_loader.get_matrix_ops_config_path()

    raw_workload_type = gen_conf.get('workload_type', '')
    workload_type = str(raw_workload_type).strip().lower()
    if workload_type == '':
        raise ValueError(
            "Invalid workload config: 'workload_type' must be a non-empty string"
        )
    if debug:
        print(f"[DEBUG] Workload type from YAML: {workload_type}")

    emb_dim = emb_conf['embedding_dim']
    embsize = emb_conf['emb_size_str']
    num_indices_per_lookup = emb_conf['pooling_factor']
    vectors_per_table = emb_conf['vectors_per_table']
    num_tables = emb_conf['num_tables']
    pooling_factor = emb_conf['pooling_factor']

    n_format_bits = gen_conf['num_format']
    n_format_byte = int(np.ceil(n_format_bits / 8))

    nbatches = args.num_batches
    warmup_batches = getattr(args, 'warmup_batches', 0)
    if warmup_batches < 0:
        raise ValueError(f"Invalid '--warmup-batches': expected non-negative integer, got {warmup_batches}")
    if warmup_batches >= nbatches:
        raise ValueError(
            f"'--warmup-batches' ({warmup_batches}) must be less than '--num-batches' ({nbatches})"
        )
    bsz = args.batch_size
    fname = args.data_generation

    if debug:
        print(f"[DEBUG] Matrix Ops CSV Config Path: {matrix_ops_csv_path}")
    if debug:
        print(f"[DEBUG] Generated Embedding Size String: {embsize[:50]}...")

    if not isinstance(args.profiling_period, int) or args.profiling_period <= 0:
        raise ValueError(
            f"Invalid '--profiling-period': expected positive integer, got {args.profiling_period}"
        )

    if args.global_bandwidth_bytes_per_cycle < 0:
        raise ValueError(
            "Invalid '--global-bandwidth-bytes-per-cycle': expected non-negative value"
        )
    if args.global_access_latency_cycles < 0:
        raise ValueError(
            "Invalid '--global-access-latency-cycles': expected non-negative integer"
        )

    output_dir = Helper.build_output_dir(
        output_base_dir=args.output_base_dir,
        emb_dim=emb_dim,
        vectors_per_table=vectors_per_table,
        num_tables=num_tables,
        pooling_factor=pooling_factor,
        batch_size=bsz,
        dataset_path=fname
    )
    
    # Append config_name as a subdirectory
    config_name = os.path.splitext(os.path.basename(args.memory_config))[0]
    output_dir = os.path.join(output_dir, config_name)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if debug:
            print(f"[DEBUG] Created output directory: {output_dir}")

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
    global_buf = mem_config['global_buffer']

    onchip_structure = _determine_onchip_structure(local_buf['mem_size'], global_buf['mem_size'])

    if onchip_structure in {"global_only", "two_level"}:
        onmem = global_buf
        cache_config = mem_config['global_cache_config']
    else:
        onmem = local_buf
        cache_config = mem_config['cache_config']

    mem_size = onmem['mem_size']
    mem_type = onmem['mem_type']
    mem_policy = onmem['mem_policy']
    mem_gran = onmem['mem_gran']
    mem_latency = onmem['mem_latency']

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
        onchip_structure=onchip_structure,
        onmem=onmem,
        onmem_cache_config=cache_config,
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
        print(f"[DEBUG] On-chip structure: {onchip_structure}")
    if debug:
        print(f"[DEBUG] Core Dimension - Row: {core_row}, Col: {core_col}")
    if debug:
        print(f"[DEBUG] Vector Unit - Lanes: {vector_lanes}, Sublanes: {vector_sublanes}, ALUs per sublanes: {vector_alus_per_sublanes}")
    if debug:
        print(f"[DEBUG] Matrix Unit - MXU dimension: {mxu_dimension}, Number of MXUs: {num_mxus}")
    if debug:
        if onchip_structure == "local_only":
            print(
                f"[DEBUG] Local Buffer - Type: {mem_type}, Size: {mem_size} KB, "
                f"Policy: {mem_policy}, Latency: {mem_latency} cycles"
            )
        else:
            print(
                f"[DEBUG] Global Buffer - Type: {mem_type}, Size: {mem_size} KB, "
                f"Policy: {mem_policy}, Latency: {mem_latency} cycles"
            )

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
        prof_period=args.profiling_period,
        script_dir=script_dir,
        offchip_memory_config=offchip_memory_config,
        npumem_config=npumem_config,
        global_bandwidth_bytes_per_cycle=args.global_bandwidth_bytes_per_cycle,
        global_access_latency_cycles=args.global_access_latency_cycles,
        mnpusim_path=mnpusim_path,
        mnpusim_config_path=mnpusim_config_path,
        matrix_config_path=matrix_config_path,
        onchip_structure=onchip_structure,
        local_onmem_type=local_buf['mem_type'],
        local_onmem_policy=local_buf['mem_policy'],
        local_onmem_size=local_buf['mem_size'],
        local_onmem_latency=local_buf['mem_latency'],
        global_onmem_type=global_buf['mem_type'],
        global_onmem_policy=global_buf['mem_policy'],
        global_onmem_size=global_buf['mem_size'],
        global_onmem_latency=global_buf['mem_latency'],
        mem_size=mem_size,
        mem_type=mem_type,
        mem_policy=mem_policy,
        mem_gran=mem_gran,
        mem_latency=mem_latency,
        cache_config=cache_config,
        num_cores=num_cores,
        workload_type=workload_type,
        vector_lanes=vector_lanes,
        vector_sublanes=vector_sublanes,
        vector_alus_per_sublanes=vector_alus_per_sublanes,
        mxu_dimension=mxu_dimension,
        num_mxus=num_mxus,
        output_filename=args.output_filename,
        warmup_batches=warmup_batches,
    )
