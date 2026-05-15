import numpy as np
from tqdm import tqdm
from helper_modules.helper import print_styled_header, print_styled_box
from onmem_policies import LRUPolicy, SRRIPPolicy, LFUPolicy, OptPolicy, SpmPolicy, ProfilePolicy

class CoreAccessIterator:
    """Iterator for a core's memory accesses in table-partitioned multicore simulation"""
    def __init__(self, emb_dataset, table_start, table_end, offmem_trace):
        self.emb_dataset = emb_dataset
        self.table_start = table_start
        self.table_end = table_end
        self.offmem_trace = offmem_trace
        
        self.batch_idx = 0
        self.table_idx = table_start
        self.vec_idx = 0
        
        self.total_accesses = self._count_total_accesses()
        self.current_access = 0
    
    def _count_total_accesses(self):
        """Count total number of accesses this core will make"""
        total = 0
        for batch in self.emb_dataset:
            for table_id in range(self.table_start, self.table_end):
                total += len(batch[table_id])
        return total
    
    def has_next(self):
        """Check if iterator has more accesses"""
        return self.current_access < self.total_accesses
    
    def get_next(self):
        """Get next memory address and its location"""
        if not self.has_next():
            return None
        
        addr = self.emb_dataset[self.batch_idx][self.table_idx][self.vec_idx]
        result = (self.batch_idx, self.table_idx, self.vec_idx, addr)
        
        self._advance()
        return result
    
    def _advance(self):
        """Move to next vector/table/batch"""
        self.vec_idx += 1
        self.current_access += 1
        
        if self.vec_idx >= len(self.emb_dataset[self.batch_idx][self.table_idx]):
            self.vec_idx = 0
            self.table_idx += 1
            
            if self.table_idx >= self.table_end:
                self.table_idx = self.table_start
                self.batch_idx += 1
                
                if self.batch_idx >= len(self.emb_dataset):
                    self.batch_idx = len(self.emb_dataset) - 1


class CoreOnmem:
    def __init__(self, mem_size, mem_type, cache_config, emb_dim, emb_dataset, n_format_byte, vectors_per_table=0, mem_gran=0, prof_period=1, mem_latency=1, num_cores=1, onchip_structure="global_only", local_onmem_config=None, global_onmem_config=None, index_trace=None, debug=False):
        self.num_cores = num_cores
        self.onchip_structure = onchip_structure
        self.debug = debug
        self.core_access_results = [[] for _ in range(num_cores)]
        self.core_policies = []
        self.local_onmem_config = dict(local_onmem_config) if local_onmem_config else {}
        self.global_onmem_config = dict(global_onmem_config) if global_onmem_config else {}
        self.index_trace = index_trace

        self.mem_policy = "init"
        self.on_mem = np.ones(1)
        self.batch_counter = 0
        self.profile_batch_counter = 0
        self.access_results = []
        self.spm_load_results = []

        self.addr_bits = 64  # set address width here

        self.mem_size = mem_size * 1024  # KB -> Byte
        self.mem_type = mem_type
        self.mem_latency = mem_latency
        self.emb_dim = emb_dim
        self.emb_dataset = emb_dataset
        self.n_format_byte = n_format_byte
        self.mem_gran = mem_gran
        self.prof_period = prof_period
        self.vectors_per_table = vectors_per_table

        if mem_gran > 0 and emb_dim > 0 and n_format_byte > 0 and vectors_per_table > 0 and emb_dataset:
            num_tables = len(emb_dataset[0])
            access_per_vector = max(1, int(np.ceil(emb_dim * n_format_byte / mem_gran)))
            max_addr = num_tables * vectors_per_table * access_per_vector * mem_gran
            if max_addr.bit_length() > self.addr_bits:
                raise ValueError(
                    f"Workload address space requires {max_addr.bit_length()} bits "
                    f"but addr_bits={self.addr_bits}. Adjust addr_bits in CoreOnmem.__init__()."
                )

        self.cache_config = {}
        self.cache_way = 0
        self.cache_line_size = 0
        self.cache_set = 0
        self.cache_index_bits = 0
        self.cache_offset_bits = 0
        self.cache_tag_bits = 0
        self.rrpv_bits = 0
        self.rrpv_insert = 0
        self.lfu_counter_bits = 8
        self.lfu_aging_interval = 0

        if self.mem_type == "cache":
            self.cache_config = dict(cache_config)
            self.cache_way = self.cache_config.get('way', 0)
            self.cache_line_size = self.cache_config.get('line_size', 0)
            self.cache_set = int(self.mem_size / self.cache_line_size / self.cache_way)
            self.cache_index_bits = int(np.ceil(np.log2(self.cache_set))) if self.cache_set > 1 else 0
            self.cache_offset_bits = int(np.log2(self.cache_line_size-1)+1)
            self.cache_tag_bits = self.addr_bits - self.cache_index_bits - self.cache_offset_bits
            if self.cache_tag_bits <= 0:
                raise ValueError(
                    f"Cache geometry needs {self.cache_index_bits + self.cache_offset_bits} address bits "
                    f"(index + offset) but addr_bits={self.addr_bits}. "
                    f"Reduce mem_size, increase cache_way, or decrease access_granularity."
                )
            self.rrpv_bits = self.cache_config.get('rrpv_bits', 0)
            self.rrpv_insert = self.cache_config.get('rrpv_insert', 0)
            self.lfu_counter_bits = self.cache_config.get('lfu_counter_bits', 8)
            self.lfu_aging_interval = self.cache_config.get('lfu_aging_interval', 0)
        elif self.mem_type == "profile":
            self.cache_config = dict(cache_config)
            # Preserve legacy profiling defaults when cache fields are not provided.
            self.cache_way = self.cache_config.get('way', 0) or 128
            self.cache_line_size = self.cache_config.get('line_size', 0) or mem_gran
            self.cache_set = max(1, int(self.mem_size / self.cache_line_size / self.cache_way))
            self.cache_index_bits = int(np.ceil(np.log2(self.cache_set))) if self.cache_set > 1 else 0
            self.cache_offset_bits = int(np.log2(self.cache_line_size - 1) + 1)
            self.cache_tag_bits = self.addr_bits - self.cache_index_bits - self.cache_offset_bits
            if self.cache_tag_bits <= 0:
                raise ValueError(
                    f"Cache geometry needs {self.cache_index_bits + self.cache_offset_bits} address bits "
                    f"(index + offset) but addr_bits={self.addr_bits}. "
                    f"Reduce mem_size, increase cache_way, or decrease access_granularity."
                )
            self.rrpv_bits = self.cache_config.get('rrpv_bits', 0)
            self.rrpv_insert = self.cache_config.get('rrpv_insert', 0)

        self.offmem_trace = [[np.full_like(self.emb_dataset[nb][nt], -1) for nt in range(len(self.emb_dataset[nb]))] for nb in range(len(self.emb_dataset))]
        if self.debug: print("[DEBUG] self.offmem_trace shape: ({}, {}, {})".format(len(self.offmem_trace), len(self.offmem_trace[0]), len(self.offmem_trace[0][0])))
    
    def _partition_tables_across_cores(self):
        """Partition embedding tables across multiple cores"""
        total_tables = len(self.emb_dataset[0])
        tables_per_core = total_tables // self.num_cores
        remainder = total_tables % self.num_cores
        
        table_ranges = []
        start_idx = 0
        
        for core_id in range(self.num_cores):
            extra = 1 if core_id < remainder else 0
            end_idx = start_idx + tables_per_core + extra
            table_ranges.append((start_idx, end_idx))
            start_idx = end_idx
        
        return table_ranges

    def set_policy(self, policy):
        self.mem_policy = policy
        if self.mem_type == "cache":
            if not policy.startswith("cache_"):
                raise ValueError(f"Invalid policy: '{policy}' for mem_type: '{self.mem_type}'")

            cache_config = dict(self.cache_config)
            cache_config['set_count'] = self.cache_set

            if self.onchip_structure in {"global_only", "two_level"}:
                # In two-level mode, local buffer is treated as a prefetching buffer.
                self.policy = self._create_cache_policy(self.mem_policy, cache_config, self.emb_dataset, self.num_cores)
                self.policy.initialize()
                self.on_mem = self.policy.on_mem
            elif self.onchip_structure == "local_only":
                self.core_policies = []
                table_ranges = self._partition_tables_across_cores()
                for core_id, (table_start, table_end) in enumerate(table_ranges):
                    core_dataset = [batch[table_start:table_end] for batch in self.emb_dataset]
                    core_policy = self._create_cache_policy(self.mem_policy, cache_config, core_dataset, 1)
                    core_policy.initialize()
                    self.core_policies.append(core_policy)

                self.on_mem = [policy.on_mem for policy in self.core_policies]
            else:
                raise NotImplementedError(f"Unknown on-chip structure: {self.onchip_structure}")
        elif self.mem_type == "spm":
            if not policy.startswith("spm_"):
                raise ValueError(f"Invalid policy: '{policy}' for mem_type: '{self.mem_type}'")
            self.policy = SpmPolicy(self.mem_size, self.mem_gran, self.emb_dim, self.n_format_byte, self.emb_dataset, self.vectors_per_table, self.prof_period, self.mem_policy, self.num_cores, debug=self.debug)
            self.policy.initialize()
            self.on_mem = self.policy.on_mem
        elif self.mem_type == "profile":
            if not policy.startswith("profile_"):
                raise ValueError(f"Invalid policy: '{policy}' for mem_type: '{self.mem_type}'")

            profile_cache_config = dict(self.cache_config)
            profile_cache_config['set_count'] = self.cache_set

            if self.onchip_structure in {"global_only", "two_level"}:
                self.policy = self._create_profile_policy(
                    self.mem_policy,
                    profile_cache_config,
                    self.emb_dataset,
                    self.index_trace,
                )
                self.policy.initialize()
                self.on_mem = self.policy.on_mem
            elif self.onchip_structure == "local_only":
                self.core_policies = []
                table_ranges = self._partition_tables_across_cores()
                for core_id, (table_start, table_end) in enumerate(table_ranges):
                    core_dataset = [batch[table_start:table_end] for batch in self.emb_dataset]
                    core_index_trace = None
                    if self.index_trace is not None:
                        core_index_trace = [batch[table_start:table_end] for batch in self.index_trace]

                    core_policy = self._create_profile_policy(
                        self.mem_policy,
                        profile_cache_config,
                        core_dataset,
                        core_index_trace,
                    )
                    core_policy.initialize()
                    self.core_policies.append(core_policy)

                self.on_mem = [policy.on_mem for policy in self.core_policies]
            else:
                raise NotImplementedError(f"Unknown on-chip structure: {self.onchip_structure}")

    def _create_cache_policy(self, policy, cache_config, policy_dataset, policy_num_cores):
        if policy == "cache_LRU":
            return LRUPolicy(cache_config)
        if policy == "cache_SRRIP":
            return SRRIPPolicy(cache_config)
        if policy == "cache_LFU":
            return LFUPolicy(cache_config)
        if policy == "cache_OPT":
            return OptPolicy(cache_config, policy_dataset, policy_num_cores)
        raise NotImplementedError(f"Policy {policy} not implemented")

    def _create_profile_policy(self, policy, cache_config, policy_dataset, policy_index_trace):
        return ProfilePolicy(
            mem_size=self.mem_size // 1024,
            mem_gran=self.mem_gran,
            emb_dim=self.emb_dim,
            n_format_byte=self.n_format_byte,
            emb_dataset=policy_dataset,
            vectors_per_table=self.vectors_per_table,
            prof_period=self.prof_period,
            profile_policy=policy,
            cache_config=cache_config,
            index_trace=policy_index_trace,
            debug=self.debug,
        )

    def print_config(self):
        content = [
            f"Number of cores: {self.num_cores}",
            f"On-chip structure: {self.onchip_structure}",
        ]

        local_size = int(self.local_onmem_config.get("mem_size", 0) or 0)
        global_size = int(self.global_onmem_config.get("mem_size", 0) or 0)

        if self.onchip_structure in {"local_only", "two_level"} and local_size > 0:
            content.extend([
                f"Local on-chip memory size: {local_size} KB",
                f"Local on-chip memory type: {self.local_onmem_config.get('mem_type', 'N/A')}",
                f"Local on-chip memory policy: {self.local_onmem_config.get('mem_policy', 'N/A')}",
                f"Local on-chip memory latency: {self.local_onmem_config.get('mem_latency', 'N/A')} cycles",
            ])

        if self.onchip_structure in {"global_only", "two_level"} and global_size > 0:
            content.extend([
                f"Global on-chip memory size: {global_size} KB",
                f"Global on-chip memory type: {self.global_onmem_config.get('mem_type', self.mem_type)}",
                f"Global on-chip memory policy: {self.global_onmem_config.get('mem_policy', self.mem_policy)}",
                f"Global on-chip memory latency: {self.global_onmem_config.get('mem_latency', self.mem_latency)} cycles",
            ])
            
        if self.mem_type == "cache":
            if self.onchip_structure == "local_only":
                content.extend([
                    f"Local on-chip memory (cache) way: {self.cache_way}-way",
                    f"Local on-chip memory (cache) line size: {self.cache_line_size} B",
                    f"Local on-chip memory (cache) set: {self.cache_set} sets",
                    f"Local on-chip memory (cache) tag bits: {self.cache_tag_bits} bits",
                ])
            else:
                content.extend([
                    f"Global on-chip memory (cache) way: {self.cache_way}-way",
                    f"Global on-chip memory (cache) line size: {self.cache_line_size} B",
                    f"Global on-chip memory (cache) set: {self.cache_set} sets",
                    f"Global on-chip memory (cache) tag bits: {self.cache_tag_bits} bits",
                ])

        print_styled_box("On-Chip Memory Configuration", content)

    def print_sim(self):
        print_styled_header("Simulation Start")

    def get_tag_bits(self, addr):
        # make bits lower than tag bits to zero
        tag_mask_bits = (1 << (self.cache_index_bits + self.cache_offset_bits)) - 1
        return addr & ~tag_mask_bits

    def get_index_bits(self, addr):
        if self.cache_index_bits == 0:
            return 0
        
        index_msb = self.cache_index_bits + self.cache_offset_bits - 1
        index_lsb = self.cache_offset_bits
        mask = ((1 << (index_msb - index_lsb + 1)) - 1) << index_lsb
        index_bits = (addr & mask) >> index_lsb    # extract only index bits
        
        # Ensure index is within bounds for non-power-of-2 cache sets, this is just for 384 MB cache simulation.
        return index_bits % self.cache_set

    def do_simulation(self):
        self.print_sim()
        if self.debug: print(f"[DEBUG] Running with {self.num_cores} core(s)")

        # Partition tables across cores
        table_ranges = self._partition_tables_across_cores()
        for core_id in range(self.num_cores):
            table_start, table_end = table_ranges[core_id]
            if self.debug: print(f"[DEBUG] [Core {core_id}] Tables {table_start}-{table_end-1}")

        for nb in range(len(self.emb_dataset)):
            print(f"Processing batch {nb}...")

            if self.mem_type == 'spm':
                batch_hit, batch_miss, core_batch_stats = self._simulate_spm_batch(nb, table_ranges)
            elif self.mem_type == 'cache':
                batch_hit, batch_miss, core_batch_stats = self._simulate_cache_batch(nb, table_ranges)
            elif self.mem_type == 'profile':
                batch_hit, batch_miss, core_batch_stats = self._simulate_profile_batch(nb, table_ranges)
            else:
                raise NotImplementedError(f"mem_type '{self.mem_type}' is not implemented.")

            self.access_results.append([batch_hit, batch_miss])
            for core_id in range(self.num_cores):  # kept for potential future use
                self.core_access_results[core_id].append(core_batch_stats[core_id])

        # Verify offmem_trace integrity
        total_miss = sum(r[1] for r in self.access_results)
        offmem_count = 0
        for nb in range(len(self.offmem_trace)):
            for nt in range(len(self.offmem_trace[nb])):
                offmem_count += np.sum(self.offmem_trace[nb][nt] != -1)

        if self.debug: print(f"[DEBUG] Total misses: {total_miss}, Offmem trace entries: {offmem_count}")
        if offmem_count != total_miss:
            print(f"[WARNING] Mismatch between miss count and offmem_trace entries!")

        print("Simulation Done")
        self.print_stats()

    def _build_core_iterators_for_batch(self, nb, table_ranges):
        # Cache/profile paths iterate one batch at a time.
        single_batch = [self.emb_dataset[nb]]
        core_iterators = []
        for core_id in range(self.num_cores):
            table_start, table_end = table_ranges[core_id]
            core_iterators.append(CoreAccessIterator(single_batch, table_start, table_end, None))
        total_accesses = sum(it.total_accesses for it in core_iterators)
        return core_iterators, total_accesses

    def _simulate_spm_batch(self, nb, table_ranges):
        batch_hit = 0
        batch_miss = 0
        core_batch_stats = [[0, 0] for _ in range(self.num_cores)]

        if isinstance(self.on_mem, set):
            on_mem_array = np.array(list(self.on_mem), dtype=np.int64)
        else:
            on_mem_array = self.on_mem

        with tqdm(total=len(self.emb_dataset[nb]), desc=f"Batch {nb}") as pbar:
            for core_id in range(self.num_cores):
                table_start, table_end = table_ranges[core_id]
                for nt in range(table_start, table_end):
                    hit_mask = np.isin(self.emb_dataset[nb][nt], on_mem_array)

                    num_hit = np.sum(hit_mask)
                    num_miss = np.sum(~hit_mask)

                    batch_hit += num_hit
                    batch_miss += num_miss
                    core_batch_stats[core_id][0] += num_hit
                    core_batch_stats[core_id][1] += num_miss

                    miss_mask = ~hit_mask
                    self.offmem_trace[nb][nt][miss_mask] = self.emb_dataset[nb][nt][miss_mask]
                    pbar.update(1)

        # Update on_mem for oracle policy after each batch.
        if self.mem_policy == "spm_oracle":
            self.batch_counter = min(self.batch_counter + 1, len(self.emb_dataset) - 1)
            self.policy.batch_counter = self.batch_counter
            if self.batch_counter % self.prof_period == 0:
                self.on_mem = self.policy.set_spm()

        return batch_hit, batch_miss, core_batch_stats

    def _simulate_cache_batch(self, nb, table_ranges):
        batch_hit = 0
        batch_miss = 0
        core_batch_stats = [[0, 0] for _ in range(self.num_cores)]

        core_iterators, total_accesses = self._build_core_iterators_for_batch(nb, table_ranges)

        if self.onchip_structure in {"global_only", "two_level"}:
            tick = 0
            active_cores = [True] * self.num_cores

            with tqdm(total=total_accesses, desc=f"Batch {nb}") as pbar:
                while any(active_cores):
                    core_id = tick % self.num_cores

                    if active_cores[core_id] and core_iterators[core_id].has_next():
                        _, table_id, vec_id, addr = core_iterators[core_id].get_next()
                        tag = self.get_tag_bits(addr)
                        index = self.get_index_bits(addr)
                        hit, _ = self.policy.handle_access(tag, index)

                        if hit:
                            batch_hit += 1
                            core_batch_stats[core_id][0] += 1
                        else:
                            batch_miss += 1
                            core_batch_stats[core_id][1] += 1
                            self.offmem_trace[nb][table_id][vec_id] = addr

                        self.policy.post_access_processing(hit, tag, index, vec_id)
                        pbar.update(1)

                        if not core_iterators[core_id].has_next():
                            active_cores[core_id] = False
                    elif active_cores[core_id]:
                        active_cores[core_id] = False

                    tick += 1

        elif self.onchip_structure == "local_only":
            with tqdm(total=total_accesses, desc=f"Batch {nb}") as pbar:
                for core_id in range(self.num_cores):
                    core_policy = self.core_policies[core_id]
                    while core_iterators[core_id].has_next():
                        _, table_id, vec_id, addr = core_iterators[core_id].get_next()
                        tag = self.get_tag_bits(addr)
                        index = self.get_index_bits(addr)
                        hit, _ = core_policy.handle_access(tag, index)

                        if hit:
                            batch_hit += 1
                            core_batch_stats[core_id][0] += 1
                        else:
                            batch_miss += 1
                            core_batch_stats[core_id][1] += 1
                            self.offmem_trace[nb][table_id][vec_id] = addr

                        core_policy.post_access_processing(hit, tag, index, vec_id)
                        pbar.update(1)

        else:
            raise NotImplementedError(f"onchip_structure '{self.onchip_structure}' is not implemented.")

        return batch_hit, batch_miss, core_batch_stats

    def _simulate_profile_batch(self, nb, table_ranges):
        batch_hit = 0
        batch_miss = 0
        core_batch_stats = [[0, 0] for _ in range(self.num_cores)]

        core_iterators, total_accesses = self._build_core_iterators_for_batch(nb, table_ranges)

        if self.onchip_structure in {"global_only", "two_level"}:
            self.policy.begin_batch(nb)
            tick = 0
            active_cores = [True] * self.num_cores

            with tqdm(total=total_accesses, desc=f"Batch {nb}") as pbar:
                while any(active_cores):
                    core_id = tick % self.num_cores

                    if active_cores[core_id] and core_iterators[core_id].has_next():
                        _, table_id, vec_id, addr = core_iterators[core_id].get_next()
                        tag = self.get_tag_bits(addr)
                        index = self.get_index_bits(addr)

                        hit, _ = self.policy.handle_access(
                            tag,
                            index,
                            addr=addr,
                            table_id=table_id,
                            vec_id=vec_id,
                            batch_idx=nb,
                            core_id=core_id,
                        )

                        if hit:
                            batch_hit += 1
                            core_batch_stats[core_id][0] += 1
                        else:
                            batch_miss += 1
                            core_batch_stats[core_id][1] += 1
                            self.offmem_trace[nb][table_id][vec_id] = addr

                        self.policy.post_access_processing(hit, tag, index, vec_id)
                        pbar.update(1)

                        if not core_iterators[core_id].has_next():
                            active_cores[core_id] = False
                    elif active_cores[core_id]:
                        active_cores[core_id] = False

                    tick += 1

            self.profile_batch_counter += 1
            if self.profile_batch_counter % self.prof_period == 0:
                self.policy.refresh_on_mem()

            self.policy.end_batch(nb)
            self.on_mem = self.policy.on_mem

        elif self.onchip_structure == "local_only":
            for core_policy in self.core_policies:
                core_policy.begin_batch(nb)

            with tqdm(total=total_accesses, desc=f"Batch {nb}") as pbar:
                for core_id in range(self.num_cores):
                    core_policy = self.core_policies[core_id]
                    table_start, _ = table_ranges[core_id]
                    while core_iterators[core_id].has_next():
                        _, table_id, vec_id, addr = core_iterators[core_id].get_next()
                        tag = self.get_tag_bits(addr)
                        index = self.get_index_bits(addr)

                        hit, _ = core_policy.handle_access(
                            tag,
                            index,
                            addr=addr,
                            table_id=table_id - table_start,
                            vec_id=vec_id,
                            batch_idx=nb,
                            core_id=core_id,
                        )

                        if hit:
                            batch_hit += 1
                            core_batch_stats[core_id][0] += 1
                        else:
                            batch_miss += 1
                            core_batch_stats[core_id][1] += 1
                            self.offmem_trace[nb][table_id][vec_id] = addr

                        core_policy.post_access_processing(hit, tag, index, vec_id)
                        pbar.update(1)

            self.profile_batch_counter += 1
            if self.profile_batch_counter % self.prof_period == 0:
                for core_policy in self.core_policies:
                    core_policy.refresh_on_mem()

            for core_policy in self.core_policies:
                core_policy.end_batch(nb)
            self.on_mem = [policy.on_mem for policy in self.core_policies]

        else:
            raise NotImplementedError(f"onchip_structure '{self.onchip_structure}' is not implemented.")

        return batch_hit, batch_miss, core_batch_stats

    def _get_profile_logger_results(self):
        if self.mem_type != 'profile':
            return None
        if self.mem_policy not in {"profile_dynamic_cache", "profile_dynamic_SRRIP"}:
            return None

        if self.onchip_structure in {"global_only", "two_level"}:
            return getattr(self.policy, "logger_results", None)

        if self.onchip_structure == "local_only":
            if not self.core_policies:
                return None
            num_batches = len(self.access_results)
            aggregated = []
            for batch_idx in range(num_batches):
                batch_hit = 0
                batch_miss = 0
                for core_policy in self.core_policies:
                    core_logger = getattr(core_policy, "logger_results", None)
                    if core_logger is None or batch_idx >= len(core_logger):
                        continue
                    batch_hit += core_logger[batch_idx][0]
                    batch_miss += core_logger[batch_idx][1]
                aggregated.append([batch_hit, batch_miss])
            return aggregated

        return None
        
    def print_stats(self):
        total_hits = 0
        total_miss = 0
        for i in range(len(self.access_results)):
            total_hits += self.access_results[i][0]
            total_miss += self.access_results[i][1]
        total_hit_ratio = total_hits / (total_hits + total_miss)
        
        content = [
            f"Total hit ratio: {total_hit_ratio:.4f}",
            f"Total accesses: {total_hits+total_miss}",
            f"Total hits: {total_hits}",
            f"Total misses: {total_miss}",
            "----------------------------------------",
            "Per batch results"
        ]
        
        for i in range(len(self.access_results)):
            batch_hit_ratio = self.access_results[i][0] / (self.access_results[i][0] + self.access_results[i][1])
            content.append(
                f"[Batch {i}] hit ratio: {batch_hit_ratio:.4f}   "
                f"accesses: {self.access_results[i][0]+self.access_results[i][1]}   "
                f"hits: {self.access_results[i][0]}   misses: {self.access_results[i][1]}"
            )

        if self.debug:
            logger_results = self._get_profile_logger_results()
            if logger_results:
                content.append("----------------------------------------")
                content.append("Per batch logger results")
                for i, logger_stats in enumerate(logger_results):
                    logger_hit = logger_stats[0]
                    logger_miss = logger_stats[1]
                    logger_access = logger_hit + logger_miss
                    logger_hit_ratio = (logger_hit / logger_access) if logger_access > 0 else 0.0
                    content.append(
                        f"[Batch {i}] logger hit ratio: {logger_hit_ratio:.4f}   "
                        f"accesses: {logger_access}   hits: {logger_hit}   misses: {logger_miss}"
                    )
        
        # Per-core statistics — uncomment below to enable per-core hit/miss output
        # content.append("----------------------------------------")
        # content.append("Per-Core Statistics:")
        # for core_id in range(self.num_cores):
        #     core_hits = sum(batch[0] for batch in self.core_access_results[core_id])
        #     core_miss = sum(batch[1] for batch in self.core_access_results[core_id])
        #     core_total = core_hits + core_miss
        #     core_hit_ratio = core_hits / core_total if core_total > 0 else 0
        #     content.append(
        #         f"[Core {core_id}] Hit ratio: {core_hit_ratio:.4f}   "
        #         f"Accesses: {core_total}   Hits: {core_hits}   Misses: {core_miss}"
        #     )
        
        print_styled_box("On-Chip Memory Stats", content)